"""Capabilities namespace — read view over the capability registry + lifecycle actions.

Capabilities are in-process integration plugins (mail, calendar, contacts) held
in a registry, not a CRUD resource. This namespace exposes a read view (list /
detail / status) over that registry plus two lifecycle action endpoints (setup /
disconnect) that mutate connection state and credentials. Read responses are
DTO-typed through the foundation boundary; lifecycle actions return 204 no-body
on success. Credentials are write-only — they appear ONLY on the inbound
:class:`CapabilitySetup` body and are never echoed on any response.

Authentication: all routes require an authenticated session or Bearer token
(:func:`~api.auth.require_auth`).
"""

import logging
from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.vault_service import VaultLockedError
from .auth import require_auth
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.capability import CapabilityDetail, CapabilityStatus, CapabilitySummary
from .dto.capability_setup import CapabilitySetup

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Protocol

    class _Capability(Protocol):
        def get_manifest(self) -> dict[str, object]: ...
        def is_connected(self) -> bool: ...
        def load_credential(self, key: str) -> object: ...
        def configure(self, credentials: object) -> None: ...
        def connect(self) -> bool: ...
        def disconnect(self) -> None: ...
        def monitor(self) -> None: ...

logger = logging.getLogger(__name__)

_ERR_LIST = "Failed to list capabilities"
_ERR_DETAIL = "Internal error fetching capability"
_ERR_STATUS = "Internal error fetching capability status"
_ERR_SETUP = "Internal error during capability setup"
_ERR_DISCONNECT = "Internal error during capability disconnect"
_ERR_VAULT = "Vault is locked — please log out and log back in"
_ERR_CONNECTION = "Connection failed — check credentials and try again"

capabilities_ns = Namespace("capabilities", description="Capability management", path="/api/capabilities")

register_dto(
    capabilities_ns,
    CapabilitySummary,
    CapabilityDetail,
    CapabilityStatus,
    CapabilitySetup,
    Error,
)

_C = capabilities_ns.models


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_caps() -> "Mapping[str, object]":
    """Imported lazily to avoid circular imports during application boot."""
    from capabilities import load_capabilities
    return load_capabilities()


def _get_last_sync_at(cap_id: str) -> str | None:
    """Read the stored last-sync timestamp; returns ``None`` on any read error."""
    try:
        from services.tool_config_service import ToolConfigService

        svc = ToolConfigService()
        config = svc.get_tool_config(cap_id)
        return config.get(f"{cap_id}:last_sync_at")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[capabilities] Could not read last_sync_at for '%s': %s", cap_id, exc)
        return None


def _manifest(cap: "_Capability", cap_id: str) -> dict[str, object]:
    """Read the capability manifest, defaulting name/version."""
    manifest = cap.get_manifest()
    manifest.setdefault("name", cap_id)
    manifest.setdefault("version", "")
    manifest.setdefault("providers", [])
    return manifest


def _summary_dto(cap_id: str, cap: "_Capability") -> CapabilitySummary:
    """Build a :class:`CapabilitySummary` from a registry entry."""
    manifest = _manifest(cap, cap_id)
    return CapabilitySummary(
        id=cap_id,
        name=cast(str, manifest["name"]),
        version=cast(str, manifest["version"]),
        connected=cap.is_connected(),
        last_sync_at=_get_last_sync_at(cap_id),
        providers=cast("list[str]", manifest["providers"]),
        fields=cast("list[dict[str, object]] | None", manifest.get("fields")),
    )


# ---------------------------------------------------------------------------
# Read view
# ---------------------------------------------------------------------------

@capabilities_ns.route("")
class CapabilityListResource(Resource):
    @require_auth
    @capabilities_ns.response(200, "All capabilities", model=[_C["CapabilitySummary"]])
    @capabilities_ns.response(500, _ERR_LIST, model=_C["Error"])
    @responds(CapabilitySummary, code=200)
    def get(self) -> list[CapabilitySummary] | ResponseReturnValue:
        """List all registered capabilities with manifest metadata and connection state."""
        try:
            caps = _load_caps()
            return [_summary_dto(cap_id, cast("_Capability", cap)) for cap_id, cap in caps.items()]
        except Exception as exc:
            logger.exception("[capabilities] list_capabilities error: %s", exc)
            return error(_ERR_LIST, 500)


@capabilities_ns.route("/<cap_id>")
@capabilities_ns.param("cap_id", "Capability identifier")
class CapabilityDetailResource(Resource):
    @require_auth
    @capabilities_ns.response(200, "Capability details", model=_C["CapabilityDetail"])
    @capabilities_ns.response(404, "Capability not found", model=_C["Error"])
    @capabilities_ns.response(500, _ERR_DETAIL, model=_C["Error"])
    @responds(CapabilityDetail, code=200)
    def get(self, cap_id: str) -> CapabilityDetail | ResponseReturnValue:
        """Get a capability's detail — password fields are excluded, only non-secret stored values."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return error(f"Capability not found: {cap_id}", 404)

            cap = cast("_Capability", caps[cap_id])
            manifest = _manifest(cap, cap_id)
            fields = cast("list[dict[str, object]]", manifest.get("fields") or [])

            config: dict[str, object] = {}
            if cap.is_connected():
                for field in fields:
                    if field.get("type") == "password":
                        continue
                    key = f"{cap_id}:{field['name']}"
                    val = cap.load_credential(key)
                    if val is not None:
                        config[cast(str, field["name"])] = val

            return CapabilityDetail(
                id=cap_id,
                name=cast(str, manifest["name"]),
                version=cast(str, manifest["version"]),
                connected=cap.is_connected(),
                last_sync_at=_get_last_sync_at(cap_id),
                config=config,
            )
        except Exception as exc:
            logger.exception("[capabilities] get_capability('%s') error: %s", cap_id, exc)
            return error(_ERR_DETAIL, 500)


@capabilities_ns.route("/<cap_id>/status")
@capabilities_ns.param("cap_id", "Capability identifier")
class CapabilityStatusResource(Resource):
    @require_auth
    @capabilities_ns.response(200, "Capability status", model=_C["CapabilityStatus"])
    @capabilities_ns.response(404, "Capability not found", model=_C["Error"])
    @capabilities_ns.response(500, _ERR_STATUS, model=_C["Error"])
    @responds(CapabilityStatus, code=200)
    def get(self, cap_id: str) -> CapabilityStatus | ResponseReturnValue:
        """Get connectivity status for a capability."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return error(f"Capability not found: {cap_id}", 404)

            cap = cast("_Capability", caps[cap_id])

            return CapabilityStatus(
                id=cap_id,
                connected=cap.is_connected(),
                last_sync_at=_get_last_sync_at(cap_id),
            )
        except Exception as exc:
            logger.exception("[capabilities] capability_status('%s') error: %s", cap_id, exc)
            return error(_ERR_STATUS, 500)


# ---------------------------------------------------------------------------
# Lifecycle actions (return no resource → 204 no body)
# ---------------------------------------------------------------------------

@capabilities_ns.route("/<cap_id>/setup")
@capabilities_ns.param("cap_id", "Capability identifier")
class CapabilitySetupResource(Resource):
    @require_auth
    @capabilities_ns.expect(_C["CapabilitySetup"])
    @capabilities_ns.response(204, "Capability connected")
    @capabilities_ns.response(403, _ERR_VAULT, model=_C["Error"])
    @capabilities_ns.response(404, "Capability not found", model=_C["Error"])
    @capabilities_ns.response(422, "Bad credentials", model=_C["Error"])
    @capabilities_ns.response(500, _ERR_SETUP, model=_C["Error"])
    @responds(code=204)
    @expects(CapabilitySetup)
    def post(self, cap_id: str, dto: CapabilitySetup) -> None | ResponseReturnValue:
        """Connect a capability with credentials; fires an initial sync on a daemon thread."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return error(f"Capability not found: {cap_id}", 404)

            cap = cast("_Capability", caps[cap_id])
            try:
                cap.configure(dto.model_dump(exclude_none=True))
            except ValueError as exc:
                logger.warning("[capabilities] setup '%s' configure failed: %s", cap_id, exc)
                return error("Invalid capability configuration", 422)

            if not cap.is_connected():
                return error(_ERR_CONNECTION, 422)

            import threading

            def _first_sync() -> None:
                try:
                    cap.monitor()
                    logger.info("[capabilities] initial sync complete for '%s'", cap_id)
                except Exception as sync_exc:
                    logger.warning("[capabilities] initial sync failed for '%s': %s", cap_id, sync_exc)

            threading.Thread(target=_first_sync, daemon=True, name=f"cap-init-{cap_id}").start()
            return None
        except VaultLockedError:
            logger.warning("[capabilities] setup '%s' failed: vault is locked", cap_id)
            return error(_ERR_VAULT, 403)
        except Exception as exc:
            logger.exception("[capabilities] setup_capability('%s') error: %s", cap_id, exc)
            return error(_ERR_SETUP, 500)


@capabilities_ns.route("/<cap_id>/disconnect")
@capabilities_ns.param("cap_id", "Capability identifier")
class CapabilityDisconnectResource(Resource):
    @require_auth
    @capabilities_ns.response(204, "Capability disconnected")
    @capabilities_ns.response(404, "Capability not found", model=_C["Error"])
    @capabilities_ns.response(500, _ERR_DISCONNECT, model=_C["Error"])
    @responds(code=204)
    def post(self, cap_id: str) -> None | ResponseReturnValue:
        """Disconnect a capability."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return error(f"Capability not found: {cap_id}", 404)
            cast("_Capability", caps[cap_id]).disconnect()
            return None
        except Exception as exc:
            logger.exception("[capabilities] disconnect_capability('%s') error: %s", cap_id, exc)
            return error(_ERR_DISCONNECT, 500)
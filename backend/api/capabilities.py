"""
Capabilities blueprint — /api/capabilities endpoints for managing capability plugins.

Capabilities are self-contained integrations (e.g. mail/calendar/contacts) that can
be connected, disconnected, and queried through this REST interface. The blueprint
delegates all business logic to the objects returned by
:func:`~capabilities.load_capabilities`. Capability tools are surfaced to the LLM via
Ability subclasses (email, calendar, contacts) which are discovered by AbilityRegistry
at startup.

Authentication
--------------
All routes require an authenticated session or a valid Bearer token (enforced via
the :func:`~api.auth.require_auth` decorator).

Blueprint prefix
----------------
``/api/capabilities``
"""

import logging
from typing import TYPE_CHECKING, cast

from flask import request
from flask.typing import ResponseReturnValue

from flask_restx import Namespace, Resource
from services.vault_service import VaultLockedError
from .auth import require_auth

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

capabilities_ns = Namespace("capabilities", description="Capability management", path="/api/capabilities")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_caps() -> "Mapping[str, object]":
    """Imported lazily to avoid circular imports during application boot."""
    from capabilities import load_capabilities
    return load_capabilities()


def _get_last_sync_at(cap_id: str) -> str | None:
    """Returns ``None`` on any read error."""
    try:
        from services.database_service import get_shared_db_service
        from services.tool_config_service import ToolConfigService

        db = get_shared_db_service()
        svc = ToolConfigService(db)
        config = svc.get_tool_config(cap_id)
        return config.get(f"{cap_id}:last_sync_at")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[capabilities] Could not read last_sync_at for '%s': %s", cap_id, exc)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@capabilities_ns.route("")
@capabilities_ns.response(200, "List of capabilities")
class CapabilityListResource(Resource):
    @require_auth
    def get(self) -> ResponseReturnValue:
        try:
            caps = _load_caps()
            result = []
            for cap_id, cap in caps.items():
                manifest = cast("_Capability", cap).get_manifest()
                result.append({
                    "id": cap_id,
                    "name": manifest.get("name", cap_id),
                    "version": manifest.get("version", ""),
                    "connected": cast("_Capability", cap).is_connected(),
                    "last_sync_at": _get_last_sync_at(cap_id),
                    "providers": manifest.get("providers", []),
                    "fields": manifest.get("fields"),
                })
            return {"capabilities": result}, 200

        except Exception as exc:
            logger.exception("[capabilities] list_capabilities error: %s", exc)
            return {"error": "Failed to list capabilities"}, 500


@capabilities_ns.route("/<cap_id>")
@capabilities_ns.response(200, "Capability details")
@capabilities_ns.response(404, "Capability not found")
@capabilities_ns.param("cap_id", "str", "Capability identifier")
class CapabilityDetailResource(Resource):
    @require_auth
    def get(self, cap_id: str) -> ResponseReturnValue:
        """Password fields are excluded — only non-secret fields are returned."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return {"error": f"Capability not found: {cap_id}"}, 404

            cap = cast("_Capability", caps[cap_id])
            manifest = cap.get_manifest()
            fields = cast("list[dict[str, object]]", manifest.get("fields") or [])

            config = {}
            if cap.is_connected():
                for field in fields:
                    if field.get("type") == "password":
                        continue
                    key = f"{cap_id}:{field['name']}"
                    val = cap.load_credential(key)
                    if val is not None:
                        config[field["name"]] = val

            return {
                "id": cap_id,
                "name": manifest.get("name", cap_id),
                "version": manifest.get("version", ""),
                "connected": cap.is_connected(),
                "last_sync_at": _get_last_sync_at(cap_id),
                "config": config,
            }, 200

        except Exception as exc:
            logger.exception("[capabilities] get_capability('%s') error: %s", cap_id, exc)
            return {"error": "Internal error fetching capability"}, 500


@capabilities_ns.route("/<cap_id>/setup")
@capabilities_ns.response(200, "Capability connected")
@capabilities_ns.response(400, "Setup failed")
@capabilities_ns.response(401, "Vault locked")
@capabilities_ns.response(404, "Capability not found")
@capabilities_ns.param("cap_id", "str", "Capability identifier")
class CapabilitySetupResource(Resource):
    @require_auth
    def post(self, cap_id: str) -> ResponseReturnValue:
        """Connect a capability with credentials."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return {"error": f"Capability not found: {cap_id}"}, 404

            cap = cast("_Capability", caps[cap_id])
            credentials = request.get_json(force=True) or {}

            try:
                cap.configure(credentials)
            except ValueError as exc:
                logger.warning("[capabilities] setup '%s' configure failed: %s", cap_id, exc)
                return {"error": str(exc)}, 400

            if not cap.is_connected():
                return {"error": "Connection failed — check credentials and try again"}, 400

            import threading

            def _first_sync() -> None:
                try:
                    cap.monitor()
                    logger.info("[capabilities] initial sync complete for '%s'", cap_id)
                except Exception as sync_exc:
                    logger.warning("[capabilities] initial sync failed for '%s': %s", cap_id, sync_exc)

            threading.Thread(target=_first_sync, daemon=True, name=f"cap-init-{cap_id}").start()

            return {"status": "connected"}, 200

        except ValueError as exc:
            logger.warning("[capabilities] setup '%s' ValueError: %s", cap_id, exc)
            return {"error": str(exc)}, 400
        except VaultLockedError:
            logger.warning("[capabilities] setup '%s' failed: vault is locked", cap_id)
            return {"error": "Vault is locked — please log out and log back in"}, 401
        except Exception as exc:
            logger.exception("[capabilities] setup_capability('%s') error: %s", cap_id, exc)
            return {"error": "Internal error during capability setup"}, 500


@capabilities_ns.route("/<cap_id>/disconnect")
@capabilities_ns.response(200, "Capability disconnected")
@capabilities_ns.response(404, "Capability not found")
@capabilities_ns.param("cap_id", "str", "Capability identifier")
class CapabilityDisconnectResource(Resource):
    @require_auth
    def post(self, cap_id: str) -> ResponseReturnValue:
        """Disconnect a capability."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return {"error": f"Capability not found: {cap_id}"}, 404

            cap = cast("_Capability", caps[cap_id])

            cap.disconnect()
            return {"status": "disconnected"}, 200

        except Exception as exc:
            logger.exception("[capabilities] disconnect_capability('%s') error: %s", cap_id, exc)
            return {"error": "Internal error during capability disconnect"}, 500


@capabilities_ns.route("/<cap_id>/status")
@capabilities_ns.response(200, "Capability status")
@capabilities_ns.response(404, "Capability not found")
@capabilities_ns.param("cap_id", "str", "Capability identifier")
class CapabilityStatusResource(Resource):
    @require_auth
    def get(self, cap_id: str) -> ResponseReturnValue:
        """Get connectivity and error status for a capability."""
        try:
            caps = _load_caps()
            if cap_id not in caps:
                return {"error": f"Capability not found: {cap_id}"}, 404

            cap = cast("_Capability", caps[cap_id])

            error_count = 0
            try:
                from services.database_service import get_shared_db_service
                from services.tool_config_service import ToolConfigService

                db = get_shared_db_service()
                svc = ToolConfigService(db)
                config = svc.get_tool_config(cap_id)
                raw_count = config.get(f"{cap_id}:error_count")
                if raw_count is not None:
                    error_count = int(raw_count)
            except Exception as ec_exc:
                logger.debug("[capabilities] could not read error_count for '%s': %s", cap_id, ec_exc)

            return {
                "id": cap_id,
                "connected": cap.is_connected(),
                "last_sync_at": _get_last_sync_at(cap_id),
                "error_count": error_count,
            }, 200

        except Exception as exc:
            logger.exception("[capabilities] capability_status('%s') error: %s", cap_id, exc)
            return {"error": "Internal error fetching capability status"}, 500

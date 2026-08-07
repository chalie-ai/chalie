"""Capabilities endpoint — read view over the capability registry, migrated from the legacy namespace.

Capabilities are in-process integration plugins (mail, calendar, contacts) held
in a registry, not a CRUD resource. This endpoint exposes the read view (list /
detail) over that registry; the lifecycle actions (status / setup / disconnect)
that mutate connection state and credentials live under
``api/actions/capabilities/`` and share the helpers defined here — this module
is their one source of truth, not just the endpoint's. Credentials are
write-only — they appear ONLY on the inbound
:class:`~api.request.capabilities.CapabilitySetupRequest` body and are never
echoed on any response.

- GET /api/capabilities/all   → get_all (paginated listing)
- GET /api/capabilities/<id>  → get

No POST/DELETE surface — the base 405 default stands (a partial Endpoint
with no create/delete).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import NotFoundError
from api.request import Request
from api.response.capabilities import CapabilityDetailResponse, CapabilitySummaryResponse

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


# ---------------------------------------------------------------------------
# Shared helpers — used by this endpoint AND api/actions/capabilities/*.
# ---------------------------------------------------------------------------

class CapabilitiesEndpoint(Endpoint):
    """Read view for the capability registry — no create/delete surface."""

    id_type: ClassVar[type[int] | type[str]] = str
    request_dto: ClassVar[type[Request] | None] = None
    response_dto = {
        "get_all": DocumentedResponse(CapabilitySummaryResponse, listing=True),
        "get": DocumentedResponse(CapabilityDetailResponse),
    }

    @staticmethod
    def _load_caps() -> "Mapping[str, object]":
        """Load the capability registry.

        Imported lazily — importing ``capabilities`` at module scope breaks a
        boot-time circular import (verified: the legacy module this replaces,
        ``api/capabilities.py``, carried the identical lazy import for the same
        stated reason).
        """
        from capabilities import load_capabilities
        return load_capabilities()

    @staticmethod
    def _get_last_sync_at(cap_id: str) -> str | None:
        """Read the stored last-sync timestamp; returns ``None`` on any read error."""
        try:
            from models.tool_config import ToolConfig
            return ToolConfig.get_value(cap_id, f"{cap_id}:last_sync_at")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[capabilities] Could not read last_sync_at for '%s': %s", cap_id, exc)
            return None

    @staticmethod
    def _manifest(cap: "_Capability", cap_id: str) -> dict[str, object]:
        """Read the capability manifest, defaulting name/version/providers."""
        manifest = cap.get_manifest()
        manifest.setdefault("name", cap_id)
        manifest.setdefault("version", "")
        manifest.setdefault("providers", [])
        return manifest

    @staticmethod
    def _summary_dto(cap_id: str, cap: "_Capability") -> CapabilitySummaryResponse:
        """Build a :class:`CapabilitySummaryResponse` from a registry entry."""
        manifest = CapabilitiesEndpoint._manifest(cap, cap_id)
        return CapabilitySummaryResponse(
            id=cap_id,
            name=cast(str, manifest["name"]),
            version=cast(str, manifest["version"]),
            connected=cap.is_connected(),
            last_sync_at=CapabilitiesEndpoint._get_last_sync_at(cap_id),
            providers=cast("list[str]", manifest["providers"]),
            fields=cast("list[dict[str, object]] | None", manifest.get("fields")),
        )

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List every registered capability with manifest metadata and connection state."""
        caps = CapabilitiesEndpoint._load_caps()
        items = [CapabilitiesEndpoint._summary_dto(cap_id, cast("_Capability", cap)) for cap_id, cap in caps.items()]
        total = len(items)
        start = (page - 1) * limit
        return CapabilitySummaryResponse.listing(
            items[start : start + limit], page=page, limit=limit, total=total
        )

    def get(self, id: int | str) -> ResponseReturnValue:
        """Get a capability's detail — password fields excluded, non-secret stored values only.

        Raises:
            NotFoundError: capability id not found.
        """
        cap_id = cast(str, id)
        caps = CapabilitiesEndpoint._load_caps()
        if cap_id not in caps:
            raise NotFoundError(f"Capability not found: {cap_id}")

        cap = cast("_Capability", caps[cap_id])
        manifest = CapabilitiesEndpoint._manifest(cap, cap_id)
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

        return CapabilityDetailResponse(
            id=cap_id,
            name=cast(str, manifest["name"]),
            version=cast(str, manifest["version"]),
            connected=cap.is_connected(),
            last_sync_at=CapabilitiesEndpoint._get_last_sync_at(cap_id),
            config=config,
        ).single()

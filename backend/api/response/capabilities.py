"""Response DTOs for the capabilities endpoint group — read view over the capability registry.

Capabilities are in-process integration plugins (mail, calendar, contacts) held
in a registry, not a CRUD resource: there is no create/delete surface, only a
read view (list/detail) plus lifecycle actions (status/setup/disconnect) under
``api/actions/capabilities/``.

``last_sync_at`` is a pre-formatted ISO string read straight from storage (not
a ``datetime``), so it stays ``str | None`` and the foundation datetime
serializer does not apply. No response DTO here ever carries a password/
credential field — the detail route excludes ``type=="password"`` manifest
fields upstream when building ``config``.
"""

from __future__ import annotations

from .response import Response


class CapabilitySummaryResponse(Response):
    """One row in the capability list — manifest metadata plus connection state."""

    id: str
    name: str
    version: str
    connected: bool
    last_sync_at: str | None
    providers: list[str]
    fields: list[dict[str, object]] | None


class CapabilityDetailResponse(Response):
    """Detail view: non-secret stored field values only (passwords excluded)."""

    id: str
    name: str
    version: str
    connected: bool
    last_sync_at: str | None
    config: dict[str, object]


class CapabilityStatusResponse(Response):
    """Connectivity status for a capability."""

    id: str
    connected: bool
    last_sync_at: str | None

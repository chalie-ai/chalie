"""Response DTOs for the policies endpoint — migrated from the legacy namespace."""

from __future__ import annotations

from datetime import datetime

from .response import Response


class PolicyResponse(Response):
    """Read shape for one policy triple, enriched for the Brain UI."""

    channel: str
    permission: str
    setting: str
    label: str
    group: str | None = None


class BlockedEntryResponse(Response):
    """Read shape for one blocked-action log entry."""

    action_id: int
    context: str
    reason: str
    created_at: datetime

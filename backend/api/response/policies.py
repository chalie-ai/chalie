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

    # A permission name (e.g. "bash.modify_file"), not a row id: the DDL column
    # is TEXT and PolicyBlockedLog derives its Policy link from
    # ``action_id == permission``.
    action_id: str
    context: str
    reason: str
    created_at: datetime

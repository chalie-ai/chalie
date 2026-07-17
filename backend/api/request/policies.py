"""Request DTOs for the policies endpoint — migrated from the legacy namespace."""

from __future__ import annotations

from pydantic import Field

from .request import Request


class PolicyUpsertRequest(Request):
    """Inbound body for a single-cell policy upsert."""

    channel: str
    permission: str = Field(..., min_length=1, max_length=200)
    setting: str


class PolicyRespondRequest(Request):
    """Inbound body resolving an in-process permission gate."""

    request_id: str = Field(..., min_length=1)
    approved: bool

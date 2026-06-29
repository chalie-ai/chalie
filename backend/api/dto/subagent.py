"""DTOs for subagent / turn-control endpoints."""

from __future__ import annotations

from .base import DTO


class Interrupted(DTO):
    """200 ack for POST /chat/interrupt and POST /chat/stop."""

    ok: bool = True
    interrupted: bool | None = None
    reason: str | None = None


class SubagentStopResult(DTO):
    """200 ack for POST /chat/subagent/<sub_id>/stop."""

    ok: bool = True
    cancelled: bool | None = None
    reason: str | None = None


class ActiveSubagents(DTO):
    """GET /chat/subagents/active body.

    Elements are free-form ``delegate.snapshot()`` dicts — no typed element DTO.
    """

    subagents: list[dict[str, object]]

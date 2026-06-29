"""DTOs for subagent / turn-control endpoints."""

from __future__ import annotations

from .base import DTO


class Interrupted(DTO):
    """200 ack for DELETE /api/thread/<turn_id> (turn interrupt). ``interrupted``
    is True when a live turn was cancelled, else ``reason`` is ``no_active_turn``."""

    ok: bool = True
    interrupted: bool | None = None
    reason: str | None = None


class SubagentStopResult(DTO):
    """200 ack for DELETE /api/subagent/<sub_id> (subagent cancel). ``cancelled``
    is True when a live delegate was stopped, else ``reason`` is ``not_found``."""

    ok: bool = True
    cancelled: bool | None = None
    reason: str | None = None


class ActiveSubagents(DTO):
    """GET /api/subagents body.

    Elements are free-form ``delegate.snapshot()`` dicts — no typed element DTO.
    """

    subagents: list[dict[str, object]]

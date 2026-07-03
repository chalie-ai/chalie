"""DTO for the turn_executions lifecycle row.

``started_at``/``ended_at`` are persisted as ISO-8601 strings at write time;
pydantic parses them to ``datetime`` and the foundation serializer re-emits
canonical ISO-8601 UTC, so handlers never call ``.isoformat()``.
"""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class TurnExecutionDTO(DTO):
    """One turn's DB-backed lifecycle row. POST and DELETE /api/thread/<turn_id>
    both return this shape, and it is exactly the payload the turn's
    ExecutionTracker broadcasts on every state flip — one shape, so no surface
    ever has to reconcile two versions of "what a turn_execution looks like"."""

    id: int
    channel: str
    type: str | None
    turn_id: int
    started_at: datetime
    ended_at: datetime | None
    cancel_requested: bool
    state: str
    stop_reason: str | None

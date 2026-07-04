"""DTO for the blocked-action log read shape.

``created_at`` is persisted as an ISO-8601 string at write time; pydantic parses
it to ``datetime`` and the foundation serializer re-emits canonical ISO-8601 UTC,
so handlers never call ``.isoformat()``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import DTO


class BlockedEntry(DTO):
    """One row of the blocked-action log."""

    action_id: str
    context: str
    reason: str
    created_at: datetime


class BlockedQuery(DTO):
    """Inbound query for the blocked-log list — capped ``limit``, default 50."""

    limit: int = Field(default=50, ge=1, le=500)
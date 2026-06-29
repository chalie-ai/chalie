"""DTO for ``GET /system/observability/errors`` — recent ERROR/CRITICAL log lines."""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class LogError(DTO):
    """One ERROR/CRITICAL log entry — timestamp + message."""

    timestamp: str
    message: str


class ErrorsList(DTO):
    """Recent error log lines (newest first), capped, with a generated-at timestamp."""

    generated_at: datetime
    errors: list[LogError]
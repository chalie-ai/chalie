"""DTO for ``GET /system/observability/write-queue`` — write-queue runtime stats."""

from __future__ import annotations

from .base import DTO


class WriteQueueStats(DTO):
    """Snapshot of the write queue: backlog, completed writes, and error count."""

    queue_size: int
    processed: int
    errors: int
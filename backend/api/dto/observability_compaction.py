"""DTO for ``GET /system/observability/compaction`` — the continuity-compaction view.

The empty state serializes to exactly ``{"compaction": null}`` (a protected test
does whole-body equality), so ``CompactionView`` carries only the ``compaction``
field. ``compacted_at`` is a UI-formatted string from ``locale_service.format_date``
(e.g. ``'2026-01-01 00:00'`` — space-separated, no ``T``/offset), NOT an ISO
datetime, so it stays ``str`` and the foundation datetime serializer does not apply.
"""

from __future__ import annotations

from .base import DTO


class CompactionRecord(DTO):
    """One durable continuity-compaction summary for the chat channel."""

    summary: str
    compacted_up_to_id: int
    compacted_at: str


class CompactionView(DTO):
    """The compaction view — ``null`` when no compaction has run yet."""

    compaction: CompactionRecord | None
"""DTOs for the Auto Research observability endpoints — list + detail views.

Run rows/detail are service-owned (the discovery-runs service shapes them), so the
``run``/``runs`` fields stay raw passthrough dicts. ``generated_at`` serializes as
ISO-8601 UTC via the foundation serializer.
"""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class ResearchRunsList(DTO):
    """Newest-first list of proactive-research runs."""

    generated_at: datetime
    runs: list[dict[str, object]]


class ResearchRunDetail(DTO):
    """One Auto Research run — grounding + full transcript output."""

    generated_at: datetime
    run: dict[str, object]
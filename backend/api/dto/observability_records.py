"""DTO for ``GET /api/system/observability/records`` — paginated record browser.

Rows are polymorphic: episodes rows carry ``location``; user/system rows carry
``key``. The row shape stays a raw ``dict`` (not a nested DTO) so both variants
round-trip without a union. ``generated_at`` serializes as ISO-8601 UTC via the
foundation serializer. The invalid-source/offset path is a preserved ``400`` (not
normalized to 422 — a protected test pins it), so it is handled in the handler,
not on a ``Field`` constraint.
"""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class RecordsPage(DTO):
    """One page of the record browser across episodes / user / system sources."""

    generated_at: datetime
    source: str
    rows: list[dict[str, object]]
    offset: int
    limit: int
    returned: int
    has_more: bool
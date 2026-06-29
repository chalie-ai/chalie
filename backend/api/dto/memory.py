"""DTOs for the memory namespace — the search/retrieval HTTP contract.

Memory search fans out across two independent stores (episodic memory + the data
graph) and merges their heterogeneous rows into one ranked result set. There is
no id-addressed resource here, only a read-only search endpoint; query
validation (trim + non-empty) lives on the ``StringConstraints``, so the handler
never hand-validates. Datetimes serialize as ISO-8601 UTC via
:class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import StringConstraints

from .base import DTO


class MemorySearchQuery(DTO):
    """Inbound query string for memory search — a non-empty, trimmed ``q``."""

    q: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MemoryHit(DTO):
    """One merged search-result row — episode or concept, told apart by ``type``.

    The two stores emit different row shapes; this superset carries a ``type``
    discriminator and leaves the per-kind fields nullable, so one model covers
    both without a ``oneOf`` union.
    """

    type: str
    content: str
    score: float
    confidence: float | None = None
    created_at: datetime | None = None


class MemorySearchResponse(DTO):
    """Ranked, merged result set across episodic memory + the data graph."""

    results: list[MemoryHit]

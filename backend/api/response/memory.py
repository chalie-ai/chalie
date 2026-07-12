"""Response DTO for the memory endpoint — migrated from the legacy namespace."""

from __future__ import annotations

from datetime import datetime

from .response import Response


class MemoryHitResponse(Response):
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

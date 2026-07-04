"""DTO for a rich-media segment — a span of a parsed assistant message."""

from __future__ import annotations

from .base import DTO


class Segment(DTO):
    """One span of a rich-media-parsed message: plain text or a tool card.

    Text spans carry ``content``; rich spans carry ``tag``/``synthesis``/``payload``.
    ``payload`` is free-form per-tool render data (Raw) and is never re-validated.
    """

    type: str
    content: str | None = None
    tag: str | None = None
    synthesis: str | None = None
    payload: object | None = None

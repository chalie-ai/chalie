"""DTO for an attachment re-rendered on a user message after refresh."""

from __future__ import annotations

from .base import DTO


class Attachment(DTO):
    """A file linked to a transcript row, with its inline-preview URL."""

    filename: str
    mime_type: str
    is_image: bool
    url: str

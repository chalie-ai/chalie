"""DTO for one transcript row in the conversation feed."""

from __future__ import annotations

from .attachment import Attachment
from .base import DTO
from .chip import Chip
from .segment import Segment


class Message(DTO):
    """One transcript row.

    Role-conditional: ``attachments`` appears on user rows only;
    ``tool_calls``/``segments`` on non-user rows only. ``timestamp`` is a
    pre-formatted locale string, not a datetime.
    """

    id: str
    role: str
    content: str
    timestamp: str
    turn_id: int | None = None
    attachments: list[Attachment] | None = None
    tool_calls: list[Chip] | None = None
    segments: list[Segment] | None = None

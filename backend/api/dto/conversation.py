"""DTOs for the conversation feed — query params and the paginated response."""

from __future__ import annotations

from pydantic import Field

from .base import DTO
from .message import Message


class RecentConversationQuery(DTO):
    """Query params for the recent-conversation feed.

    Absent params fall back to the field defaults; present-but-invalid values
    are rejected with 422.
    """

    limit: int = Field(default=12, ge=1, le=120)
    offset: int = Field(default=0, ge=0)


class RecentConversationResponse(DTO):
    """Paginated conversation feed (turn-level, not row-level)."""

    messages: list[Message]
    has_more: bool
    turns_returned: int

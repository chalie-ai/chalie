"""DTOs for the moments resource — the pin/list/forget/search HTTP contract.

A *moment* is a user-curated bookmark of one assistant transcript turn. The read
shape synthesizes a client-facing ``key`` (``moment_<transcript_id>``) and exposes
both ``value`` and ``message_text`` (its alias) — the frontend reads both today.
Datetimes serialize as ISO-8601 UTC via :class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from .base import DTO


class Moment(DTO):
    """Read shape for one pinned moment."""

    id: int
    transcript_id: int
    key: str
    value: str
    message_text: str
    created_at: datetime


class MomentCreate(DTO):
    """Inbound body to pin a moment — exactly one of transcript_id / message_text."""

    transcript_id: int | None = None
    message_text: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> Self:
        has_id = self.transcript_id is not None
        has_text = bool(self.message_text and self.message_text.strip())
        if has_id == has_text:
            raise PydanticCustomError(
                "value_error", "Exactly one of transcript_id or message_text is required", None
            )
        return self


class MomentSearch(DTO):
    """Query for lexical + semantic moment search."""

    q: str

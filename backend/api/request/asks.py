"""Request DTOs for the asks endpoints."""

from __future__ import annotations

from pydantic import Field, StrictBool

from .request import Request


class AskReplyRequest(Request):
    """Inbound body answering an open ask."""

    value: StrictBool | None = Field(
        ...,
        description="The answer: true or false (allowed) for a permission ask, null (done) for a hand-off.",
    )

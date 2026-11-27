"""Request DTOs for the asks endpoints."""

from __future__ import annotations

from pydantic import Field, StrictBool

from services.ask_user import CodeReply, QuestionReply
from .request import Request


class AskReplyRequest(Request):
    """Inbound body answering an open ask."""

    value: StrictBool | QuestionReply | CodeReply | None = Field(
        ...,
        description=(
            "The answer: true or false (allowed) for a permission ask, null (done) for a hand-off, "
            "for a question either {\"labels\": [...]} (the answers picked) or {\"other\": \"...\"} (the user's own text), "
            "and {\"code\": \"...\"} (the 6 digits typed) for a code ask."
        ),
    )

"""Request DTOs for the asks endpoints."""

from __future__ import annotations

from pydantic import Field, StrictBool

from services.ask_user import QuestionReply
from .request import Request


class AskReplyRequest(Request):
    """Inbound body answering an open ask."""

    value: StrictBool | QuestionReply | None = Field(
        ...,
        description=(
            "The answer: true or false (allowed) for a permission ask, null (done) for a hand-off, "
            "and for a question either {\"labels\": [...]} (the answers picked) or {\"other\": \"...\"} (the user's own text)."
        ),
    )

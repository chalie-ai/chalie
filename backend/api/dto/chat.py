"""DTOs for the chat namespace — request bodies and synchronous ack responses."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .base import DTO


class ChatRequest(DTO):
    """Form-data fields for POST /chat (files are handled separately via request.files)."""

    text: str | None = None
    source: str = "text"
    echo_id: str = ""


class Accepted(DTO):
    """202 synchronous ack for POST /chat and POST /action."""

    status: str = "accepted"


class ActionRequest(DTO):
    """JSON body for POST /action.  Unknown extra keys are forwarded as tool params."""

    model_config = ConfigDict(extra="allow")

    skill: str = Field(..., min_length=1)

    @property
    def params(self) -> dict[str, object]:
        """Extra fields beyond ``skill``, forwarded verbatim to ToolDispatcher."""
        return dict(self.model_extra or {})

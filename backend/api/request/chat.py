"""Request DTO for the chat namespace — the action-button dispatch body."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .request import Request


class ActionRequest(Request):
    """POST /api/chat/action body: a skill name plus an open bag of tool params.

    Unlike every other Request (which inherits ``extra="forbid"``), this DTO
    deliberately sets ``extra="allow"``: unknown top-level keys are not errors
    but the skill's parameters, forwarded verbatim to the dispatcher.
    """

    model_config = ConfigDict(extra="allow")

    skill: str = Field(..., min_length=1)

    @property
    def params(self) -> dict[str, object]:
        """Extra keys beyond ``skill``, forwarded verbatim to the ToolDispatcher."""
        return dict(self.model_extra or {})

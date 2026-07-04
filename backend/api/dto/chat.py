"""DTOs for the chat namespace — the action request body and its synchronous result."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .base import DTO


class ActionRequest(DTO):
    """JSON body for POST /action.  Unknown extra keys are forwarded as tool params."""

    model_config = ConfigDict(extra="allow")

    skill: str = Field(..., min_length=1)

    @property
    def params(self) -> dict[str, object]:
        """Extra fields beyond ``skill``, forwarded verbatim to ToolDispatcher."""
        return dict(self.model_extra or {})


class ActionResult(DTO):
    """200 result for POST /action — the dispatched skill's rendered output.

    A rich-card action is a fixed-confidence ACT-mode mutation, so ``mode`` and
    ``confidence`` are constant; ``content`` is the sanitized tool output and
    ``duration_ms`` the dispatch wall-time.
    """

    content: str
    mode: str = "ACT"
    confidence: float = 0.95
    duration_ms: int

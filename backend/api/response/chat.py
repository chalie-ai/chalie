"""Response DTO for the chat namespace — the action-button dispatch result."""

from __future__ import annotations

from .response import Response


class ActionResult(Response):
    """POST /api/chat/action result — the dispatched skill's rendered output.

    A rich-card action is a fixed-confidence ACT-mode mutation, so ``mode`` and
    ``confidence`` are constant; ``content`` is the sanitized tool output and
    ``duration_ms`` the dispatch wall-time.
    """

    content: str
    mode: str = "ACT"
    confidence: float = 0.95
    duration_ms: int

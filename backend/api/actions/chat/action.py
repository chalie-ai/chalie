"""Action-button dispatch — POST /api/chat/action.

A rich-card control click runs its bound skill through an inert
MessageProcessor and returns the rendered result inline. A card action is a
silent state mutation, not a conversation turn: it never persists to the
transcript and crosses no WS frame, so the calling card resolves its
optimistic update straight from this HTTP response.
"""

from __future__ import annotations

import time
from typing import cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.chat import ActionRequest
from api.response.chat import ActionResult
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from exceptions import EndpointError
from services.markup import sanitize
from services.processor_config import ProcessorConfig


class _ActionButtonConfig(ProcessorConfig):
    """Minimal ProcessorConfig for action-button dispatches — no ACT loop runs.

    prompt_service returns "" for the action_button channel, so no prompt
    builders are needed; the skip_* flags and broadcast_to=None keep the
    dispatch silent (no transcript row, no input row, no WS frame).
    """

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.ACTION_BUTTON.value,
            role="action_button",
            policy_channel=PolicyChannel.CHAT,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )


class ChatAction(Action):
    """Dispatch a rich-card control click and return its rendered result inline."""

    request_dto = ActionRequest
    response_dto = {"post": DocumentedResponse(ActionResult)}

    def slug(self) -> str:
        return "chat"

    def verb(self) -> str:
        return "action"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Run the skill and return its result inline — the WS bus stays signal-only."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        body = cast("ActionRequest", data)
        started = time.time()

        # Inert: no process()/begin(), so zero DB/WS side-effects at construction.
        # The config's skip_* flags + broadcast_to=None keep dispatch silent, and
        # with no turn_executions row should_stop() fails open (there is no running
        # turn a stop button could reach).
        mp = MessageProcessor(_ActionButtonConfig())
        result_text = mp.dispatch_service.dispatch(body.skill, body.params)
        if result_text.startswith("Unknown tool:"):
            raise EndpointError(f"Unknown skill: {body.skill}")

        return ActionResult(
            content=sanitize(result_text or "Done."),
            duration_ms=int((time.time() - started) * 1000),
        ).single()

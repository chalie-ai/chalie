# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageParentAbility — a subagent's message to the agent that spawned it.

Held only by a subagent the person's chat spawned (:func:`holds`): every other
caller waits for its subagent, so nothing is free to answer a message. The
processor offers it on such a subagent's turn itself, never through a config,
since one config runs a subagent whichever caller spawned it.

The message is delivered to the chat (:meth:`MessageProcessor.deliver`): it
opens a new chat turn, or queues behind the one working. Its row is linked to
this call's own ``tool_calls`` id — the subagent's replies are linked to the
spawning call, which is how the two are told apart. The call returns at once;
the chat's answer, sent with ``message_subagent``, opens the subagent's next
turn. A turn whose last call is this one, delivered
(:func:`ended_on_message_parent`), posts no reply and ends its subagent
waiting for that answer; each chat turn,
as it opens, names the subagents then waiting whose message has reached the
chat (:func:`waiting_reminder`).
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import UNKNOWN_TOOL, Ability
from abilities._agents import label
from abilities._result import ToolResult
from abilities.message_subagent import MessageSubagentAbility
from configs.enums.channels import Channel
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import MessageParentParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.agent_service import AgentService

_SENT = "Message sent."
_FOLLOW_UP = (
    "You'll be invoked again once the reply lands. "
    "If there is nothing else you can do without the answer just stop."
)


class MessageParentAbility(Ability[MessageParentParams]):
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "message_parent"
    PARAMS: ClassVar[type[ParamBag] | None] = MessageParentParams
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.message,)}

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.message: {"type": "string", "description": "What to send."},
        },
        "required": [Keys.message],
    }

    def get_summary(self) -> str:
        return "Send a message to the agent that spawned you. You are invoked again when its reply lands."

    def get_examples(self) -> list[str]:
        return [
            "ask which account to use",
            "ask for the missing file path",
            "report progress on the task",
            "ask whether to include last year's results",
            "check which of two readings of the task is meant",
            "say the search has found nothing so far",
        ]

    def get_search_tooltip(self) -> str:
        return self.get_summary()

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def get_follow_up(self, tr: ToolResult) -> str:
        return _FOLLOW_UP

    def run(self, params: MessageParentParams) -> ToolResult:
        # Lazy: the chat's config and the controller both load this module.
        from configs.channels import UserConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        if not holds(mp.agent):
            return ToolResult.err(UNKNOWN_TOOL.format(tool_name=self.NAME), code="unknown-tool")
        if self.tool_call_id is None:
            raise RuntimeError(f"{self.NAME}: {mp.channel} turn {mp.turn_id} ran it without a tool_calls row")
        MessageProcessor.deliver(UserConfig(), params.message, self.tool_call_id)
        return ToolResult.ok(_SENT)


WAITING_REMINDER = (
    "When this turn began, these subagents were waiting for your reply: {agents}. "
    f"Reply with `{MessageSubagentAbility.NAME}` so they can continue."
)

_SPAWNED = "You have been spawned by agent {turn_id} to perform the task; {prompt}. "
_ASK = (
    f"You can ask for more context if needed by using `{MessageParentAbility.NAME}`. "
    "This will send a message to the agent that spawned you; you are invoked again when its reply lands. Else, finish"
)
_FINISH = " the task naturally and your final output will be sent to the agent that spawned you automatically."


def holds(agent: Agent | None) -> bool:
    """Whether the turn of ``agent`` holds ``message_parent`` — ``agent`` being
    ``None`` on a turn that is not a subagent's. Only a subagent the person's
    chat spawned does: the one caller that does not wait for it."""
    return agent is not None and agent.spawned_by_channel == Channel.USER


def spawn_message(agent: Agent, prompt: str) -> str:
    """The task as the subagent ``agent`` reads it: who spawned it, the
    ``prompt`` it was given, and how its answer gets back — with
    ``message_parent`` offered only when it holds it (:func:`holds`)."""
    ask = _ASK if holds(agent) else "Finish"
    return _SPAWNED.format(turn_id=agent.spawned_in_turn, prompt=prompt) + ask + _FINISH


def _origin(call: ToolCall) -> Transcript:
    """The subagent's row that made the ``message_parent`` ``call`` — there
    for as long as the call is."""
    origin = Transcript.get(call.transcript_id)
    if origin is None or origin.turn_id is None:
        raise RuntimeError(f"{call.tool_name} call {call.id}: its row {call.transcript_id} is gone or has no turn")
    return origin


def sender(call: ToolCall) -> Agent:
    """The subagent that made the ``message_parent`` ``call`` — there for as
    long as the call is: a subagent's record is never deleted, and the call
    goes only with its row (:func:`_origin`)."""
    origin = _origin(call)
    agent = AgentService.of_channel(origin.channel)
    if agent is None:
        raise RuntimeError(f"{call.tool_name} call {call.id}: {origin.channel} has no subagent")
    return agent


def ended_on_message_parent(channel: str, turn_id: int) -> bool:
    """Whether turn ``turn_id`` on ``channel`` ended by sending its caller a
    message: its last tool call is a ``message_parent`` that delivered (a
    refused or failed one sent nothing, so there is no reply to wait for) —
    the one test of a turn that ended asking its caller, so its subagent waits
    for the reply and posts none."""
    calls = ToolCall.by_turn(channel, turn_id)
    return bool(calls) and calls[-1].tool_name == MessageParentAbility.NAME and calls[-1].state == ToolCall.DONE


def waiting_reminder(channel: str, opener: int | None) -> str:
    """:data:`WAITING_REMINDER` naming, oldest first, every subagent
    ``channel`` spawned that waits for its reply and whose message has reached
    ``channel`` — less an archived one, which no tool can reply to any more
    (:meth:`AgentService.archived`), and the one whose message, the
    ``tool_calls`` id ``opener``, opened this turn, since that message already
    asks for the reply — else ``''``. A waiting subagent's last call is the
    message it waits on (:func:`ended_on_message_parent`)."""
    agents = [
        agent
        for agent in (
            Agent.filter("status", Agent.WAITING)
            .filter("spawned_by_channel", channel)
            .order_by("agent_id ASC")
            .get()
        )
        if not AgentService.archived(agent)
        and (call := ToolCall.by_turn(agent.channel, agent.turn_id)[-1]).id != opener
        and Transcript.filter("channel", channel).filter("tool_call_id", str(call.id)).exists()
    ]
    if not agents:
        return ""
    return WAITING_REMINDER.format(agents=", ".join(label(agent) for agent in agents))

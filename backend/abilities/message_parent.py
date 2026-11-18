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
turn. A subagent whose turn's latest call is this one waits for that answer
(:func:`waiting`) — while it finishes that turn and once it has ended, when
it posts no reply for that turn — and is named, for the chat's model, after
every message the person sends the chat (:func:`waiting_notice`).
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
from models.queued_message import QueuedMessage
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


PARENT_NUDGE = (
    "This subagent is waiting for your reply. "
    f"Reply with `{MessageSubagentAbility.NAME}` so it can continue."
)

WAITING_NOTICE = (
    "Subagents waiting for your reply: {agents}. "
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


def _awaited(agent: Agent) -> ToolCall | None:
    """The ``message_parent`` call whose answer ``agent`` still waits for:
    the latest call of its current turn, which is still running or ended
    done, and no message the chat sent it since waits in its queue. ``None``
    when it waits for none."""
    if agent.status not in (Agent.RUNNING, Agent.DONE):
        return None
    calls = ToolCall.by_turn(agent.channel, agent.turn_id)
    if not calls or calls[-1].tool_name != MessageParentAbility.NAME:
        return None
    call = calls[-1]
    if QueuedMessage.filter("channel", agent.channel).filter("tool_call_id", call.id, ">").exists():
        return None
    return call


def waiting(row: Transcript) -> Agent | None:
    """The subagent whose ``message_parent`` call posted chat ``row``, while
    it still waits for the answer to that call (:func:`_awaited`). ``None``
    for any other row."""
    link = row.tool_call_id
    if link is None or not link.isdecimal():
        return None
    call = ToolCall.get(int(link))
    if call is None or call.tool_name != MessageParentAbility.NAME:
        return None
    agent = sender(call)
    awaited = _awaited(agent)
    return agent if awaited is not None and awaited.id == call.id else None


def waiting_for(channel: str) -> list[Agent]:
    """Every subagent ``channel`` spawned that waits for an answer
    (:func:`_awaited`) to a message that has reached ``channel``, oldest
    first — only the chat's ever do (:func:`holds`)."""
    live = (
        Agent.filter_in("status", (Agent.RUNNING, Agent.DONE))
        .filter("spawned_by_channel", channel)
        .order_by("agent_id ASC")
        .get()
    )
    return [
        agent for agent in live
        if (call := _awaited(agent)) is not None
        and Transcript.filter("channel", channel).filter("tool_call_id", str(call.id)).exists()
    ]


def waiting_notice(channel: str) -> str:
    """What follows a message the person sent ``channel``, in the model's
    prompt: a newline and :data:`WAITING_NOTICE` naming every subagent still
    waiting for a reply (:func:`waiting_for`), else ``''``. Derived on every
    read, never stored, so the message keeps the person's own text."""
    agents = waiting_for(channel)
    if not agents:
        return ""
    return "\n" + WAITING_NOTICE.format(agents=", ".join(label(agent) for agent in agents))


def parent_nudge(row: Transcript) -> str:
    """What follows chat ``row`` in the model's prompt: a newline and
    :data:`PARENT_NUDGE` while ``row`` is a subagent's message and that
    subagent still waits for a reply (:func:`waiting`), else ``''``. Derived
    on every read, never stored."""
    return f"\n{PARENT_NUDGE}" if waiting(row) is not None else ""

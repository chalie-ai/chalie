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

The message is posted to the chat turn that spawned the subagent, exactly as
its late reply is (:func:`post_to_caller`), but linked to this call's own
``tool_calls`` id — the reply is linked to the spawning call, which is how the
two are told apart. The call then waits for the next message to join the
subagent's turn — the chat's ``message_subagent`` — and returns it as its
result with :data:`SUBAGENT_NUDGE`. A message taken here is never filed for
the act trail, so it reaches the model once. A stop ends the wait with no
reply — and stopping the chat turn the message opened or joined stops the
subagent waiting on it (:func:`waiting_on`). While any subagent waits, every
message the person sends the chat reads, for the chat's model, followed by a
line naming each one (:func:`waiting_notice`).
"""

from __future__ import annotations

import time
from typing import ClassVar, cast

from abilities._ability import UNKNOWN_TOOL, Ability
from abilities._agents import label
from abilities._delegate import late_caller, post_to_caller
from abilities._result import ToolResult
from abilities.message_subagent import MessageSubagentAbility
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS, Channel
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import MessageParentParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.agent_service import AgentService

# How often the wait checks for a reply or a stop — the ask gate's own beat.
_POLL_SECONDS = 0.25

_STOPPED = "The message was sent. The wait was stopped before a reply arrived."


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
        return "Send a message to the agent that spawned you and wait for its reply."

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

    def run(self, params: MessageParentParams) -> ToolResult:
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        if not holds(mp.agent):
            return ToolResult.err(UNKNOWN_TOOL.format(tool_name=self.NAME), code="unknown-tool")
        resolved = late_caller(mp.channel, mp.turn_id)
        if resolved is None or self.tool_call_id is None:
            raise RuntimeError(f"{self.NAME}: {mp.channel} turn {mp.turn_id} has no chat turn to message")
        post_to_caller(resolved[0], params.message, self.tool_call_id)
        while not (rows := mp.take_joined()):
            if mp.turn_execution_service.should_stop():
                return ToolResult.ok(_STOPPED)
            time.sleep(_POLL_SECONDS)
        replies = "\n\n".join(row.content for row in rows)
        return ToolResult.ok(f"{replies}\n{SUBAGENT_NUDGE}")


PARENT_NUDGE = (
    "This subagent is waiting for your reply. "
    f"Reply with `{MessageSubagentAbility.NAME}` so it can continue."
)

WAITING_NOTICE = (
    "Subagents waiting for your reply: {agents}. "
    f"Reply with `{MessageSubagentAbility.NAME}` so they can continue."
)

SUBAGENT_NUDGE = (
    "Do not acknowledge this message. "
    f"Use `{MessageParentAbility.NAME}` again only to report progress or to ask for clarification. "
    "Otherwise, complete the task."
)

_SPAWNED = "You have been spawned by agent {turn_id} to perform the task; {prompt}. "
_ASK = (
    f"You can ask for more context if needed by using `{MessageParentAbility.NAME}`. "
    "This will send a message to the agent that spawned you and wait for its reply. Else, finish"
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
    agent = AgentService.on_turn(origin.channel, cast("int", origin.turn_id))
    if agent is None:
        raise RuntimeError(f"{call.tool_name} call {call.id}: {origin.channel} turn {origin.turn_id} has no subagent")
    return agent


def waiting(row: Transcript) -> Agent | None:
    """The subagent whose ``message_parent`` call posted chat ``row``, while
    it still waits for a reply: the call is still running, so is its
    subagent, and no message has joined the subagent's turn since ``row``.
    ``None`` for any other row."""
    link = row.tool_call_id
    if link is None or not link.isdecimal():
        return None
    call = ToolCall.get(int(link))
    if call is None or call.tool_name != MessageParentAbility.NAME or call.state != ToolCall.STARTED:
        return None
    agent = sender(call)
    if agent.status != Agent.RUNNING:
        return None
    if Transcript.joined_since(Channel.for_delegate(agent.type), agent.turn_id, cast("int", row.id)):
        return None
    return agent


def waiting_on(channel: str, turn_id: int) -> list[Agent]:
    """Every subagent whose message sits in turn ``(channel, turn_id)`` —
    the chat turn it opened or joined — and still waits for a reply
    (:func:`waiting`): the turn it waits on, so stopped with it."""
    rows = Transcript.filter("channel", channel).filter("turn_id", turn_id).get()
    return [agent for row in rows if (agent := waiting(row)) is not None]


def waiting_for(channel: str) -> list[Agent]:
    """Every subagent ``channel`` spawned whose message still waits for a
    reply (:func:`waiting`), oldest first — only the chat's ever do
    (:func:`holds`)."""
    found: list[Agent] = []
    for agent in Agent.running():
        if agent.spawned_by_channel != channel:
            continue
        for call in ToolCall.by_turn(Channel.for_delegate(agent.type), agent.turn_id):
            if call.tool_name != MessageParentAbility.NAME or call.state != ToolCall.STARTED:
                continue
            row = Transcript.filter("channel", channel).filter("tool_call_id", str(call.id)).first()
            if row is not None and waiting(row) is not None:
                found.append(agent)
                break
    return found


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


def joined_nudge(row: Transcript) -> str:
    """What follows ``row``, a message joined into a running turn, in that
    turn's act trail: a newline and :data:`SUBAGENT_NUDGE` when the turn is a
    subagent's — only the chat sends one (``message_subagent``), and only
    to a subagent it spawned, which holds ``message_parent`` — else ``''``."""
    if row.channel not in WATCHABLE_DELEGATE_CHANNELS:
        return ""
    return f"\n{SUBAGENT_NUDGE}"

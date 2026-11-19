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

Every text a subagent and its caller read each other's messages through lives
here, one constant per text. None is ever stored. :data:`WAITING_REMINDER` is
added after a chat turn's prompt
(:meth:`~services.prompt_service.PromptService._user_prompt`); every other
text only as a turn's input is read
(:meth:`~services.prompt_service.PromptService._task`), the input row keeping
the bare text, so later history shows it alone. There are four cases:

1. A subagent's first turn reads its task in :data:`SPAWN_TASK`
   (:func:`first_turn`): the task, the first row of the turn that spawned it as
   context (:data:`SPAWN_CONTEXT`), and how its answer gets back —
   :data:`SPAWN_INSTRUCTIONS_ASK` when it holds this tool,
   :data:`SPAWN_INSTRUCTIONS_FINISH` when it does not.
2. The chat turn a message sent with this tool opens reads that message in
   :data:`SUBAGENT_MESSAGE`, which asks the chat to reply (:func:`chat_turn`).
   Later chat history shows the message alone, under its speaker label.
3. A subagent's later turn whose previous turn delivered a message with this
   tool reads the chat's answer beside the last such message, in
   :data:`PARENT_RESPONSE` (:func:`later_turn`).
4. Any other later turn, a steer, reads the chat's message word for word.
"""

from __future__ import annotations

import json
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

#: A subagent's first turn: its task, the context it was spawned in
#: (:data:`SPAWN_CONTEXT`, or nothing when there is none) and its instructions.
SPAWN_TASK = "## Task\n{task}\n\n{context}## Instructions\n{instructions}"

#: The first turn's Context section: the first row of the turn that spawned it.
SPAWN_CONTEXT = (
    "## Context\n"
    "Below is a snippet of what the parent agent is working on so you can better orient yourself on this task;\n"
    "{context}\n\n"
)

#: The first turn's instructions for a subagent that holds ``message_parent``.
SPAWN_INSTRUCTIONS_ASK = (
    "You have been spawned by another agent to perform the `task` above. "
    "Perform this ask to the best of your ability. "
    f"If you need to ask questions or provide status updates use the `{MessageParentAbility.NAME}`. "
    "Only provide updates and ask questions if you take decisions or change anything from the provided plan "
    "of action. If the task is clear and you can complete it exactly as described simply finish it and your "
    "response will automatically reach the agent that spawned you."
)

#: The first turn's instructions for a subagent that does not hold
#: ``message_parent``.
SPAWN_INSTRUCTIONS_FINISH = (
    "You have been spawned by another agent to perform the `task` above. "
    "Perform this ask to the best of your ability. "
    "Simply finish it and your response will automatically reach the agent that spawned you."
)

#: A subagent's later turn when the chat answers what it sent.
PARENT_RESPONSE = (
    "## What you sent to the parent agent\n{sent}\n\n"
    "## Parent response\n{response}\n\n"
    "## Instructions\n"
    "Use the response to continue your work if anything else is left pending. "
    "Do not acknowledge the receipt of this message, simply carry on whatever is left if any. "
    "Your final response will be delivered automatically to the parent agent."
)

#: The chat turn a subagent's ``message_parent`` message opens.
SUBAGENT_MESSAGE = (
    "Agent {agent_id} has sent the following message; {message}.\n---\n\n"
    f"Reply to the agent by using the {MessageSubagentAbility.NAME} to acknowledge the receipt of this message. "
    "If they have questions or you need to provide additional context, do so. "
    "If everything is clear and no follow up is needed simply acknowledge receipt."
)


def holds(agent: Agent | None) -> bool:
    """Whether the turn of ``agent`` holds ``message_parent`` — ``agent`` being
    ``None`` on a turn that is not a subagent's. Only a subagent the person's
    chat spawned does: the one caller that does not wait for it."""
    return agent is not None and agent.spawned_by_channel == Channel.USER


def first_turn(agent: Agent, task: str) -> str:
    """The ``task`` the subagent ``agent`` was spawned with, as its first turn
    reads it (:data:`SPAWN_TASK`): with the first row of the turn that spawned
    it as context, read by its id (``agent.context_id``), and how its answer
    gets back — ``message_parent`` offered only when it holds it
    (:func:`holds`). A spawning turn that wrote no input row (a hidden-input
    turn, such as the one disclosing an outside agent's exchange) has no
    context to give, so the Context section is left out."""
    context = ""
    if agent.context_id is not None:
        row = Transcript.get(agent.context_id)
        if row is None:
            raise RuntimeError(f"subagent #{agent.agent_id}: its context row {agent.context_id} is gone")
        context = SPAWN_CONTEXT.format(context=row.content)
    instructions = SPAWN_INSTRUCTIONS_ASK if holds(agent) else SPAWN_INSTRUCTIONS_FINISH
    return SPAWN_TASK.format(task=task, context=context, instructions=instructions)


def later_turn(channel: str, previous_turn: int, message: str) -> str:
    """The chat's ``message`` as the subagent on ``channel`` reads it on a
    later turn, ``previous_turn`` being its turn before this one: beside the
    last message that turn delivered with ``message_parent``
    (:data:`PARENT_RESPONSE`), since the chat is answering it; or, when that
    turn delivered none, word for word — a steer."""
    sent = [
        call
        for call in ToolCall.by_turn(channel, previous_turn)
        if call.tool_name == MessageParentAbility.NAME and call.state == ToolCall.DONE
    ]
    if not sent:
        return message
    return PARENT_RESPONSE.format(sent=json.loads(sent[-1].params)[Keys.message], response=message)


def chat_turn(opener: int | None, message: str) -> str:
    """``message`` as the chat reads it on the turn it opened, ``opener``
    being the ``tool_calls`` id its input row is linked to: a subagent's
    ``message_parent`` message in :data:`SUBAGENT_MESSAGE`, naming the
    sender's ``agent_id`` (:func:`sender`) and asking for the reply; any other
    message — the person's own, or a delegate's late reply — as it is."""
    call = (
        None
        if opener is None
        else ToolCall.filter("id", opener).filter("tool_name", MessageParentAbility.NAME).first()
    )
    if call is None:
        return message
    return SUBAGENT_MESSAGE.format(agent_id=sender(call).agent_id, message=message)


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

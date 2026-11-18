# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageSubagentAbility — send a message to a subagent the chat spawned.

The message is delivered to the subagent's own channel
(:meth:`MessageProcessor.deliver`), on the config its next turn runs on
(:func:`agent_config`): a finished subagent picks up at once in its next
turn, with all its earlier turns in its history, and a running one gets it as
the turn after the one working. Its row carries this call's own
``tool_calls`` id, and its turn the chat turn's origin, so a tool that asks
for the person's approval there asks on the chat, as at its spawn. The result
is the same either way, naming the label the subagent's answer will come back
under. A subagent that ended cancelled or failed opens no turn again:
:meth:`MessageProcessor.deliver` checks that in the transaction it queues or
opens in, so a stop from another thread of the chat lands wholly before the
check or wholly after the message, and on :class:`~exceptions.AgentStopped`
the call returns what the subagent last gave
(:meth:`AgentService.last_output`), read off the row that check read.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._agents import AgentAbility, label
from abilities._delegate import agent_config, reply_label
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentMessageParams
from contracts.params.param_bag import ParamBag
from exceptions import AgentStopped
from services.agent_service import AgentService

_SENT = 'Message sent to subagent {label}. Its reply will reach you as "{reply}".'


class MessageSubagentAbility(AgentAbility[AgentMessageParams]):
    NAME: ClassVar[str] = "message_subagent"
    PARAMS: ClassVar[type[ParamBag] | None] = AgentMessageParams
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.agent_id, Keys.message)}
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("message agent", "tell subagent", "reply to subagent")

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.agent_id: {
                "type": "integer",
                "description": "The agent_id of a subagent you spawned, running or finished.",
            },
            Keys.message: {"type": "string", "description": "What to tell the subagent."},
        },
        "required": [Keys.agent_id, Keys.message],
    }

    def get_summary(self) -> str:
        return (
            "Send a message to a subagent you spawned, or reply to one it sent you. "
            "A finished subagent picks up again with all its earlier work."
        )

    def get_examples(self) -> list[str]:
        return [
            "tell the subagent to focus on prices only",
            "send agent 2 the file path it asked for",
            "reply to the subagent's question",
            "let the web search know I only want results from this year",
            "tell the code agent to skip the tests",
            "answer the browsing agent: use the second account",
        ]

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: AgentMessageParams) -> ToolResult:
        # Lazy: the registry and the controller both load this module.
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        agent = self.owned(params.agent_id)
        if isinstance(agent, ToolResult):
            return agent
        try:
            MessageProcessor.deliver(agent_config(agent), params.message, self.tool_call_id, self.bound().origin)
        except AgentStopped as stopped:
            # A cancelled or failed agent's last output is always its notice.
            return ToolResult.err(cast("str", AgentService.last_output(stopped.agent)), code="agent-stopped")
        return ToolResult.ok(_SENT.format(label=label(agent), reply=reply_label(agent.type, agent.agent_id)))

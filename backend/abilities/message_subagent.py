# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageSubagentAbility — send a message into a running subagent's turn.

The message joins the subagent's own turn (:meth:`MessageProcessor.join`) on
its own channel, never opening another. It is written as an input row on the
config the subagent was spawned on — read from the one place that chooses it,
:meth:`DelegateAbility.child_config` — bound to that channel, carrying
the ``tool_calls`` id of the call that spawned it, as the subagent's first row
does, so the row belongs to that call wherever the call is followed (its stop,
its pill, its reply). The running loop takes the message in at its next
boundary. A subagent that is no longer running gets how it ended instead
(:func:`ended`).
"""

from __future__ import annotations

from typing import ClassVar

from abilities._agents import AgentAbility, ended, label, not_found
from abilities._delegate import DelegateAbility
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentMessageParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from services.agent_service import AgentService


class MessageSubagentAbility(AgentAbility[AgentMessageParams]):
    NAME: ClassVar[str] = "message_subagent"
    PARAMS: ClassVar[type[ParamBag] | None] = AgentMessageParams
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.agent_id, Keys.message)}
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("message agent", "tell subagent", "reply to subagent")

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.agent_id: {"type": "integer", "description": "The subagent's agent_id."},
            Keys.message: {"type": "string", "description": "What to tell the subagent."},
        },
        "required": [Keys.agent_id, Keys.message],
    }

    def get_summary(self) -> str:
        return "Send a message to a running subagent, or reply to a message it sent you."

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
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        agent = self.owned(params.agent_id)
        if agent is None:
            return not_found(params.agent_id)
        if agent.status == Agent.RUNNING:
            spawner = AbilityRegistry.get(agent.type)
            if not isinstance(spawner, DelegateAbility):
                raise RuntimeError(f"{self.NAME}: subagent #{agent.agent_id}'s type {agent.type!r} is not a delegate")
            config = spawner.child_config(mp.config)
            if config is None:
                raise RuntimeError(f"{self.NAME}: {agent.type} has no config for a {mp.channel} caller")
            joined = MessageProcessor(config.on_channel(agent.channel), agent.turn_id).join(
                params.message, spine_only=True, tool_call_id=AgentService.spawn_call_id(agent),
            )
            if joined is not None:
                return ToolResult.ok(f"Message sent to subagent {label(agent)}.")
            # Nothing to join: the subagent ended before the message landed.
            agent = Agent.by_id(agent.agent_id)
            if agent is None or agent.status == Agent.RUNNING:
                raise RuntimeError(f"{self.NAME}: subagent #{params.agent_id} is running but has no turn to join")
        return ended(agent)

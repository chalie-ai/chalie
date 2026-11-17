# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AgentStatusAbility — one subagent's status and the last thing it wrote.

The last thing it wrote is the newest assistant row of its own turn that says
something (:meth:`Transcript.last_said`) — its final reply once it is done,
else its latest words so far; ``None`` while it has written nothing, and once
its turn number belongs to a newer subagent (a purged channel numbers its turns
from the start again, so its own rows are gone). Only a subagent the chat
spawned is read (:meth:`AgentAbility.owned`).
"""

from __future__ import annotations

from typing import ClassVar

from abilities._agents import AgentAbility, not_found
from abilities._result import ToolResult
from configs.enums.channels import Channel
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentIdParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from models.transcript import Transcript


class AgentStatusAbility(AgentAbility[AgentIdParams]):
    NAME: ClassVar[str] = "agent_status"
    PARAMS: ClassVar[type[ParamBag] | None] = AgentIdParams
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.agent_id,)}
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("subagent status", "check agent", "agent progress")

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.agent_id: {"type": "integer", "description": "The subagent's agent_id."},
        },
        "required": [Keys.agent_id],
    }

    def get_summary(self) -> str:
        return "Get a subagent's status and the last thing it wrote."

    def get_examples(self) -> list[str]:
        return [
            "how is the subagent I started doing?",
            "check on agent 3",
            "has my background search finished yet?",
            "what has the code agent written so far?",
            "is the subtask still running?",
            "get the status of the browsing agent",
        ]

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: AgentIdParams) -> ToolResult:
        agent = self.owned(params.agent_id)
        if agent is None:
            return not_found(params.agent_id)
        return ToolResult.ok({
            "agent_id": agent.agent_id,
            "type": agent.type,
            "status": agent.status,
            "last_output": _last_output(agent),
        })


def _last_output(agent: Agent) -> str | None:
    owner = Agent.by_turn(agent.type, agent.turn_id)
    if owner is None or owner.agent_id != agent.agent_id:
        return None
    return Transcript.last_said(Channel.for_delegate(agent.type), agent.turn_id)

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AgentStatusAbility — one subagent's status and what it last gave.

What it last gave is :meth:`AgentService.last_output`: the crashed or the
cancelled text once it failed or was stopped, else the reply its latest
finished turn ended on — ``None`` before its first turn has finished. Only a
subagent the chat spawned, and not archived, is read
(:meth:`AgentAbility.owned`).
"""

from __future__ import annotations

from typing import ClassVar

from abilities._agents import AgentAbility
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentIdParams
from contracts.params.param_bag import ParamBag
from services.agent_service import AgentService


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
        if isinstance(agent, ToolResult):
            return agent
        return ToolResult.ok({
            "agent_id": agent.agent_id,
            "type": agent.type,
            "status": agent.status,
            "last_output": AgentService.last_output(agent),
        })

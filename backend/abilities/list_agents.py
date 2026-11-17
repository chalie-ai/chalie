# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ListAgentsAbility — the subagents the chat spawned that are still running.

Each entry is drawn the way the interface's agent list draws it
(:meth:`AgentService.view`): its number, its type, the one-line summary of the
call that spawned it, and its status. With none running — always so for any
caller other than the chat (:meth:`AgentAbility.mine`) — it says so in words.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._agents import AgentAbility
from abilities._result import ToolResult
from contracts.params.agent_params_bag import ListAgentsParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from services.agent_service import AgentService

_FIELDS = ("agent_id", "type", "summary", "status")
_NONE_RUNNING = "No agents are currently running"


class ListAgentsAbility(AgentAbility[ListAgentsParams]):
    NAME: ClassVar[str] = "list_agents"
    PARAMS: ClassVar[type[ParamBag] | None] = ListAgentsParams
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("list subagents", "running agents", "my agents")

    _PARAMETERS: ClassVar[dict[str, object]] = {"type": "object", "properties": {}}

    def get_summary(self) -> str:
        return (
            "List the subagents you spawned that are still running, with each one's "
            "agent_id, type, summary and status."
        )

    def get_examples(self) -> list[str]:
        return [
            "which subagents are still running?",
            "list the agents I started",
            "are any of my background agents still working?",
            "show me the subagents in progress",
            "what agents have I spawned that haven't finished?",
            "find the agent_id of the running web search",
        ]

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ListAgentsParams) -> ToolResult:
        running = [AgentService.view(agent) for agent in Agent.running() if self.mine(agent)]
        if not running:
            return ToolResult.ok(_NONE_RUNNING)
        return ToolResult.ok([{key: view[key] for key in _FIELDS} for view in running])

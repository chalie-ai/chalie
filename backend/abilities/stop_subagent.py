# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""StopSubagentAbility — stop a subagent the chat spawned.

A stop always ends the subagent cancelled, whatever its status. A working one
goes through :meth:`TurnExecutionService.cancel_turn`, the one cancel path,
stamped :data:`STOP_REASON_PARENT_STOPPED` as when the caller's own stop
reaches it: the chat asked for the stop, so no "was stopped" reply comes back
to it, and every subagent the stopped one spawned is stopped with it. One that
already ended is moved to cancelled (:meth:`AgentService.cancel`), and the
panel is told.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._agents import AgentAbility, label, not_found
from abilities._result import ToolResult
from configs.enums.channels import Channel
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentIdParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from services.agent_service import AgentService


class StopSubagentAbility(AgentAbility[AgentIdParams]):
    NAME: ClassVar[str] = "stop_subagent"
    PARAMS: ClassVar[type[ParamBag] | None] = AgentIdParams
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.agent_id,)}
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("stop agent", "cancel subagent", "kill agent")

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.agent_id: {"type": "integer", "description": "The subagent's agent_id."},
        },
        "required": [Keys.agent_id],
    }

    def get_summary(self) -> str:
        return "Stop a running subagent."

    def get_examples(self) -> list[str]:
        return [
            "stop the subagent, I don't need it anymore",
            "cancel agent 4",
            "kill the background web search",
            "stop the code agent before it changes anything else",
            "never mind, call off that subtask",
            "abort the browsing agent",
        ]

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: AgentIdParams) -> ToolResult:
        # Lazy: the service reaches the channel configs, which load this module.
        from services.turn_execution_service import (  # noqa: PLC0415
            STOP_REASON_PARENT_STOPPED,
            TurnExecutionService,
        )

        agent = self.owned(params.agent_id)
        if agent is None:
            return not_found(params.agent_id)
        if agent.status == Agent.RUNNING and not TurnExecutionService.cancel_turn(
            Channel.for_delegate(agent.type), agent.turn_id, stop_reason=STOP_REASON_PARENT_STOPPED,
        ):
            # Nothing was cancelled: the subagent ended on its own first.
            current = Agent.by_id(agent.agent_id)
            if current is None or current.status == Agent.RUNNING:
                raise RuntimeError(f"{self.NAME}: subagent #{params.agent_id} is still running after its stop")
        cancelled = AgentService.cancel(agent.agent_id)
        if cancelled is not None:
            self.bound().announce(AgentService.frame(cancelled))
        return ToolResult.ok(f"Subagent {label(agent)} stopped.")

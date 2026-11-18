# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""StopSubagentAbility — stop a subagent the chat spawned.

A stop always ends the subagent cancelled, whatever its status. It is one
``BEGIN IMMEDIATE`` transaction, the kind every
:meth:`MessageProcessor.deliver` and :meth:`MessageProcessor.drain` also
takes, so a message another thread of the chat sends the subagent, and a turn
of its that a message opens, land wholly before or wholly after it: every
message still queued for it is deleted, so none can open a turn of its, and
its status is read and acted on in the same transaction. A working one has
its turn stamped cancelled by :meth:`TurnExecutionService.stamp_cancelled`,
the writes of the one cancel path, with :data:`STOP_REASON_PARENT_STOPPED`
as when the caller's own stop reaches it: the chat asked for the stop, so no
"was stopped" reply comes back to it. Once the stop commits,
:meth:`TurnExecutionService.propagate_cancel` sends the frames and stops
every subagent the stopped one spawned. One running with no turn open yet —
its spawn before its execution opens — raises, and the whole stop rolls
back. One that already ended is moved to cancelled
(:meth:`AgentService.cancel`), and the panel is told once that commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abilities._agents import AgentAbility, label
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.agent_params_bag import AgentIdParams
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from models.queued_message import QueuedMessage
from services.agent_service import AgentService
from services.database import Database

if TYPE_CHECKING:
    from services.turn_execution_service import CancelledTurn


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
        return "Stop a subagent you spawned. It ends cancelled, whatever its status."

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
        if isinstance(agent, ToolResult):
            return agent
        stamped: CancelledTurn | None = None
        cancelled: Agent | None = None
        with Database.transaction():
            QueuedMessage.filter("channel", agent.channel).delete()
            current = Agent.by_id(agent.agent_id)
            if current is None:
                raise RuntimeError(f"{self.NAME}: subagent #{params.agent_id} vanished during its stop")
            if current.status == Agent.RUNNING:
                stamped = TurnExecutionService.stamp_cancelled(
                    current.channel, current.turn_id, STOP_REASON_PARENT_STOPPED,
                )
                if stamped is None:
                    # Running with no open execution to stamp, as a spawn is
                    # before its execution opens: the whole stop rolls back.
                    raise RuntimeError(f"{self.NAME}: subagent #{params.agent_id} is still running after its stop")
            else:
                cancelled = AgentService.cancel(agent.agent_id)
        if stamped is not None:
            TurnExecutionService.propagate_cancel(stamped)
        elif cancelled is not None:
            self.bound().announce(AgentService.frame(cancelled))
        return ToolResult.ok(f"Subagent {label(agent)} stopped.")

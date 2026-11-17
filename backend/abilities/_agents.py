# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared base of the tools that manage the subagents a turn spawned —
``list_agents``, ``agent_status``, ``stop_subagent`` and ``message_subagent``.

Only the person's chat manages subagents, since only the chat is replied to
late (:func:`replies_late`) — every other caller waits for its subagent, so
there is nothing running for it to manage. And the chat manages only the
subagents it spawned. :meth:`AgentAbility.mine` is that one rule; a subagent
it does not cover reads exactly as one that does not exist.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from abilities._ability import Ability, B
from abilities._delegate import replies_late
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from models.agent import Agent
from models.transcript import Transcript

_SPAWN_AGAIN = "If you need to perform follow-ups you can spawn a new agent"


class AgentAbility(Ability[B], ABC):
    """Abstract (lists ``ABC`` directly), so it is never registered as a tool
    in its own right; each concrete tool fills in the getters and ``run()``."""

    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.DELEGATE

    def get_search_tooltip(self) -> str:
        return self.get_summary()

    def mine(self, agent: Agent) -> bool:
        """Whether this call may manage ``agent``: the caller is the person's
        chat, and ``agent`` is one the chat spawned."""
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        return replies_late(mp.config.type()) and agent.spawned_by_channel == mp.channel

    def owned(self, agent_id: int) -> Agent | None:
        """The subagent ``agent_id`` names when this call may manage it, else
        ``None``."""
        agent = Agent.by_id(agent_id)
        return agent if agent is not None and self.mine(agent) else None


def not_found(agent_id: int) -> ToolResult:
    """The one answer for a subagent the caller may not manage — whether it
    does not exist, another channel spawned it, or the caller is not the chat."""
    return ToolResult.err(f"No subagent with agent_id {agent_id}.", code="agent-not-found")


def label(agent: Agent) -> str:
    """``<type> #<agent_id>`` — how a subagent is named to the model."""
    return f"{agent.type} #{agent.agent_id}"


def ended(agent: Agent) -> ToolResult:
    """What a call aimed at a subagent that is no longer running gets back: its
    final response when it finished with one, else that it was stopped."""
    response = Transcript.get(agent.response_id) if agent.response_id is not None else None
    if agent.status == Agent.DONE and response is not None and response.content.strip():
        return ToolResult.err(
            f"This agent has finished working. The final response was; {response.content}. {_SPAWN_AGAIN}",
            code="agent-finished",
        )
    return ToolResult.err(f"This agent was stopped before it finished. {_SPAWN_AGAIN}", code="agent-stopped")

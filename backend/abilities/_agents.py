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
it does not cover reads exactly as one that does not exist. One of its own
that is archived (:meth:`AgentService.archived`) is managed no more: a tool
naming it answers that it was archived (:meth:`AgentAbility.owned`).
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

from abilities._ability import Ability, B
from abilities._delegate import replies_late
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from models.agent import Agent
from services.agent_service import AgentService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

#: What a management tool answers for an archived subagent, read by the model.
ARCHIVED_TEXT = "Subagent {type} #{agent_id} has been archived. Spawn a new one for this work."


class AgentAbility(Ability[B], ABC):
    """Abstract (lists ``ABC`` directly), so it is never registered as a tool
    in its own right; each concrete tool fills in the getters and ``run()``."""

    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.DELEGATE

    def get_search_tooltip(self) -> str:
        return self.get_summary()

    def bound(self) -> MessageProcessor:
        """The MessageProcessor this call runs under — every agent tool has one."""
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        return mp

    def mine(self, agent: Agent) -> bool:
        """Whether this call may manage ``agent``: the caller is the person's
        chat, and ``agent`` is one the chat spawned."""
        mp = self.bound()
        return replies_late(mp.config.type()) and agent.spawned_by_channel == mp.channel

    def owned(self, agent_id: int) -> Agent | ToolResult:
        """The subagent ``agent_id`` names when this call may manage it, else
        the error the call returns: :func:`not_found` for one it may not
        manage, ``agent-archived`` for one of its own that is archived
        (:meth:`AgentService.archived`)."""
        agent = Agent.by_id(agent_id)
        if agent is None or not self.mine(agent):
            return not_found(agent_id)
        if AgentService.archived(agent):
            return ToolResult.err(ARCHIVED_TEXT.format(type=agent.type, agent_id=agent.agent_id), code="agent-archived")
        return agent


def not_found(agent_id: int) -> ToolResult:
    """The one answer for a subagent the caller may not manage — whether it
    does not exist, another channel spawned it, or the caller is not the chat."""
    return ToolResult.err(f"No subagent with agent_id {agent_id}.", code="agent-not-found")


def label(agent: Agent) -> str:
    """``<type> #<agent_id>`` — how a subagent is named to the model."""
    return f"{agent.type} #{agent.agent_id}"


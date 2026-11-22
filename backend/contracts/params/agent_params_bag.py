"""The typed input contracts of the tools that manage the subagents a turn
spawned: ``list_agents`` takes nothing, ``agent_status`` and ``stop_subagent``
take the subagent's ``agent_id``, and ``message_subagent`` takes it with the
message to send. ``message_parent``, held by a subagent the chat spawned,
takes only the message to send back.

``agent_id`` is the number the spawn result and ``list_agents`` give — the N in
"Subagent <type> #N". It must be sent as an integer of at least 1: strict, as
every integer parameter, so digits sent as text are refused here, before
``run()``, never coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ListAgentsParams(ParamBag):
    """Empty bag — ``list_agents`` accepts no parameters."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()


@dataclass(frozen=True, slots=True)
class AgentIdParams(ParamBag):
    """``agent_id`` — the subagent ``agent_status`` reads or ``stop_subagent``
    stops."""

    agent_id: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        if isinstance(agent_id := cls.require_int(params, Keys.agent_id, lo=1), ToolResult):
            return agent_id
        return cls(agent_id=agent_id)


@dataclass(frozen=True, slots=True)
class AgentMessageParams(ParamBag):
    """``agent_id`` — the subagent to message; ``message`` — what to send it."""

    agent_id: int
    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        if isinstance(agent_id := cls.require_int(params, Keys.agent_id, lo=1), ToolResult):
            return agent_id
        if isinstance(message := cls.require_str(params, Keys.message), ToolResult):
            return message
        return cls(agent_id=agent_id, message=message)


@dataclass(frozen=True, slots=True)
class MessageParentParams(ParamBag):
    """``message`` — what a subagent sends the agent that spawned it."""

    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        if isinstance(message := cls.require_str(params, Keys.message), ToolResult):
            return message
        return cls(message=message)

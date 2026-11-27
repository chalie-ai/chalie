"""Outbound DTO for the agents endpoint group."""

from __future__ import annotations

from .response import Response


class AgentResponse(Response):
    """One ``agents`` row as plain data, plus what a surface draws it from: the
    subagent's own ``channel`` and the ``summary`` of the call that started it
    (``""`` when the call gave none). The same shape the ``agent_changed``
    frame carries."""

    agent_id: int
    spawned_by_channel: str
    spawned_in_turn: int
    type: str
    prompt: str
    status: str
    response_id: int | None
    turn_id: int
    context_id: int | None
    channel: str
    summary: str

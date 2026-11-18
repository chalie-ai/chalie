# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AgentService — an ``agents`` row's end, and the plain data a surface draws
it from.

Every way a subagent's turn ends — it answers, it is stopped, it crashes, or
the process died under it — closes the agent in the transaction that closes
that turn, through :meth:`end`, or at boot through :meth:`fail_running`, and
keeps the row the turn ended on. A stop ends one cancelled whatever its
status: a running one with its stopped turn, through :meth:`end`, one that
already ended through :meth:`cancel`. :meth:`last_output` is what the agent last
gave, worked out on read; :meth:`archived` is whether an ended one is past
managing. :meth:`view` is the row plus what a surface needs to
draw it (its channel, and the summary of the call that started it);
:meth:`frame` carries that on the wire whenever a row is written. Mp-less
(Rule-3 depth): its callers hold the transaction.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from configs.enums.channels import delegate_agent_id
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from models.ws_message import WsMessage
from services.time_utils import parse_utc, utc_now


#: What a crashed subagent last gave, read by the model.
CRASHED_TEXT = "This session has crashed, start a new one"
#: What a cancelled subagent last gave, read by the model.
CANCELLED_TEXT = "This session has been cancelled, start a new one"
#: How long after the row its latest turn ended on an ended subagent is
#: archived (:meth:`AgentService.archived`).
ARCHIVE_AFTER = timedelta(days=7)


class AgentService:
    """Ends agents with their turns, and projects them for the panel."""

    #: How a subagent's turn ending maps onto its agent's status.
    _STATUS: ClassVar[dict[str, str]] = {
        TurnExecution.COMPLETED: Agent.DONE,
        TurnExecution.CANCELLED: Agent.CANCELLED,
        TurnExecution.CRASHED: Agent.FAILED,
    }

    @classmethod
    def end(cls, channel: str, turn_id: int, state: str, response_id: int | None = None) -> Agent | None:
        """Close the agent whose current turn is ``turn_id`` on ``channel``, to
        the status the turn's ending ``state`` maps to, keeping
        ``response_id`` — the row the turn ended on: its final reply when it
        answered, its last row when it was stopped or crashed. ``None`` when
        nothing changed: the turn is not a subagent's current one, or its
        agent already ended — each turn ends its agent once."""
        agent = cls.on_turn(channel, turn_id)
        if agent is None or not Agent.end(agent.agent_id, turn_id, cls._STATUS[state], response_id):
            return None
        return Agent.by_id(agent.agent_id)

    @classmethod
    def on_turn(cls, channel: str, turn_id: int) -> Agent | None:
        """The agent whose current turn is ``turn_id`` on ``channel``, or
        ``None`` when that turn is not a subagent's current one — another turn
        on its channel (its memory step, or a turn it already ended) is
        not."""
        agent = cls.of_channel(channel)
        return agent if agent is not None and agent.turn_id == turn_id else None

    @staticmethod
    def of_channel(channel: str) -> Agent | None:
        """The agent whose own channel ``channel`` is, or ``None``."""
        agent_id = delegate_agent_id(channel)
        agent = Agent.by_id(agent_id) if agent_id is not None else None
        return agent if agent is not None and agent.channel == channel else None

    @staticmethod
    def fail_running() -> list[Agent]:
        """End every agent still running ``failed`` — at boot, when no turn is
        live, so each was cut off by the process that died under it, whether
        or not its turn ever got an execution row to sweep — keeping its
        turn's last row."""
        ended = (
            Agent.by_id(a.agent_id)
            for a in Agent.running()
            if Agent.end(a.agent_id, a.turn_id, Agent.FAILED, Transcript.last_id(a.channel, a.turn_id))
        )
        return [a for a in ended if a is not None]

    @staticmethod
    def cancel(agent_id: int) -> Agent | None:
        """End agent ``agent_id`` cancelled, whatever its status — ``None``
        when it already was."""
        return Agent.by_id(agent_id) if Agent.cancel(agent_id) else None

    @staticmethod
    def last_output(agent: Agent) -> str | None:
        """What ``agent`` last gave: the crashed text once it failed, the
        cancelled text once it was stopped, else the content of the row its
        latest ending kept — ``None`` before it has one."""
        if agent.status == Agent.FAILED:
            return CRASHED_TEXT
        if agent.status == Agent.CANCELLED:
            return CANCELLED_TEXT
        if agent.response_id is None:
            return None
        row = Transcript.get(agent.response_id)
        if row is None:
            raise RuntimeError(f"subagent #{agent.agent_id}'s response row {agent.response_id} does not exist")
        return row.content

    @staticmethod
    def archived(agent: Agent) -> bool:
        """Whether ``agent`` is archived: never while it runs; once it ended,
        when the row its latest turn ended on is gone — never kept (a NULL
        ``response_id`` matches no row) or deleted — or is older than
        :data:`ARCHIVE_AFTER`. One primary-key read."""
        if agent.status == Agent.RUNNING:
            return False
        row = Transcript.get(agent.response_id)
        return row is None or parse_utc(row.created_at) < utc_now() - ARCHIVE_AFTER

    @staticmethod
    def spawn_call_id(agent: Agent) -> int | None:
        """The ``tool_calls`` id of the call that spawned ``agent``, read off
        the first row on its channel (its task) — or ``None`` once that row is
        gone."""
        first = Transcript.filter("channel", agent.channel).order_by("id ASC").first()
        if first is None or first.tool_call_id is None:
            return None
        return int(first.tool_call_id)

    @classmethod
    def summary(cls, agent: Agent) -> str:
        """The summary the call that spawned ``agent`` gave — empty when it
        gave none or its call is gone."""
        call_id = cls.spawn_call_id(agent)
        call = ToolCall.filter("id", call_id).first() if call_id is not None else None
        return call.summary if call is not None else ""

    @classmethod
    def view(cls, agent: Agent) -> dict[str, object]:
        """The row as plain data, plus the subagent's channel and the summary
        the spawning call gave (:meth:`summary`)."""
        return {**agent.to_dict(), "channel": agent.channel, "summary": cls.summary(agent)}

    @classmethod
    def frame(cls, agent: Agent) -> WsMessage:
        """The frame sent each time an agent's row is written."""
        return WsMessage(type="agent_changed", agent=cls.view(agent))

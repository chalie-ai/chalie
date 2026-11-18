# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Agent — one ``agents`` row: a subagent some turn spawned, and how it ended.

Active-record row-model. The table's key is ``agent_id`` —
there is **no** ``id`` column — so the base ``Model.save()``/``delete()``
(id-centric) do not apply; ``__columns__`` omits ``id`` and this model carries
its own :meth:`insert`. ``type`` is the spawn tool's name and ``turn_id`` the
subagent's current turn on its own channel (:attr:`channel`) — its spawn turn
first, then each turn a message opens (:meth:`resume`). Holds no mp, calls
no service (Rule-3 depth).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self, cast

from configs.enums.channels import Channel
from models.model import Model


class Agent(Model):
    """One spawned subagent, keyed by ``agent_id`` — the number the model uses."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "agent_id", "spawned_by_channel", "spawned_in_turn", "type", "prompt",
        "status", "response_id", "turn_id",
    )

    @classmethod
    def get_table(cls) -> str:
        return "agents"

    #: The ``status`` vocabulary — two non-terminal (running, and waiting for
    #: its caller's reply), three terminal (DB CHECK).
    RUNNING: ClassVar[str] = "running"
    WAITING: ClassVar[str] = "waiting"
    DONE: ClassVar[str] = "done"
    CANCELLED: ClassVar[str] = "cancelled"
    FAILED: ClassVar[str] = "failed"

    agent_id: int
    spawned_by_channel: str
    spawned_in_turn: int
    type: str
    prompt: str
    status: str
    response_id: int | None
    turn_id: int

    @property
    def channel(self) -> str:
        """The channel this subagent runs its turns on — its own."""
        return Channel.for_agent(self.type, self.agent_id)

    def insert(self) -> Self:
        """Write this row as a new, running agent and read its ``agent_id``
        back. Never commits — the caller's ``Database.transaction()`` does."""
        self.status = self.RUNNING
        cursor = self._bound_connection().execute(
            "INSERT INTO agents (spawned_by_channel, spawned_in_turn, type, prompt, status, turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.spawned_by_channel, self.spawned_in_turn, self.type, self.prompt, self.status, self.turn_id),
        )
        self.agent_id = cast("int", cursor.lastrowid)
        self.response_id = None
        return self

    @classmethod
    def by_id(cls, agent_id: int) -> Agent | None:
        """The agent the model calls ``agent_id``, or ``None``."""
        return cls.filter("agent_id", agent_id).first()

    @classmethod
    def running(cls) -> Sequence[Agent]:
        """Every agent still running, oldest first."""
        return cls.filter("status", cls.RUNNING).order_by("agent_id ASC").get()

    @classmethod
    def active(cls) -> Sequence[Agent]:
        """Every agent running or waiting for its caller's reply, oldest
        first."""
        return cls.filter_in("status", (cls.RUNNING, cls.WAITING)).order_by("agent_id ASC").get()

    @classmethod
    def resume(cls, agent_id: int, turn_id: int) -> bool:
        """Move an agent that ended done or waiting onto its next turn,
        ``turn_id``, as running again — in the transaction that opens that
        turn. Returns whether it moved; a running, cancelled or failed agent
        never does."""
        moved = (
            cls.filter("agent_id", agent_id)
            .filter_in("status", (cls.DONE, cls.WAITING))
            .update(status=cls.RUNNING, turn_id=turn_id)
        )
        return moved > 0

    @classmethod
    def end(cls, agent_id: int, turn_id: int, status: str, response_id: int | None) -> bool:
        """Move a running agent to the ``status`` its current turn,
        ``turn_id``, ended it with — each turn ends its agent once: a turn
        that already ended it, or that is no longer its current turn, moves
        nothing, and only a stop (:meth:`cancel`) ends it again. Returns
        whether it moved."""
        moved = (
            cls.filter("agent_id", agent_id)
            .filter("turn_id", turn_id)
            .filter("status", cls.RUNNING)
            .update(status=status, response_id=response_id)
        )
        return moved > 0

    @classmethod
    def cancel(cls, agent_id: int) -> bool:
        """Move an agent to ``cancelled`` from any status — a stop always ends
        it cancelled, whatever it was. Returns whether it moved."""
        moved = cls.filter("agent_id", agent_id).filter("status", cls.CANCELLED, "!=").update(status=cls.CANCELLED)
        return moved > 0

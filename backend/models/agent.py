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
subagent's own turn on that tool's delegate channel. Turn ids are counted per
channel and start over once retention has purged a channel's newest turns, so
the pair is not unique: the newest row for it is the live one. Holds no mp,
calls no service (Rule-3 depth).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self, cast

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

    #: The ``status`` vocabulary — one non-terminal, three terminal (DB CHECK).
    RUNNING: ClassVar[str] = "running"
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
    def by_turn(cls, type_: str, turn_id: int) -> Agent | None:
        """The agent whose own turn is ``turn_id`` on ``type_``'s channel — the
        newest, since a purged channel numbers its turns from the start again."""
        return cls.filter("type", type_).filter("turn_id", turn_id).order_by("agent_id DESC").first()

    @classmethod
    def running(cls) -> Sequence[Agent]:
        """Every agent still running, oldest first."""
        return cls.filter("status", cls.RUNNING).order_by("agent_id ASC").get()

    @classmethod
    def end(cls, agent_id: int, status: str, response_id: int | None) -> bool:
        """Move a running agent to its terminal ``status`` — once: an agent
        that already ended keeps its first outcome. Returns whether it moved."""
        moved = (
            cls.filter("agent_id", agent_id)
            .filter("status", cls.RUNNING)
            .update(status=status, response_id=response_id)
        )
        return moved > 0

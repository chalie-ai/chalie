# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thread — one ``threads`` row: a conversation thread, a channel's turn.

Active-record row-model. A thread is named by its ``(channel, turn_id)`` pair,
which is unique; every transcript row the thread holds points at its ``id``.
It carries what is stored once per thread: its topical label (``summary``).
Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class Thread(Model):
    """One conversation thread, keyed by its channel and turn number."""

    __columns__: ClassVar[tuple[str, ...]] = ("id", "channel", "turn_id", "summary")

    @classmethod
    def get_table(cls) -> str:
        return "threads"

    channel: str
    turn_id: int
    summary: str | None

    @classmethod
    def ensure(cls, channel: str, turn_id: int) -> int:
        """The id of the thread for ``(channel, turn_id)``, created if it has none.

        Idempotent: a second call for the same pair — another writer on the
        same turn — returns the same id."""
        connection = cls._bound_connection()
        connection.execute(
            "INSERT INTO threads (channel, turn_id) VALUES (?, ?) "
            "ON CONFLICT(channel, turn_id) DO NOTHING",
            (channel, turn_id),
        )
        row = connection.execute(
            "SELECT id FROM threads WHERE channel = ? AND turn_id = ?",
            (channel, turn_id),
        ).fetchone()
        return int(row[0])

    @classmethod
    def set_summary(cls, channel: str, turn_id: int, summary: str) -> bool:
        """Store (or replace) the label of the thread ``(channel, turn_id)``;
        return False when there is no such thread to carry it."""
        cursor = cls._bound_connection().execute(
            "UPDATE threads SET summary = ? WHERE channel = ? AND turn_id = ?",
            (summary, channel, turn_id),
        )
        return cursor.rowcount > 0

    @classmethod
    def summaries(cls, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """``{turn_id: summary}`` for every thread of ``turn_ids`` on ``channel``
        that has a label — the thread feed's collapsed labels."""
        if not turn_ids:
            return {}
        placeholders = ", ".join("?" for _ in turn_ids)
        rows = cls._bound_connection().execute(
            f"SELECT turn_id, summary FROM threads WHERE channel = ? AND turn_id IN ({placeholders}) "
            "AND summary IS NOT NULL",
            (channel, *turn_ids),
        ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

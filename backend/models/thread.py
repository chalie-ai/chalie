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
It carries what is stored once per thread: its topical label (``summary``),
and the model (``provider_id``) and thinking level its turns run on. A new
thread starts with the model and level of its channel's latest thread.
Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class Thread(Model):
    """One conversation thread, keyed by its channel and turn number."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "channel", "turn_id", "summary", "provider_id", "thinking_level",
    )

    @classmethod
    def get_table(cls) -> str:
        return "threads"

    channel: str
    turn_id: int
    summary: str | None
    provider_id: int | None
    thinking_level: str

    @classmethod
    def ensure(cls, channel: str, turn_id: int) -> int:
        """The id of the thread for ``(channel, turn_id)``, created if it has none.

        A thread it creates takes the model and thinking level of the channel's
        latest thread; the channel's first thread has no model, which means the
        default. Idempotent: a second call for the same pair — another writer on
        the same turn — returns the same id."""
        connection = cls._bound_connection()
        connection.execute(
            "INSERT INTO threads (channel, turn_id, provider_id, thinking_level) VALUES (?, ?, "
            "(SELECT provider_id FROM threads WHERE channel = ? ORDER BY turn_id DESC LIMIT 1), "
            "COALESCE((SELECT thinking_level FROM threads WHERE channel = ? ORDER BY turn_id DESC LIMIT 1), 'auto')) "
            "ON CONFLICT(channel, turn_id) DO NOTHING",
            (channel, turn_id, channel, channel),
        )
        row = connection.execute(
            "SELECT id FROM threads WHERE channel = ? AND turn_id = ?",
            (channel, turn_id),
        ).fetchone()
        return int(row[0])

    @classmethod
    def of(cls, channel: str, turn_id: int) -> Thread | None:
        """The thread ``(channel, turn_id)``, or None when it has none."""
        return cls.filter("channel", channel).filter("turn_id", turn_id).first()

    @classmethod
    def latest(cls, channel: str) -> Thread | None:
        """The channel's latest thread (highest turn id), or None on a channel
        with no thread yet."""
        return cls.filter("channel", channel).order_by("turn_id DESC").first()

    @classmethod
    def set_settings(cls, channel: str, turn_id: int, provider_id: int, thinking_level: str) -> bool:
        """Store the model and thinking level of the thread ``(channel, turn_id)``;
        return False when there is no such thread."""
        cursor = cls._bound_connection().execute(
            "UPDATE threads SET provider_id = ?, thinking_level = ? WHERE channel = ? AND turn_id = ?",
            (provider_id, thinking_level, channel, turn_id),
        )
        return cursor.rowcount > 0

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

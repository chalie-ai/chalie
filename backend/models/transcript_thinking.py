"""TranscriptThinking — one ``transcript_thinking`` row: a chain-of-thought
trace anchored to a transcript row.

Active-record row-model (Rule 5 / §4.1). Holds no ``mp``, imports no service,
never emits WS, never reaches upstream (Rule-3 depth). Multiple rows per
``transcript_id`` are expected (one per thinking pass), so the model has an
autoincrement ``id`` (like ``tool_calls``) rather than keyed by
``transcript_id``.
"""

from __future__ import annotations

import json
from typing import ClassVar, Self

from models.model import Model


class TranscriptThinking(Model):
    """One ``transcript_thinking`` row: a chain-of-thought trace anchored to
    a transcript row."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "transcript_id",
        "thinking_trace",
        "duration_ms",
        "tokens",
    )

    @classmethod
    def get_table(cls) -> str:
        return "transcript_thinking"

    transcript_id: int
    thinking_trace: str
    duration_ms: int
    tokens: int

    @classmethod
    def insert(cls, transcript_id: int, thinking_trace: str, duration_ms: int, tokens: int) -> Self:
        """Persist one thinking trace row. Returns the saved instance with
        ``id`` populated. Never commits — the bound connection autocommits
        and ``Database.transaction()`` groups multi-write atomic blocks (I6)."""
        return cls(
            transcript_id=transcript_id,
            thinking_trace=thinking_trace,
            duration_ms=duration_ms,
            tokens=tokens,
        ).save()

    @classmethod
    def for_transcripts(cls, transcript_ids: list[int]) -> list[Self]:
        """Every ``transcript_thinking`` row for the given transcript ids,
        ordered by autoincrement ``id``. Binds the whole id list as one
        JSON-array parameter via ``json_each`` (see
        :meth:`~models.tool_call.ToolCall.delete_by_transcripts`) because a
        GC sweep's candidate set can exceed SQLite's bound-variable limit.
        Empty input is a clean no-op — no query runs."""
        if not transcript_ids:
            return []
        rows = cls._bound_connection().execute(
            "SELECT * FROM transcript_thinking "
            "WHERE transcript_id IN (SELECT value FROM json_each(?)) "
            "ORDER BY id",
            (json.dumps(transcript_ids),),
        ).fetchall()
        return [cls.hydrate(row) for row in rows]

    @classmethod
    def by_turn(cls, channel: str, turn_id: int) -> list[Self]:
        """Every thinking trace of one logical turn, ordered by autoincrement
        ``id``.

        ``transcript_thinking`` carries no turn column; a turn's traces are
        derived by joining ``transcript`` on (channel, turn_id). Ordered by
        ``tt.id`` (not created_at — one-second granularity makes created_at
        ambiguous when rows land in the same second)."""
        cursor = cls._bound_connection().execute(
            "SELECT tt.* FROM transcript_thinking tt "
            "JOIN transcript t ON tt.transcript_id = t.id "
            "WHERE t.channel = ? AND t.turn_id = ? ORDER BY tt.id",
            (channel, turn_id),
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

    @classmethod
    def by_exchange(cls, channel: str, turn_id: int, since_transcript_id: int | None = None) -> list[Self]:
        """This turn's thinking traces from the CURRENT exchange only — the
        traces of one MP recursion instance.

        A turn can hold several exchanges (a reply forks into it, an async
        result re-enters it): each is its own MP instance with its own input
        row, and every trace it writes anchors to that input row
        (``since_transcript_id`` = ``mp.uid``) or to an assistant row written
        after it — so all carry ``transcript_id >= since_transcript_id``.
        Prior exchanges anchored to lower ids are excluded: their raw traces
        stay in the DB while their synthesis carries forward via Previous
        Messages. Reduces to :meth:`by_turn` when ``since_transcript_id`` is
        None (channels without an input row): they never fork, so the turn
        and the exchange coincide. Ordered by ``tt.id``."""
        if since_transcript_id is None:
            return cls.by_turn(channel, turn_id)
        cursor = cls._bound_connection().execute(
            "SELECT tt.* FROM transcript_thinking tt "
            "JOIN transcript t ON tt.transcript_id = t.id "
            "WHERE t.channel = ? AND t.turn_id = ? AND tt.transcript_id >= ? "
            "ORDER BY tt.id",
            (channel, turn_id, since_transcript_id),
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

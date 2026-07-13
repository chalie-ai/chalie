"""One ``tool_calls`` row — the active-record model for the ACT-loop tool trail.

Pure CRUD (Rule-3 depth): holds no ``mp``, imports no service, never emits WS,
never reaches upstream. It only stores its fields, projects itself, and runs its
own table's SQL on the connection bound onto the :class:`Model` base.

A tool call anchors ONLY to the transcript input row that drove it
(``transcript_id``); its logical turn is derived by joining ``transcript`` on
(channel, turn_id), so this table carries no turn_id / channel column of its own
— hence :meth:`by_turn` holds that join SQL (§2.6, a multi-table read the generic
builder can't express).

WS privacy (§6.2): the emitted frame (:meth:`to_json`) exposes ONLY
type/turn_id/id/tool_name/summary/created_at/ended_at/state. ``params`` and
``result`` are the model-facing turn DATA and NEVER cross the wire. ``type`` and
``turn_id`` are transient envelope fields the ``ToolCallService`` sets on the
instance before broadcast; they are not columns, so :meth:`Model.to_dict` /
:meth:`save` exclude them. :meth:`to_dict` (inherited, full-field) stays the
internal projection, distinct from this wire-safe whitelist.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar, Self, cast

from models.model import Model


class ToolCall(Model):
    """One ``tool_calls`` row: field storage + CRUD + wire-safe projection."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "transcript_id",
        "tool_name",
        "params",
        "result",
        "summary",
        "created_at",
        "ended_at",
        "state",
    )

    @classmethod
    def get_table(cls) -> str:
        return "tool_calls"

    # The ``tool_calls.state`` vocabulary — one non-terminal, two terminal
    # (§6.5, DB CHECK).
    STARTED: ClassVar[str] = "started"
    DONE: ClassVar[str] = "done"
    ERROR: ClassVar[str] = "error"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    transcript_id: int
    tool_name: str
    params: str
    result: str
    summary: str
    created_at: str
    ended_at: str | None
    state: str

    # Transient WS envelope (NOT columns) — the ToolCallService sets these on the
    # instance immediately before broadcast; they never persist or project via
    # to_dict.
    type: str
    turn_id: int

    def to_json(self) -> str:
        """The single WS tool frame (§6.2). Carries the row's identity + live
        pill fields — ``params`` and ``result`` NEVER cross the wire (the surface
        refetches the turn over REST for those). The shared DOM-contract keys
        ``turn_id`` and ``type`` are injected by :meth:`Websocket.broadcast`'s
        enrichment step, never hand-written here."""
        return json.dumps(
            {
                "id": self.id,
                "tool_name": self.tool_name,
                "summary": self.summary,
                "created_at": self.created_at,
                "ended_at": self.ended_at,
                "state": self.state,
            },
            default=self._json_default,
        )

    @classmethod
    def by_turn(cls, channel: str, turn_id: int) -> list[Self]:
        """Every tool call of one logical turn, ordered by autoincrement id.

        ``tool_calls`` carries no turn column; a turn's calls are derived by
        joining ``transcript`` on (channel, turn_id). Each turn has exactly one
        input row; an async result or delegate re-entry is a NEW turn with its
        own turn_id, so the join gathers only this turn's calls. Ordered by
        ``tc.id`` (not created_at — one-second granularity makes created_at
        ambiguous when rows land in the same second)."""
        cursor = cls._bound_connection().execute(
            "SELECT tc.* FROM tool_calls tc "
            "JOIN transcript t ON tc.transcript_id = t.id "
            "WHERE t.channel = ? AND t.turn_id = ? ORDER BY tc.id",
            (channel, turn_id),
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

    @classmethod
    def by_exchange(cls, channel: str, turn_id: int, since_transcript_id: int) -> list[Self]:
        """This turn's tool calls from the CURRENT exchange only — the calls of
        one MP recursion instance.

        A turn can hold several exchanges (a reply forks into it, an async
        result re-enters it): each is its own MP instance with its own input
        row, and every call it makes anchors to that input row
        (``since_transcript_id`` = ``mp.uid``) or to an assistant row written
        after it — so all carry ``transcript_id >= since_transcript_id``. Prior
        exchanges anchored to lower ids and are excluded: their raw trail stays
        in the DB while their synthesis carries forward via Previous Messages.
        Reduces to :meth:`by_turn` for a non-forked turn (its input row is the
        turn's lowest id). Ordered by ``tc.id``."""
        cursor = cls._bound_connection().execute(
            "SELECT tc.* FROM tool_calls tc "
            "JOIN transcript t ON tc.transcript_id = t.id "
            "WHERE t.channel = ? AND t.turn_id = ? AND tc.transcript_id >= ? "
            "ORDER BY tc.id",
            (channel, turn_id, since_transcript_id),
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

    @classmethod
    def by_transcript(cls, transcript_id: int) -> list[Self]:
        """All tool calls anchored to one input row, ordered by autoincrement id
        — the narrow single-anchor read. Equivalent to :meth:`by_turn` for a turn
        with a single input row; the turn-keyed read is preferred in the loop
        because it also spans async / delegate re-entries."""
        return cls.filter("transcript_id", transcript_id).order_by("id").get()

    @classmethod
    def by_transcripts(cls, transcript_ids: list[int]) -> list[Self]:
        """All tool calls anchored to any of the given input rows, ordered by
        autoincrement id — the batch-anchor read. Empty input short-circuits to
        ``[]``."""
        return cls.filter_in("transcript_id", transcript_ids).order_by("id").get()

    @classmethod
    def memory_recall_results(cls, transcript_ids: list[int]) -> list[str]:
        """The non-null ``result`` payloads of every ``memory`` tool call
        anchored to the given input rows — the recall envelopes an episode
        window already cited (``tool_name='memory'`` covers both auto-seed and
        LLM-invoked recall). Empty input short-circuits to ``[]``."""
        return [
            cast("str", result)
            for result in cls.filter_in("transcript_id", transcript_ids)
            .filter("tool_name", "memory")
            .filter("result", None, "IS NOT")
            .pluck("result")
        ]

    # Off-turn GC: an orphaned tool call (its anchoring transcript row purged by
    # retention out from under it) is unreachable — every read joins transcript —
    # and is hard-deleted once it ages past this window. Still-anchored calls and
    # young orphans are always kept.
    ORPHAN_GC_AFTER_DAYS: ClassVar[int] = 14

    @classmethod
    def decay(cls) -> int:
        """Off-turn Decayable sweep: hard-delete every tool_calls row older than
        :attr:`ORPHAN_GC_AFTER_DAYS` whose anchoring ``transcript`` row no longer
        exists. Retention purges transcript rows out from under the calls that
        referenced them, leaving dangling ``transcript_id``s no live read can
        reach (they all join ``transcript``); once safely aged out those orphans
        are removed — the one sweep that keeps ``tool_calls`` from growing without
        bound. A still-anchored call is untouched however old, and a young orphan
        waits out the window. The age cutoff is computed in SQL (``julianday``, UTC
        ``'now'``) so this model keeps its no-service-import invariant; the window
        constant binds in as the ``?`` modifier. Single autocommitting DELETE on
        the bound connection (I6 — ``Database`` owns multi-write transactions).
        Returns rows deleted (``0`` = ran, nothing orphaned past the window)."""
        cursor = cls._bound_connection().execute(
            f"DELETE FROM {cls.get_table()} "
            "WHERE julianday(created_at) < julianday('now', ?) "
            "  AND NOT EXISTS (SELECT 1 FROM transcript t WHERE t.id = tool_calls.transcript_id)",
            (f"-{cls.ORPHAN_GC_AFTER_DAYS} days",),
        )
        return cursor.rowcount or 0

    @classmethod
    def usage_counts(cls, exclude: Sequence[str] = ()) -> list[dict[str, object]]:
        """Per-tool usage aggregate — ``{tool_name, count, last_used_at}`` for
        every tool, most-recently-used first. ``exclude`` drops infra tool names
        (compaction/thinking families) the observability view hides; an empty
        ``exclude`` counts every tool. A cross-row ``GROUP BY`` the AND-only
        builder can't express, so it owns its raw SQL on the bound connection."""
        clause = ""
        params: tuple[str, ...] = ()
        if exclude:
            placeholders = ",".join("?" for _ in exclude)
            clause = f"WHERE tool_name NOT IN ({placeholders}) "
            params = tuple(exclude)
        cursor = cls._bound_connection().execute(
            "SELECT tool_name, COUNT(*) AS count, MAX(created_at) AS last_used_at "
            f"FROM {cls.get_table()} {clause}"
            "GROUP BY tool_name "
            "ORDER BY last_used_at DESC",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

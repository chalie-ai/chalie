"""Active-record model for one ``transcript`` row.

Pure CRUD (Rule-3 depth): holds no ``mp``, imports no service, never emits WS,
never reaches upstream. It owns EVERY piece of ``transcript``-table SQL — the
row-shape reads/writes flow through the inherited active-record engine
(``Transcript.filter(...).get()`` / ``Transcript(...).save()``), and the
aggregate/feed shapes a generic builder cannot express (MAX / DISTINCT /
GROUP BY / HAVING / correlated settle subselect) live here as named
classmethods running their own parametrized SQL on the bound connection
(§2.6). Cross-table effects (tool-call un-settle, GC of tool_calls/episodes,
document links, location backfill) belong to the service layer — this model
only touches its own table.
"""

from __future__ import annotations

from typing import ClassVar, Self, cast

from models.model import Model
from models.query import Query


class Transcript(Model):
    """One ``transcript`` row: the persistent conversation record."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "channel", "role", "content", "tool_call_id", "tool_name",
        "internal", "deliberation_score", "created_at", "xml_migrated",
        "location_lat", "location_lon", "location_name", "turn_id", "settled",
        "thinking_level",
    )

    @classmethod
    def get_table(cls) -> str:
        return "transcript"

    # settle0 — the FIRST assistant row of a turn with settled=1: the boundary
    # between a turn's main exchange and its fork continuation. The write path
    # stamps assistant rows settled=1; a settling tool-call demotes to 0
    # (§6.9). No alias needed — every query below is single-table.
    _SETTLE_PREDICATE: ClassVar[str] = "role = 'assistant' AND settled = 1"

    # NULL-safe turn key. Legacy rows carry a NULL turn_id; -id is negative so
    # it never collides with a positive turn_id and sorts older than every chain
    # turn — each legacy row becomes its own singleton turn in chronological
    # order (§6.1).
    _TURN_KEY: ClassVar[str] = "COALESCE(turn_id, -id)"

    _PREVIEW_CHARS: ClassVar[int] = 200

    # ── query scopes ────────────────────────────────────────────────────────

    @classmethod
    def settled_assistants(cls) -> Query[Self]:
        """Late-binding scope over the settle predicate — the settled-assistant
        rows (turn settle0 candidates); compose channel/turn filters onto it."""
        return cls.filter("role", "assistant").filter("settled", 1)

    # ── turn-id allocation (§6.8, atomic in a single-writer begin()) ─────────

    @classmethod
    def next_turn_id(cls, turn_id: int | None, channel: str) -> int:
        """Resolve the turn to write on: ``turn_id`` verbatim when supplied (a
        reply appending to an existing thread), else the channel's next turn —
        ``MAX(turn_id)+1`` (``1`` on an empty channel). The COALESCE formula is
        the single canonical allocation; atomicity comes from the single-writer
        ``begin()`` transaction, not a Python lock (§6.8)."""
        row = cls._bound_connection().execute(
            "SELECT COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1 "
            "FROM transcript WHERE channel = ?))",
            (turn_id, channel),
        ).fetchone()
        return int(row[0])

    # ── settle boundary + deliberation columns ──────────────────────────────

    @classmethod
    def settle0(cls, channel: str, turn_id: int) -> int | None:
        """``id`` of the turn's settle0 (its first settled assistant row), or
        None for a turn that has not settled yet (in-progress/failed)."""
        row = cls._bound_connection().execute(
            f"SELECT MIN(id) FROM transcript WHERE channel = ? AND turn_id = ? "
            f"AND {cls._SETTLE_PREDICATE}",
            (channel, turn_id),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @classmethod
    def latest_thinking_level(cls, channel: str, turn_id: int | None = None) -> str | None:
        """``thinking_level`` of the most recent row (highest id) for ``channel``
        that has a non-NULL ``thinking_level``. When ``turn_id`` is given, also
        filter on it — used to read back the level written onto the current
        turn's input row; when ``turn_id`` is None, span every turn of the
        channel (the spine's "latest level for this type"). Returns None when
        no such row exists (legacy/empty channel)."""
        sql = (
            "SELECT thinking_level FROM transcript WHERE channel = ? "
            "AND thinking_level IS NOT NULL"
        )
        params: list[object] = [channel]
        if turn_id is not None:
            sql += " AND turn_id = ?"
            params.append(turn_id)
        sql += " ORDER BY id DESC LIMIT 1"
        row = cls._bound_connection().execute(sql, tuple(params)).fetchone()
        return row[0] if row and row[0] is not None else None

    def unsettle(self) -> Self:
        """Demote this row's settle flag to 0 — the transcript-table half of the
        cross-table un-settle a tool-call opening triggers (§6.9). The SERVICE
        owns loading the owning row and calling this; the model only flips."""
        self.settled = 0
        return self.save()

    def set_deliberation_score(self, score: float) -> Self:
        """Persist this row's per-turn deliberation score — the value that
        drives thinking-level selection (§6.12). Read back via the
        ``deliberation_score`` column."""
        self.deliberation_score = score
        return self.save()

    # ── channel/turn aggregates (feed + subconscious cursors) ────────────────

    @classmethod
    def last_user_message_at(cls) -> str | None:
        """``created_at`` of the most recent user-role row (subconscious
        user-active gate), or None when no user row exists."""
        row = cls._bound_connection().execute(
            "SELECT MAX(created_at) FROM transcript WHERE role = 'user'",
        ).fetchone()
        return cast("str | None", row[0]) if row else None

    @classmethod
    def latest_id(cls, channels: list[str], *, exclude_roles: tuple[str, ...] = (),
                  require_location: bool = False) -> int | None:
        """``MAX(id)`` over a channel allowlist (the subconscious cursor-delta
        check), optionally floored to geolocated rows. None on an empty set."""
        if not channels:
            return None
        where = [f"channel IN ({cls._placeholders(len(channels))})"]
        params: list[object] = list(channels)
        if require_location:
            where.append("location_lat IS NOT NULL AND location_lon IS NOT NULL")
        for role in exclude_roles:
            where.append("role != ?")
            params.append(role)
        row = cls._bound_connection().execute(
            f"SELECT MAX(id) FROM transcript WHERE {' AND '.join(where)}",
            tuple(params),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @classmethod
    def channel_activity(cls, since_days: int) -> list[tuple[str, int, str]]:
        """Per-channel user-message frequency over the last ``since_days`` (world
        awareness's topic-interest signal): ``(channel, freq, last_seen)`` ordered
        by frequency."""
        rows = cls._bound_connection().execute(
            "SELECT channel, COUNT(*) AS freq, MAX(created_at) AS last_seen "
            "FROM transcript "
            "WHERE created_at >= datetime('now', '-' || ? || ' days') "
            "  AND role = 'user' AND channel IS NOT NULL AND channel != '' "
            "GROUP BY channel ORDER BY freq DESC LIMIT 20",
            (since_days,),
        ).fetchall()
        return [(cast("str", r[0]), int(r[1]), cast("str", r[2])) for r in rows]

    @classmethod
    def distinct_channels(cls) -> list[str]:
        """Every channel present in the transcript (migration/GC discovery)."""
        rows = cls._bound_connection().execute(
            "SELECT DISTINCT channel FROM transcript ORDER BY channel",
        ).fetchall()
        return [cast("str", r[0]) for r in rows]

    # ── thread feed (collapsed, turn-keyed) ──────────────────────────────────

    @classmethod
    def count_turns(cls, channel: str, *, exclude_roles: tuple[str, ...] = (),
                    query: str = "") -> int:
        """Distinct turn count for a channel (NULL-safe ``_TURN_KEY``) — the
        feed's ``has_more`` source. A non-empty ``query`` counts only threads
        matching the search filter."""
        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        having, having_params = cls._thread_query_filter(query)
        row = cls._bound_connection().execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM transcript "
            f"WHERE channel = ?{role_filter} GROUP BY {cls._TURN_KEY}{having})",
            (channel, *exclude_roles, *having_params),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    @classmethod
    def recent_threads(cls, channel: str, *, exclude_roles: tuple[str, ...] = (),
                       limit: int = 20, offset: int = 0, query: str = "",
                       ) -> tuple[list[dict[str, object]], bool, int]:
        """Thread-level metadata for the collapsed feed: one summary per thread
        (turn_id), ordered by last activity (``MAX(id)``). Each dict carries
        ``turn_id``, ``last_activity_at``, ``last_row_id``, ``row_count``,
        ``preview`` (first row's content, truncated) and ``working`` (no settled
        assistant row yet). Legacy NULL-turn_id rows each form their own
        singleton thread via ``_TURN_KEY``. A non-empty ``query`` restricts the
        page to threads with a matching user-role row. Returns
        ``(threads, has_more, threads_returned)``."""
        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        having, having_params = cls._thread_query_filter(query)
        conn = cls._bound_connection()
        page_keys = [
            int(r[0]) for r in conn.execute(
                f"SELECT {cls._TURN_KEY} AS k FROM transcript "
                f"WHERE channel = ?{role_filter} "
                f"GROUP BY k{having} ORDER BY MAX(id) DESC LIMIT ? OFFSET ?",
                (channel, *exclude_roles, *having_params, limit, offset),
            ).fetchall()
        ]
        if not page_keys:
            return [], False, 0

        placeholders = cls._placeholders(len(page_keys))
        meta_rows = conn.execute(
            f"SELECT {cls._TURN_KEY} AS k, MAX(id) AS last_id, MAX(created_at) AS last_ts, "
            f"COUNT(*) AS cnt, "
            f"MIN(CASE WHEN {cls._SETTLE_PREDICATE} THEN id END) IS NULL AS working "
            f"FROM transcript "
            f"WHERE channel = ?{role_filter} AND {cls._TURN_KEY} IN ({placeholders}) "
            f"GROUP BY k",
            (channel, *exclude_roles, *page_keys),
        ).fetchall()
        first_rows = conn.execute(
            f"SELECT {cls._TURN_KEY} AS k, content FROM transcript "
            f"WHERE channel = ?{role_filter} AND {cls._TURN_KEY} IN ({placeholders}) "
            f"GROUP BY k ORDER BY MIN(id) ASC",
            (channel, *exclude_roles, *page_keys),
        ).fetchall()

        meta: dict[int, dict[str, object]] = {}
        for r in meta_rows:
            key = int(r[0])
            meta[key] = {
                "last_activity_at": r[2],
                "last_row_id": int(r[1]),
                "row_count": int(r[3]),
                "working": bool(r[4]),
            }
        first_content: dict[int, str] = {
            int(r[0]): cast("str", r[1] or "")[:cls._PREVIEW_CHARS] for r in first_rows
        }

        threads: list[dict[str, object]] = [
            {
                "turn_id": key if key > 0 else None,
                "last_activity_at": meta.get(key, {}).get("last_activity_at"),
                "last_row_id": meta.get(key, {}).get("last_row_id", 0),
                "row_count": meta.get(key, {}).get("row_count", 0),
                "preview": first_content.get(key, ""),
                "working": meta.get(key, {}).get("working", False),
            }
            for key in page_keys
        ]
        threads.sort(key=lambda t: cast("int", t.get("last_row_id") or 0), reverse=True)
        threads_returned = len(threads)
        has_more = cls.count_turns(
            channel, exclude_roles=exclude_roles, query=query,
        ) > offset + threads_returned
        return threads, has_more, threads_returned

    @classmethod
    def by_turn(cls, channel: str, turn_id: int) -> list[dict[str, object]]:
        """Every row of one thread (``channel``/``turn_id``), oldest-first, as
        plain column dicts — the expand-on-click / turn-serialisation payload
        the REST reads (``api.threads.serialize_turn``) and the
        ``delegate:thread_gist`` prompt project. The WHOLE turn, no settle0
        floor: the caller tags rows past settle0 itself. Dict-shaped (like
        ``recent_threads``) because every consumer treats rows as
        ``r['id']``/``r['role']``/``r['content']``/``r['created_at']``."""
        return [
            row.to_dict()
            for row in cls.filter("channel", channel)
            .filter("turn_id", turn_id)
            .order_by("id ASC")
            .get()
        ]

    @classmethod
    def by_ids(cls, ids: list[int]) -> list[dict[str, object]]:
        """Rows for an explicit ``id`` list, ordered by ``id`` ascending, as
        plain column dicts — the memory-retrieval span fetch and the
        super-episode transcript projection. Empty list → no query, empty
        result; absent ids are simply omitted. Dict-shaped (like ``by_turn``)
        because callers read ``r['id']``/``r['role']``/``r['content']``/
        ``r['created_at']``/``r['tool_name']``."""
        if not ids:
            return []
        return [
            row.to_dict()
            for row in cls.filter_in("id", ids)
            .order_by("id ASC")
            .get()
        ]

    @classmethod
    def turn_scope_ids(cls, ids: list[int]) -> list[int]:
        """Every transcript id that shares a turn with any of the given ids.

        Given a mixed set of transcript row ids, return every transcript id
        belonging to the same turn(s). Rows whose ``turn_id`` is NULL (e.g.
        skip_transcript channels) are excluded from the turn-key lookup; if
        every input row has a NULL turn_id the input list is returned
        unchanged as a fallback. Ordered by ``id`` ascending. Empty input
        short-circuits to ``[]``."""
        if not ids:
            return []
        conn = cls._bound_connection()
        placeholders = cls._placeholders(len(ids))
        channel_turn_rows = conn.execute(
            f"SELECT DISTINCT channel, turn_id FROM transcript "
            f"WHERE id IN ({placeholders}) AND turn_id IS NOT NULL",
            tuple(ids),
        ).fetchall()
        if not channel_turn_rows:
            return list(ids)
        # Row-value IN needs each element parenthesised — ``(?,?),(?,?)`` — not a
        # flat ``?,?`` list (SQLite: "IN(...) element has 1 term - expected 2").
        scope_placeholders = ",".join("(?,?)" for _ in channel_turn_rows)
        scope_ids = conn.execute(
            f"SELECT id FROM transcript "
            f"WHERE (channel, turn_id) IN ({scope_placeholders}) "
            f"ORDER BY id",
            tuple(c for pair in channel_turn_rows for c in pair),
        ).fetchall()
        if not scope_ids:
            return list(ids)
        return [int(r[0]) for r in scope_ids]

    # ── SQL fragment builders (shared, no duplication) ───────────────────────

    # A turn's trailing, reply-less exchange whose most recent turn_executions
    # row ended cancelled (MessageProcessor._step's cancel checkpoint discards
    # the in-flight response before any reply row is stored, §2.7) — excluded
    # from the feed so it never surfaces as a dangling, unanswered thread.
    # ``MAX(CASE WHEN role != 'user' THEN 1 ELSE 0 END) = 0`` means NO row in
    # the turn-group is non-user — i.e. every row is role='user', zero
    # assistant/tool content — deliberately generalized from an earlier
    # ``COUNT(*) = 1`` (a lone reply-less row only ever gated by the FE) since
    # a direct API/automation POST into an open turn_id can append a second
    # (or third) reply-less user row before the cancel lands, and this still
    # correctly leaves a partially-completed turn (any real assistant/tool
    # content already written) alone per existing doctrine. The correlated
    # subquery mirrors ``recent_threads``' own
    # ``MIN(CASE WHEN {predicate} THEN id END) IS NULL AS working`` idiom —
    # an aggregate over a per-row expression, not a join, so it costs nothing
    # on the common case and needs no schema change. COALESCE to '' guards
    # legacy NULL-``turn_id`` rows (no turn_executions row exists for them at
    # all): a bare comparison against a NULL subquery result is NULL, not
    # FALSE, and HAVING drops NULL rows just like FALSE ones — silently
    # excluding every legacy singleton thread from the feed.
    _CANCELLED_ORPHAN_HAVING: ClassVar[str] = (
        "NOT (MAX(CASE WHEN role != 'user' THEN 1 ELSE 0 END) = 0 "
        "AND MAX(COALESCE((SELECT te.state FROM turn_executions te "
        "WHERE te.channel = transcript.channel AND te.turn_id = transcript.turn_id "
        "ORDER BY te.id DESC LIMIT 1), '')) = 'cancelled')"
    )

    # A thread matches a search term when either its user-role content or its
    # (LLM-generated) gist label carries it. The gist half is a correlated
    # subquery keyed on the real ``transcript.turn_id`` (not the NULL-safe
    # ``_TURN_KEY``) — ``thread_gist`` rows only ever exist for a real turn_id,
    # never for a legacy NULL-turn_id singleton, so this correctly matches
    # nothing for those. Wrapped in MAX(COALESCE(..., 0)) to mirror the
    # ``_CANCELLED_ORPHAN_HAVING`` idiom: a per-row correlated subquery
    # aggregated across the GROUP, since HAVING cannot reference a
    # non-aggregated, non-grouped column directly.
    _GIST_MATCH: ClassVar[str] = (
        "MAX(COALESCE((SELECT 1 FROM thread_gist tg WHERE tg.channel = transcript.channel "
        "AND tg.turn_id = transcript.turn_id AND tg.gist LIKE ?), 0)) = 1"
    )

    @classmethod
    def _thread_query_filter(cls, query: str) -> tuple[str, tuple[str, ...]]:
        """The feed's HAVING clause over a per-turn GROUP: always drops a
        cancelled reply-less orphan (``_CANCELLED_ORPHAN_HAVING``), and when
        ``query`` is non-empty additionally keeps only threads carrying a
        user-role row OR a thread_gist label matching it (case-insensitive
        substring). Shared by ``recent_threads`` (the page) and ``count_turns``
        (its ``has_more``) so search and feed run one identical path."""
        conditions = [cls._CANCELLED_ORPHAN_HAVING]
        params: list[str] = []
        query = (query or "").strip()
        if query:
            term = f"%{query}%"
            conditions.append(
                "(MAX(CASE WHEN role = 'user' AND content LIKE ? THEN 1 ELSE 0 END) = 1"
                f" OR {cls._GIST_MATCH})",
            )
            params.extend([term, term])
        return " HAVING " + " AND ".join(conditions), tuple(params)

    @classmethod
    def _placeholders(cls, count: int) -> str:
        """A comma-joined ``?`` list of length ``count`` for an ``IN (...)`` clause."""
        return ",".join("?" * count)

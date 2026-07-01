"""
Transcript Service — persistent conversation record.

DEPRECATED: this is legacy. New functionality belongs in ``services/transcript.py``
(``TranscriptService``). Hard rule: consult user before adding new functions to
``services/transcript.py``.
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"


# ── Read API ─────────────────────────────────────────────────────────
# Every reader projects ONE row shape through _select — the single place the
# SELECT column list and dict-mapping live. Readers compose a static WHERE
# fragment; values are always bound, never interpolated. _select propagates DB
# errors; best-effort readers (get_recent / latest_id / count_turns /
# channel_activity) wrap to a neutral default to preserve their callers.

_COLS = (
    "id, channel, role, content, tool_call_id, tool_name, "
    "internal, created_at, turn_id, location_lat, location_lon, location_name, settled"
)

# NULL-safe turn key. Legacy rows (pre-chain, or rebuilt by
# migrate_transcript_rebuild) carry a NULL turn_id; keying on raw turn_id would
# drop them from turn-paginated history. -id is negative — it can never collide
# with a positive turn_id and sorts older than every chain turn (turn_id
# allocation began only after every legacy row existed), so each legacy row
# becomes its own singleton turn in correct chronological order.
_TURN_KEY = "COALESCE(turn_id, -id)"


def _thread_query_filter(query: str) -> tuple[str, tuple[str, ...]]:
    """The §5.2 search filter as a HAVING clause over a per-turn GROUP: keep only
    threads carrying a user-role row whose content matches ``query`` (case-
    insensitive substring). Empty query → no clause (the unfiltered feed). Shared by
    ``recent_threads`` (the page) and ``count_turns`` (its ``has_more``) so search and
    feed run one identical path."""
    query = (query or "").strip()
    if not query:
        return "", ()
    return (
        " HAVING MAX(CASE WHEN role = 'user' AND content LIKE ? THEN 1 ELSE 0 END) = 1",
        (f"%{query}%",),
    )


# settle0 — the FIRST assistant row of a turn with settled=1. Written as 1 by the
# write path; ActTrail.start() demotes to 0 when a settling tool (counts_as_settle
# True) is recorded. Internal passes (chat_history_compactor, thinking) never
# demote. The predicate scans the row aliased ``t``; no bound params needed.
_SETTLE_PREDICATE = "t.role = 'assistant' AND t.settled = 1"


import functools as _functools


@_functools.lru_cache(maxsize=1)
def _non_settling_tools() -> "frozenset[str]":
    """Registry-derived set of tool_names whose ability has counts_as_settle=False.

    Cached after the first call. Import is deferred to avoid a circular import
    (abilities._registry imports transcript_service indirectly via the DB service).
    ActTrail imports this lazily for the same reason."""
    from abilities._registry import _get_registry  # noqa: PLC0415
    return frozenset(
        name for name, ability in _get_registry().items()
        if not ability.counts_as_settle
    )

class Transcript:
    """Static read/write surface for the transcript table. All access goes
    through ``Transcript.<method>`` — no free functions, no instances."""

    @staticmethod
    def _resolve_location(lat: float | None, lon: float | None, name: str | None, channel: str) -> tuple[float | None, float | None, str | None]:
        """Back-fill live location ONLY for channels whose source profile permits it
        (``location_backfill`` — user activity). Muted / non-user-activity channels
        store NULL location so background work never corrupts the geo signal.
        Returns (None, None, None) on any failure.
        """
        if lat is not None or lon is not None or name is not None:
            return lat, lon, name
        # Bidirectional dependency: the per-source allowlist lives in
        # services/source_profiles.py; this is the location-backfill consumer.
        from services.source_profiles import profile_for

        if not profile_for(channel).location_backfill:
            return None, None, None
        try:
            from services.locale_service import get_location
            loc = get_location()
            return cast("float | None", loc.get('lat')), cast("float | None", loc.get('lon')), cast("str | None", loc.get('name'))
        except Exception:
            return None, None, None


    @staticmethod
    def _in(n: int) -> str:
        return ",".join("?" * n)


    @staticmethod
    def _select(where: str, params: tuple[object, ...], *, order_by: str = "id ASC",
                limit: int | None = None, offset: int = 0) -> list[dict[str, object]]:
        from services.database_service import get_shared_db_service

        sql = f"SELECT {_COLS} FROM transcript WHERE {where} ORDER BY {order_by}"
        args = params
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            args = (*params, limit, offset)
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                'id': r[0], 'channel': r[1], 'role': r[2], 'content': r[3],
                'tool_call_id': r[4], 'tool_name': r[5], 'internal': bool(r[6]),
                'created_at': r[7], 'turn_id': r[8],
                'location_lat': r[9], 'location_lon': r[10], 'location_name': r[11],
                'settled': bool(r[12]),
            }
            for r in rows
        ]


    @staticmethod
    def get_recent(channel: str, limit: int = 20, since_id: int | None = None, *,
                   role: str | None = None, _context: object = None) -> list[dict[str, object]]:
        """Recent rows of a channel — every row after ``since_id`` (oldest-first), or
        the newest ``limit`` rows (also returned oldest-first), optionally restricted
        to one ``role``. Best-effort: empty list on DB error, its pre-existing contract."""
        role_sql = " AND role = ?" if role else ""
        role_args = (role,) if role else ()
        try:
            if since_id is not None:
                return Transcript._select(f"channel = ? AND id > ?{role_sql}", (channel, since_id, *role_args))
            rows = Transcript._select(f"channel = ?{role_sql}", (channel, *role_args), order_by="id DESC", limit=limit)
            rows.reverse()
            return rows
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} get_recent failed: {e}")
            return []


    @staticmethod
    def by_ids(ids: list[int]) -> list[dict[str, object]]:
        """Rows for an explicit id set (rich-media turn spans, episodic window
        formatting, super-episode spans), oldest-first. Empty ids → empty list."""
        if not ids:
            return []
        return Transcript._select(f"id IN ({Transcript._in(len(ids))})", tuple(ids))


    @staticmethod
    def by_turn(channel: str, turn_id: int | None = None) -> list[dict[str, object]]:
        """The one getter. ``turn_id`` given → every row of that ``(channel,
        turn_id)`` (a turn is many rows under the chain model); ``turn_id`` None →
        the whole channel spine, every turn oldest-first. No watermark, no settle0,
        no role filter — consumers mutate the returned rows in Python. id-only
        callers read the ``id`` field off the uniform shape.

        This is the chokepoint of the read+compaction flow — never add a second
        getter, a per-turn loop, or push filtering in here. The whole flow (getter
        → consumer mutations → two-axis watermark → checkpoint summary → writer) is
        narrated on ``MessageProcessor._previous_rows``; read it before changing
        any read path."""
        if turn_id is None:
            return Transcript._select("channel = ?", (channel,))
        return Transcript._select("channel = ? AND turn_id = ?", (channel, turn_id))


    @staticmethod
    def last_user_message_at() -> "str | None":
        """The ``created_at`` of the most recent user-role row, or ``None`` when no
        user row exists. Drives the subconscious worker's user-active gate."""
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                "SELECT MAX(created_at) FROM transcript WHERE role = 'user'",
            ).fetchone()
        return row[0] if row else None


    @staticmethod
    def settle0(channel: str, turn_id: int) -> int | None:
        """``id`` of the turn's settle0 — its first settled assistant row
        (see ``_SETTLE_PREDICATE``), or None for a turn that has not settled yet
        (in-progress/failed). settle0 is the boundary between a turn's main
        exchange and its fork continuation: the shared pivot of the two views."""
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                f"SELECT MIN(t.id) FROM transcript t "
                f"WHERE t.channel = ? AND t.turn_id = ? AND {_SETTLE_PREDICATE}",
                (channel, turn_id),
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else None


    @staticmethod
    def by_time(channels: list[str], lo: str, hi: str) -> list[dict[str, object]]:
        """Rows whose ``created_at`` falls in ``[lo, hi]`` for the given channels —
        review_transcript's ±N-minute re-read. Bounds are ISO datetime strings."""
        if not channels:
            return []
        return Transcript._select(
            f"created_at BETWEEN ? AND ? AND channel IN ({Transcript._in(len(channels))})",
            (lo, hi, *channels),
        )


    @staticmethod
    def window(channels: list[str], *, after_id: int | None = None,
               before_id: int | None = None, require_location: bool = False,
               require_content: bool = False, exclude_roles: tuple[str, ...] = (),
               ) -> list[dict[str, object]]:
        """An id-bounded window over a channel allowlist with optional filters — the
        pattern / geo-pattern behaviour window. Every filter self-no-ops at its
        default, so one query serves both callers (geo passes require_location)."""
        if not channels:
            return []
        where = [f"channel IN ({Transcript._in(len(channels))})"]
        params: list[object] = list(channels)
        if after_id is not None:
            where.append("id > ?")
            params.append(after_id)
        if before_id is not None:
            where.append("id <= ?")
            params.append(before_id)
        if require_content:
            where.append("content IS NOT NULL AND content != ''")
        if require_location:
            where.append("location_lat IS NOT NULL AND location_lon IS NOT NULL")
        for role in exclude_roles:
            where.append("role != ?")
            params.append(role)
        return Transcript._select(" AND ".join(where), tuple(params))


    @staticmethod
    def latest_id(channels: list[str], *, exclude_roles: tuple[str, ...] = (),
                  require_location: bool = False) -> int | None:
        """``MAX(id)`` over a channel allowlist — the subconscious cursor-delta
        check. None on an empty set or DB error."""
        if not channels:
            return None
        where = [f"channel IN ({Transcript._in(len(channels))})"]
        params: list[object] = list(channels)
        if require_location:
            where.append("location_lat IS NOT NULL AND location_lon IS NOT NULL")
        for role in exclude_roles:
            where.append("role != ?")
            params.append(role)
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    f"SELECT MAX(id) FROM transcript WHERE {' AND '.join(where)}",
                    tuple(params),
                ).fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} latest_id failed: {e}")
            return None


    @staticmethod
    def count_turns(channel: str, *, exclude_roles: tuple[str, ...] = (), query: str = "") -> int:
        """Distinct turn count for a channel (NULL-safe ``_TURN_KEY``) — the single
        source for the feed's ``has_more`` and the dashboard turn metric. A non-empty
        ``query`` counts only threads matching the §5.2 search filter. 0 on error."""
        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        having, having_params = _thread_query_filter(query)
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM (SELECT 1 FROM transcript "
                    f"WHERE channel = ?{role_filter} GROUP BY {_TURN_KEY}{having})",
                    (channel, *exclude_roles, *having_params),
                ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} count_turns failed: {e}")
            return 0


    @staticmethod
    def recent_threads(
        channel: str, *, exclude_roles: tuple[str, ...] = (),
        limit: int = 20, offset: int = 0, query: str = "",
    ) -> tuple[list[dict[str, object]], bool, int]:
        """Thread-level metadata for the collapsed feed: one summary row per
        thread (turn_id), ordered by last activity (MAX(created_at) / MAX(id)).

        Returns ``(threads, has_more, threads_returned)`` where each thread dict
        carries ``turn_id``, ``last_activity_at`` (MAX(created_at)), ``last_row_id``
        (MAX(id) — the recency key), ``row_count`` and ``first_content`` (the
        earliest row's content, truncated — the collapsed preview).
        Legacy NULL-turn_id rows each form their own singleton thread via
        ``_TURN_KEY``.

        A non-empty ``query`` turns the feed into search (§5.2): the page is
        restricted to threads with a user-role row matching the keyword — same
        rows, same ordering, same shape, just filtered (the caller caps the limit).
        """
        from services.database_service import get_shared_db_service

        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        having, having_params = _thread_query_filter(query)
        db = get_shared_db_service()
        with db.connection() as conn:
            page_keys = [
                r[0] for r in conn.execute(
                    f"SELECT {_TURN_KEY} AS k FROM transcript "
                    f"WHERE channel = ?{role_filter} "
                    f"GROUP BY k{having} ORDER BY MAX(id) DESC LIMIT ? OFFSET ?",
                    (channel, *exclude_roles, *having_params, limit, offset),
                ).fetchall()
            ]
        if not page_keys:
            return [], False, 0

        placeholders = ",".join("?" * len(page_keys))
        with db.connection() as conn:
            meta_rows = conn.execute(
                f"SELECT {_TURN_KEY} AS k, MAX(id) AS last_id, MAX(created_at) AS last_ts, "
                f"COUNT(*) AS cnt, "
                f"MIN(CASE WHEN role = 'assistant' AND settled = 1 THEN id END) IS NULL AS working "
                f"FROM transcript "
                f"WHERE channel = ?{role_filter} AND {_TURN_KEY} IN ({placeholders}) "
                f"GROUP BY k",
                (channel, *exclude_roles, *page_keys),
            ).fetchall()
            first_rows = conn.execute(
                f"SELECT {_TURN_KEY} AS k, content FROM transcript "
                f"WHERE channel = ?{role_filter} AND {_TURN_KEY} IN ({placeholders}) "
                f"GROUP BY k ORDER BY MIN(id) ASC",
                (channel, *exclude_roles, *page_keys),
            ).fetchall()

        meta: dict[int, dict[str, object]] = {}
        for r in meta_rows:
            key = int(r[0])
            meta[key] = {
                "turn_id": key if key > 0 else None,
                "last_activity_at": r[2],
                "last_row_id": int(r[1]),
                "row_count": int(r[3]),
                "working": bool(r[4]),
            }
        first_content: dict[int, str] = {}
        for r in first_rows:
            key = int(r[0])
            first_content[key] = cast("str", r[1] or "")[:200]

        threads = []
        for k in page_keys:
            key = int(k)
            m = meta.get(key, {})
            threads.append({
                "turn_id": key if key > 0 else None,
                "last_activity_at": m.get("last_activity_at"),
                "last_row_id": m.get("last_row_id", 0),
                "row_count": m.get("row_count", 0),
                "preview": first_content.get(key, ""),
                "working": m.get("working", False),
            })

        threads.sort(key=lambda t: cast("int", t.get("last_row_id") or 0), reverse=True)
        threads_returned = len(threads)
        has_more = Transcript.count_turns(
            channel, exclude_roles=exclude_roles, query=query,
        ) > offset + threads_returned
        return threads, has_more, threads_returned


    @staticmethod
    def distinct_channels() -> list[str]:
        """Every channel present in the transcript (migration discovery). Empty on error."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT channel FROM transcript ORDER BY channel"
                ).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} distinct_channels failed: {e}")
            return []


    @staticmethod
    def channel_activity(since_days: int) -> list[tuple[str, int, str]]:
        """Per-channel user-message frequency over the last ``since_days`` — world
        awareness's topic-interest signal. Returns ``(channel, freq, last_seen)``
        ordered by frequency. Empty on error."""
        from services.time_utils import utc_now
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT channel, COUNT(*) AS freq, MAX(created_at) AS last_seen "
                    "FROM transcript "
                    "WHERE created_at >= datetime(?, '-' || ? || ' days') "
                    "  AND role = 'user' AND channel IS NOT NULL AND channel != '' "
                    "GROUP BY channel ORDER BY freq DESC LIMIT 20",
                    (utc_now().isoformat(), since_days),
                ).fetchall()
            return [(r[0], int(r[1]), r[2]) for r in rows]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} channel_activity failed: {e}")
            return []


    @staticmethod
    def cleanup_unlinked_entries(channel: str | None = None) -> int:
        """Scope-aware transcript GC. Discovers compacted scopes from the
        ``compactions`` table and, per turn, deletes only rows a checkpoint has
        folded — the **intersection** of the two scopes that can read a row:

        * **MAIN** reads each turn's opening exchange (``id <= settle0``) until it
          absorbs the turn (``turn_id <= main_watermark``, the turn_id axis);
        * **THREAD** (a forked turn — one with any row past settle0) reads the
          WHOLE turn (``id > fork_watermark``, the transcript-id axis), so it owns
          even the pre-settle0 head the main spine also reads.

        A row dies only when BOTH are done with it: a thread compacting past its
        early rows must not delete the head the main spine still needs, and main
        absorbing a turn must not delete rows its live thread still reads. settle0
        itself is never collected — the spine re-derives it on every read. Episode-
        cited rows are always preserved. Best-effort: 0 on error."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            if channel:
                channels = [channel]
            else:
                with db.connection() as conn:
                    channels = [r[0] for r in conn.execute(
                        "SELECT DISTINCT channel FROM compactions"
                    ).fetchall()]

            total_deleted = sum(Transcript._gc_channel(ch) for ch in channels)
            # Always log the count — a steady 0 across channels is the signature
            # of a watermark/discovery regression and must not stay invisible.
            logger.info(
                f"{LOG_PREFIX} Transcript GC: deleted {total_deleted} folded "
                f"entries across {len(channels)} channel(s)"
            )
            return total_deleted
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} cleanup_unlinked_entries failed: {e}")
            return 0


    @staticmethod
    def _gc_channel(channel: str) -> int:
        """Delete one channel's folded rows under the two-axis watermark."""
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        # Latest watermark per scope: main (for_turn_id NULL → turn_id axis) plus
        # one per fork (for_turn_id = T → transcript.id axis).
        with db.connection() as conn:
            scope_rows = conn.execute(
                "SELECT c.for_turn_id, c.compacted_up_to FROM compactions c "
                "WHERE c.channel = ? AND c.id = (SELECT MAX(c2.id) FROM compactions c2 "
                "WHERE c2.channel = c.channel AND c2.for_turn_id IS c.for_turn_id)",
                (channel,),
            ).fetchall()
        main_wm: int | None = None
        fork_wm: dict[int, int] = {}
        for for_tid, up_to in scope_rows:
            if for_tid is None:
                main_wm = int(up_to)
            else:
                fork_wm[int(for_tid)] = int(up_to)

        # Candidate turns: every fork-compacted turn, plus every main-absorbed turn
        # that still carries a main-owned row below its settle0 (the correlated
        # settle subquery bounds the main sweep to turns with something to delete).
        candidate_turns: set[int] = set(fork_wm)
        if main_wm is not None:
            settle = (
                "(SELECT MIN(t.id) FROM transcript t WHERE t.channel = transcript.channel "
                f"AND t.turn_id = transcript.turn_id AND {_SETTLE_PREDICATE})"
            )
            with db.connection() as conn:
                candidate_turns.update(int(r[0]) for r in conn.execute(
                    "SELECT DISTINCT transcript.turn_id FROM transcript "
                    f"WHERE channel = ? AND turn_id IS NOT NULL AND turn_id <= ? AND id < {settle}",
                    (channel, main_wm),
                ).fetchall())

        cited = Transcript._episode_cited_ids(channel)
        deleted = 0
        for tid in candidate_turns:
            settle_id = Transcript.settle0(channel, tid)
            if settle_id is None:
                continue  # unsettled turn — its boundary is undefined, leave it whole
            main_absorbed = main_wm is not None and tid <= main_wm
            fwm = fork_wm.get(tid)
            with db.connection() as conn:
                ids = [int(r[0]) for r in conn.execute(
                    "SELECT id FROM transcript WHERE channel = ? AND turn_id = ?",
                    (channel, tid),
                ).fetchall()]
            # A forked turn (any row past settle0) reads as the WHOLE turn, so the
            # thread owns every row above its watermark — including the pre-settle0
            # head main also reads. A row dies only when BOTH scopes are done: main
            # needs id <= settle0 until it absorbs the turn; the thread needs
            # id > its fork watermark (fork_floor; 0 ⇒ never compacted ⇒ keeps
            # everything). settle0 is re-derived every read, so it is never collected.
            is_forked = any(rid > settle_id for rid in ids)
            fork_floor = fwm if fwm is not None else 0
            dead = [
                rid for rid in ids
                if rid not in cited and rid != settle_id
                and (rid > settle_id or main_absorbed)
                and (not is_forked or rid <= fork_floor)
            ]
            if not dead:
                continue
            ph = ','.join('?' * len(dead))
            # Children before parent: tool_calls FK-references transcript with no
            # ON DELETE CASCADE, so a folded row's audit rows go first.
            with db.connection() as conn:
                conn.execute(f"DELETE FROM tool_calls WHERE transcript_id IN ({ph})", dead)
                conn.execute(f"DELETE FROM transcript WHERE id IN ({ph})", dead)
            deleted += len(dead)
        return deleted


    @staticmethod
    def _episode_cited_ids(channel: str) -> set[int]:
        """Transcript ids cited by any live episode of the channel — GC never
        deletes a row an episode still points at."""
        import json
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT transcript_ids FROM episodes WHERE channel = ? "
                "AND deleted_at IS NULL AND transcript_ids IS NOT NULL "
                "AND transcript_ids != '[]'",
                (channel,),
            ).fetchall()
        cited: set[int] = set()
        for (blob,) in rows:
            try:
                ids = json.loads(blob)
            except Exception:
                continue
            if isinstance(ids, list):
                cited.update(int(i) for i in ids if i is not None)
        return cited


    @staticmethod
    def write_input_row(
        channel: str, role: str, content: str, *, turn_id: "int | None" = None,
    ) -> int:
        """Write a turn's anchoring input row, opening the next turn (or appending
        to an existing thread).

        turn_id is the per-channel monotonic turn boundary — redefined as the
        *thread* id: a thread-starter allocates a new turn_id; a reply appends rows
        carrying the **existing** turn_id. When ``turn_id`` is None (thread-starter),
        the next value, ``MAX(turn_id)+1`` for the channel, is computed inside the
        INSERT so the allocation is atomic with the write — no read-then-insert
        race when two same-channel turns open concurrently. When ``turn_id`` is
        supplied (reply), that value is written verbatim — no allocation. Read the
        allocated value back with :func:`turn_id_of_row` (only needed on the
        thread-starter path; the reply path already knows its turn_id).
        """
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        lat, lon, loc_name = Transcript._resolve_location(None, None, None, channel)
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcript (channel, role, content, xml_migrated,
                                        location_lat, location_lon, location_name, turn_id,
                                        settled)
                VALUES (?, ?, ?, 1, ?, ?, ?,
                        COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1
                                     FROM transcript WHERE channel = ?)),
                        ?)
                """,
                (channel, role, content, lat, lon, loc_name, turn_id, channel,
                 1 if role == 'assistant' else 0),
            )
            row_id = cursor.lastrowid
            cursor.close()

        return cast("int", row_id)


    @staticmethod
    def turn_id_of_row(row_id: int) -> int:
        """The turn_id the INSERT's COALESCE subquery opened for a transcript row.

        write_input_row allocates the next turn atomically inside the write (under
        the SQLite writer lock), so the caller cannot know the value up front — it
        reads it back here by row id. Reading MAX(turn_id) instead would race: a
        concurrent same-channel turn could advance the max between the write and the
        read. Returns 0 if the row is gone or still unnumbered."""
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                "SELECT turn_id FROM transcript WHERE id = ?", (row_id,)
            ).fetchone()
        return row[0] if row and row[0] is not None else 0


    @staticmethod
    def link_transcript_doc(transcript_id: int, doc_id: str) -> None:
        """Link an uploaded document to the transcript turn that carried it.

        Powers chat-attachment persistence across page refresh: the live preview is a
        browser-only blob: URL, so on reload the rebuild (api.threads
        .serialize_turn) joins this table to re-render the image/file from
        /api/documents/<id>/preview.  INSERT OR IGNORE keeps it idempotent against the
        composite primary key.  Called from message_processor._seed_upload_attachment.
        """
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO transcript_docs (transcript_id, doc_id) VALUES (?, ?)",
                (transcript_id, doc_id),
            )


    @staticmethod
    def latest_input_content(channel: str) -> "str | None":
        """The most recent input-row content on a channel — the post-compaction
        continuation's "the user query was: …".

        An input row is any NON-assistant row: there are many input roles (user /
        proactive_thought / external_agent / vision / …), so we exclude the one
        output-shaped role rather than hardcode role='user'. Ordered by monotonic id
        (not created_at — one-second granularity ties). Returns None on an empty
        channel."""
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                "SELECT content FROM transcript "
                "WHERE channel = ? AND role != 'assistant' "
                "ORDER BY id DESC LIMIT 1",
                (channel,),
            ).fetchone()
        return row[0] if row else None


    @staticmethod
    def write_assistant_row(channel: str, content: str, turn_id: "int | None" = None) -> int:
        """Write one assistant row for a single chain step, grounded to its turn.

        Under the recursive turn chain () every step persists its own row
        via ``MessageProcessor._store_row`` — the prose of each tool-bearing step
        plus the final settle text — so one turn produces MULTIPLE assistant rows
        that share a ``turn_id``, not a single end message. The MP supplies its
        current ``turn_id`` so each row shares the boundary of the turn's input row.
        A ``None`` turn_id falls back to the same fresh-turn allocation as
        write_input_row — the path an anchorless re-entry (no input row) takes to
        open its own turn."""
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        lat, lon, loc_name = Transcript._resolve_location(None, None, None, channel)
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcript (channel, role, content, xml_migrated,
                                        location_lat, location_lon, location_name, turn_id,
                                        settled)
                VALUES (?, ?, ?, 1, ?, ?, ?,
                        COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1
                                     FROM transcript WHERE channel = ?)),
                        1)
                """,
                (channel, 'assistant', content, lat, lon, loc_name, turn_id, channel),
            )
            row_id = cursor.lastrowid
            cursor.close()

        return cast("int", row_id)

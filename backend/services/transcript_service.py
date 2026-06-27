"""
Transcript Service — persistent conversation record.
"""

import logging
import threading
from collections import Counter
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.database_service import DatabaseService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"

# Rolling episode extraction — DB-state-driven.
#
# On every transcript append, count the transcripts in the channel whose id
# is greater than the channel's latest episode.transcript_id_end. When the
# tail reaches _EXTRACTION_THRESHOLD (20), fire extraction. The extractor
# pulls the last _EXTRACTION_WINDOW (25) rows — 20 new + 5 overlap with
# the prior window — producing one episode per ~20 new transcript turns.
#
# No process-local counter, no boot catch-up: trigger state lives entirely
# in the DB (transcript.id vs episodes.transcript_id_end) so restarts never
# desync from accumulated history.
_EXTRACTION_THRESHOLD = 20
_EXTRACTION_WINDOW = 25
_EXTRACTION_OVERLAP = 5


# ── Read API ─────────────────────────────────────────────────────────
# Every reader projects ONE row shape through _select — the single place the
# SELECT column list and dict-mapping live. Readers compose a static WHERE
# fragment; values are always bound, never interpolated. _select propagates DB
# errors; best-effort readers (get_recent / latest_id / count_turns /
# channel_activity) wrap to a neutral default to preserve their callers.

_COLS = (
    "id, channel, role, content, tool_call_id, tool_name, "
    "internal, created_at, turn_id, location_lat, location_lon, location_name"
)

# NULL-safe turn key. Legacy rows (pre-chain, or rebuilt by
# migrate_transcript_rebuild) carry a NULL turn_id; keying on raw turn_id would
# drop them from turn-paginated history. -id is negative — it can never collide
# with a positive turn_id and sorts older than every chain turn (turn_id
# allocation began only after every legacy row existed), so each legacy row
# becomes its own singleton turn in correct chronological order.
_TURN_KEY = "COALESCE(turn_id, -id)"

# settle0 — the FIRST assistant row of a turn whose tool_calls carry NO
# model-driven tool. The two internally-dispatched passes (chat_history_compactor,
# thinking) DO record a tool_calls row but are not model tools, so they never
# demote a settle. The orchestrator ends a turn on the first no-tool response, so
# "first no-model-tool assistant" IS the turn's settle by construction. The
# predicate scans the row aliased ``t``; every reader that uses it binds
# ``*_INTERNAL_TOOLS`` in textual order after its own scope params.
_INTERNAL_TOOLS: tuple[str, ...] = ("chat_history_compactor", "thinking")
_SETTLE_PREDICATE = (
    "t.role = 'assistant' AND NOT EXISTS (SELECT 1 FROM tool_calls tc "
    "WHERE tc.transcript_id = t.id AND tc.tool_name NOT IN "
    f"({','.join('?' * len(_INTERNAL_TOOLS))}))"
)

# Cold-start bound for the MAIN spine read: at watermark 0 (no checkpoint yet)
# the spine would otherwise pull every turn and blow the context cap before a
# checkpoint can form. Cap it to the most-recent N turns' exchanges; compaction
# then advances the watermark on the first cap trip and the bound stops biting.
_SPINE_TURN_LIMIT = 12


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
    def append(
        channel: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        internal: bool = False,
        location_lat: float | None = None,
        location_lon: float | None = None,
        location_name: str | None = None,
    ) -> int | None:
        if not channel:
            return None
        if not content and role != 'assistant':
            return None

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            lat, lon, loc_name = Transcript._resolve_location(
                location_lat, location_lon, location_name, channel
            )

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO transcript (
                        channel, role, content, tool_call_id, tool_name, internal,
                        xml_migrated, location_lat, location_lon, location_name
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (channel, role, content, tool_call_id, tool_name, 1 if internal else 0,
                     lat, lon, loc_name),
                )
                rowid = cursor.lastrowid
                cursor.close()

            Transcript._maybe_trigger_extraction(channel, rowid)

            return rowid

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to append: {e}")
            return None





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
    def by_turn(channel: str, turn_id: int) -> list[dict[str, object]]:
        """Every row of one ``(channel, turn_id)`` — a turn is many rows under the
        chain model. id-only callers read the ``id`` field off the uniform shape."""
        return Transcript._select("channel = ? AND turn_id = ?", (channel, turn_id))


    @staticmethod
    def settle0(channel: str, turn_id: int) -> int | None:
        """``id`` of the turn's settle0 — its first no-model-tool assistant row
        (see ``_SETTLE_PREDICATE``), or None for a turn that has not settled yet
        (in-progress/failed). settle0 is the boundary between a turn's main
        exchange and its fork continuation: the shared pivot of the two views."""
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                f"SELECT MIN(t.id) FROM transcript t "
                f"WHERE t.channel = ? AND t.turn_id = ? AND {_SETTLE_PREDICATE}",
                (channel, turn_id, *_INTERNAL_TOOLS),
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else None


    @staticmethod
    def main_spine(channel: str, after_turn_id: int, *,
                   limit: int = _SPINE_TURN_LIMIT) -> list[dict[str, object]]:
        """The MAIN view's previous_messages — the multi-turn spine.

        For each settled turn with ``turn_id > after_turn_id`` (the main watermark,
        a turn_id), emit the opening exchange: rows with ``id <= settle0(turn)``.
        Fork continuations (rows past settle0) never reach the spine; in-progress
        turns (no settle) are skipped, so the live turn is excluded without a
        special case. Bounded to the most-recent ``limit`` turns so a cold
        (watermark-0) read can't blow the cap before a checkpoint forms. Rows are
        id-ordered — chronological for the spine, since each turn's opening
        exchange is written contiguously before the next turn starts (fork
        replies, which append later high ids, are exactly what the cut excludes)."""
        settle = (
            "(SELECT MIN(t.id) FROM transcript t WHERE t.channel = transcript.channel "
            f"AND t.turn_id = transcript.turn_id AND {_SETTLE_PREDICATE})"
        )
        recent = (
            "SELECT t.turn_id FROM transcript t "
            f"WHERE t.channel = ? AND t.turn_id > ? AND {_SETTLE_PREDICATE} "
            "GROUP BY t.turn_id ORDER BY t.turn_id DESC LIMIT ?"
        )
        where = f"channel = ? AND turn_id > ? AND id <= {settle} AND turn_id IN ({recent})"
        params = (
            channel, after_turn_id, *_INTERNAL_TOOLS,         # outer scope + correlated settle
            channel, after_turn_id, *_INTERNAL_TOOLS, limit,  # recent-turns subquery
        )
        return Transcript._select(where, params)


    @staticmethod
    def fork_rows(channel: str, turn_id: int, watermark: int,
                  before_id: int | None) -> list[dict[str, object]]:
        """The THREAD view's previous_messages for one thread — the whole turn.

        Emit EVERY row of turn ``turn_id`` with ``id > watermark`` (the thread's
        transcript-id checkpoint axis) AND ``id < before_id`` — the upper bound
        drops the current reply's own rows so the live message isn't rendered twice
        (it renders separately), mirroring how the spine skips its in-progress turn.
        ``before_id`` None ⇒ no upper bound. No ``settle0`` floor: a thread is the
        whole turn — opening user message and pre-settle interims included — so read
        and render are both trivial (paint every row of the ``turn_id``)."""
        where = (
            "channel = ? AND turn_id = ? "
            "AND id > ? AND id < COALESCE(?, 9223372036854775807)"
        )
        return Transcript._select(where, (channel, turn_id, watermark, before_id))


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
    def count_turns(channel: str, *, exclude_roles: tuple[str, ...] = ()) -> int:
        """Distinct turn count for a channel (NULL-safe ``_TURN_KEY``) — the single
        source for the feed's ``has_more`` and the dashboard turn metric. 0 on error."""
        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    f"SELECT COUNT(DISTINCT {_TURN_KEY}) FROM transcript "
                    f"WHERE channel = ?{role_filter}",
                    (channel, *exclude_roles),
                ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} count_turns failed: {e}")
            return 0


    @staticmethod
    def recent_threads(
        channel: str, *, exclude_roles: tuple[str, ...] = (),
        limit: int = 20, offset: int = 0,
    ) -> tuple[list[dict[str, object]], bool, int]:
        """Thread-level metadata for the collapsed feed: one summary row per
        thread (turn_id), ordered by last activity (MAX(created_at) / MAX(id)).

        Returns ``(threads, has_more, threads_returned)`` where each thread dict
        carries ``turn_id``, ``last_activity_at`` (MAX(created_at)), ``last_row_id``
        (MAX(id) — the recency key), ``row_count`` and ``first_content`` (the
        earliest row's content, truncated — the collapsed preview).
        Legacy NULL-turn_id rows each form their own singleton thread via
        ``_TURN_KEY``.
        """
        from services.database_service import get_shared_db_service

        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        db = get_shared_db_service()
        with db.connection() as conn:
            page_keys = [
                r[0] for r in conn.execute(
                    f"SELECT {_TURN_KEY} AS k FROM transcript "
                    f"WHERE channel = ?{role_filter} "
                    f"GROUP BY k ORDER BY MAX(id) DESC LIMIT ? OFFSET ?",
                    (channel, *exclude_roles, limit, offset),
                ).fetchall()
            ]
        if not page_keys:
            return [], False, 0

        placeholders = ",".join("?" * len(page_keys))
        with db.connection() as conn:
            meta_rows = conn.execute(
                f"SELECT {_TURN_KEY} AS k, MAX(id) AS last_id, MAX(created_at) AS last_ts, "
                f"COUNT(*) AS cnt FROM transcript "
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
            })

        threads.sort(key=lambda t: cast("int", t.get("last_row_id") or 0), reverse=True)
        threads_returned = len(threads)
        has_more = Transcript.count_turns(channel, exclude_roles=exclude_roles) > offset + threads_returned
        return threads, has_more, threads_returned


    @staticmethod
    def thread_rows(
        channel: str, turn_id: int, *, exclude_roles: tuple[str, ...] = (),
    ) -> list[dict[str, object]]:
        """Every row of one ``(channel, turn_id)`` thread — the expand-on-click
        fetch. Excludes compaction/subagent_return by default so the feed view is
        clean; callers that need the full row set use ``by_turn`` directly."""
        role_filter = "".join(" AND role != ?" for _ in exclude_roles)
        return Transcript._select(
            f"channel = ?{role_filter} AND turn_id = ?",
            (channel, *exclude_roles, turn_id),
        )


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
                    (channel, main_wm, *_INTERNAL_TOOLS),
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


    # ── Internal helpers ─────────────────────────────────────────────────


    @staticmethod
    def _maybe_trigger_extraction(channel: str, rowid: int | None) -> None:
        """Fire episode extraction when the channel has accumulated
        _EXTRACTION_THRESHOLD untriggered transcripts.

        Gate: only channels whose source profile has ``extract_episodes`` produce
        episodes (user, dmn, external-agent:*). Every other channel (delegate:*,
        skills_building, scheduled, …) resolves to the muted default and returns
        here before any DB work, keeping the episodes table free of housekeeping
        noise and bounding which channels SubconsciousWorker._step_consolidate()
        iterates.

        DB-state-driven: counts transcripts with id > MAX(episodes.transcript_id_end)
        for the channel. When count >= threshold, fire. No process-local state,
        so restarts and container rebuilds cannot desync the trigger from the
        actual tail of accumulated history.

        Never raises — failures logged and silently ignored.
        """
        if rowid is None:
            return
        # Bidirectional dependency: the per-source allowlist lives in
        # services/source_profiles.py; this is the episode-gate consumer.
        from services.source_profiles import profile_for

        if not profile_for(channel).extract_episodes:
            return

        try:
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM transcript
                    WHERE channel = ?
                      AND id > COALESCE(
                          (SELECT MAX(transcript_id_end)
                           FROM episodes
                           WHERE channel = ? AND deleted_at IS NULL
                             AND transcript_id_end IS NOT NULL),
                          0
                      )
                    """,
                    (channel, channel),
                ).fetchone()
            untriggered = row[0] if row else 0
            if untriggered >= _EXTRACTION_THRESHOLD:
                Transcript._trigger_episode_extraction(channel, rowid)
        except Exception as e:
            logger.warning(
                f"{LOG_PREFIX} _maybe_trigger_extraction failed "
                f"(channel={channel}, rowid={rowid}): {e}"
            )


    @staticmethod
    def _trigger_episode_extraction(channel: str, rowid: int) -> None:
        """Fire-and-forget episode extraction for the window ending at rowid.

        Never raises — any failure is logged only.
        """
        def _run() -> None:
            try:
                import threading

                from configs.channels import EpisodeEncoderConfig
                from services.database_service import get_shared_db_service
                from services.message_processor import MessageProcessor
                from services.episodic_service import (
                    EpisodicService, _fetch_novelty_comparison_set, compute_novelty,
                )
                from services.salience_service import compute_salience
                from services.embedding_service import get_embedding_service

                db = get_shared_db_service()

                # ── 1. Fetch the window ──────────────────────────────────────────
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id, role, content, tool_name, created_at,
                               location_lat, location_lon, location_name
                        FROM transcript
                        WHERE channel = ? AND id <= ?
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (channel, rowid, _EXTRACTION_WINDOW),
                    )
                    rows = cursor.fetchall()
                    cursor.close()

                if not rows:
                    return

                entries = [
                    {
                        'id': r[0],
                        'role': r[1],
                        'content': r[2],
                        'tool_name': r[3],
                        'created_at': r[4],
                        'location_lat': r[5],
                        'location_lon': r[6],
                        'location_name': r[7],
                        'channel': channel,
                    }
                    for r in reversed(rows)
                ]

                # ── 2. Format window string ──────────────────────────────────────
                window_str = Transcript._format_window_entries(entries)

                # ── 3. Fetch referenced episodes via tool_calls ──────────────────
                referenced_episodes = Transcript._fetch_referenced_episodes(entries, db)
                referenced_str = Transcript._format_episodes_for_prompt(referenced_episodes)

                # ── 4. Encode the window via the flat episode-encoder channel ────
                # _window / _referenced are read by EpisodeEncoderConfig's
                # get_user_prompt; set them on the instance before _run().
                emp = object.__new__(MessageProcessor)
                MessageProcessor.__init__(emp, "", None)
                emp.config = EpisodeEncoderConfig()
                emp.uid = None
                emp.cancel_event = threading.Event()
                emp.thinking_level = "low"
                setattr(emp, '_window', window_str)
                setattr(emp, '_referenced', referenced_str)
                response = emp._run()
                snapshots = cast("list[dict[str, object]]", Transcript._safe_json_load(response))
                if not snapshots:
                    return

                # ── 5. Resolve service handles ───────────────────────────────────
                episodic_svc = EpisodicService(db)
                emb_svc = get_embedding_service()

                valid_ids = {e['id'] for e in entries}

                # Hoist novelty comparison set ONCE — shared across all snapshots
                prior_embeddings = _fetch_novelty_comparison_set(channel)

                # ── 6. Store snapshots ───────────────────────────────────────────
                for ep in snapshots:
                    try:
                        if Transcript._is_delete_only(ep):
                            delete_id = ep.get('delete_id')
                            if delete_id:
                                episodic_svc.soft_delete(cast("str", delete_id))
                            continue

                        # Filter transcript_ids to valid window IDs
                        raw_ids = cast("list[object]", ep.get('transcript_ids') or [])
                        ep['transcript_ids'] = [i for i in raw_ids if i in valid_ids]
                        if not ep['transcript_ids']:
                            continue

                        ep['transcript_id_start'] = min(cast("list[int]", ep['transcript_ids']))
                        ep['transcript_id_end'] = max(cast("list[int]", ep['transcript_ids']))
                        ep['channel'] = channel

                        dominant_location = Transcript._aggregate_dominant_location(entries)
                        if dominant_location.get('lat') is not None:
                            ep['location_lat'] = dominant_location['lat']
                            ep['location_lon'] = dominant_location['lon']
                            ep['location_name'] = dominant_location.get('name')

                        gist = cast("str", ep.get('gist', '') or '')
                        embedding = emb_svc.generate_embedding(gist) if gist else None

                        novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
                        ep['salience'] = compute_salience(
                            valence=float(cast("float", ep.get('emotional_valence') or 0.0)),
                            arousal=float(cast("float", ep.get('emotional_arousal') or 0.0)),
                            has_open_loop=bool(ep.get('has_open_loop', False)),
                            novelty=novelty,
                        )

                        # Pop transient fields — not persisted
                        ep.pop('has_open_loop', None)
                        update_id = ep.pop('update_id', None)
                        ep.pop('delete_id', None)  # defensive — should be None here

                        if update_id:
                            episodic_svc.update_episode(cast("str", update_id), ep, embedding=embedding)
                        else:
                            episodic_svc.store_episode(ep, embedding=embedding)

                        # Update the novelty comparison set so subsequent snapshots
                        # in this same extraction run are compared against already-stored
                        # episodes, not just the stale pre-run set.
                        if embedding is not None:
                            from services.embedding_utils import pack_embedding as _pack
                            blob = _pack(embedding)
                            if blob is not None:
                                prior_embeddings.append(blob)

                    except Exception as ep_err:
                        logger.warning(f"{LOG_PREFIX} Episode store failed in trigger: {ep_err}")

            except Exception as e:
                logger.warning(
                    f"{LOG_PREFIX} Episode extraction trigger failed "
                    f"(channel={channel}, rowid={rowid}): {e}"
                )

        threading.Thread(target=_run, daemon=True).start()


    # ── Episode extraction helpers ───────────────────────────────────────────────


    @staticmethod
    def _aggregate_dominant_location(entries: list[dict[str, object]]) -> dict[str, object]:
        """When all location names are unique (no majority), falls back to the most recent non-null row."""
        located = [
            e for e in entries
            if e.get('location_lat') is not None or e.get('location_name') is not None
        ]
        if not located:
            return {'lat': None, 'lon': None, 'name': None}

        name_counts = Counter(
            e['location_name'] for e in located if e.get('location_name') is not None
        )

        if name_counts:
            top_name, top_count = name_counts.most_common(1)[0]
            if top_count > 1:
                # Clear dominant name — use any row with that name for coords
                for e in reversed(located):
                    if e.get('location_name') == top_name:
                        return {
                            'lat': e.get('location_lat'),
                            'lon': e.get('location_lon'),
                            'name': top_name,
                        }

        # All names are unique (or no names) — use the most recent located row
        most_recent = located[-1]
        return {
            'lat': most_recent.get('location_lat'),
            'lon': most_recent.get('location_lon'),
            'name': most_recent.get('location_name'),
        }


    @staticmethod
    def _format_window_entries(entries: list[dict[str, object]]) -> str:
        lines = []
        for entry in entries:
            entry_id = entry.get('id', '?')
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            tool_name = entry.get('tool_name')
            created_at = entry.get('created_at', '')
            if tool_name:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
            else:
                lines.append(f"[{entry_id}] ({created_at}) {role}: {content}")
        return "\n".join(lines)


    @staticmethod
    def _parse_episode_ids_from_results(result_texts: list[str]) -> set[str]:
        """Extract episode IDs from rendered memory recall envelopes.

        Only rows with ``kind == "episode"`` are collected — data-graph rows are
        keyed by data-graph key (not an episode id) and are intentionally skipped.
        Malformed / non-recall envelopes are skipped silently.
        """
        import json as _json

        episode_ids: set[str] = set()
        for text in result_texts:
            if not text or "[end:memory]" not in text:
                continue
            nl = text.find("]\n")
            end = text.find("\n[end:memory]")
            if nl == -1 or end == -1 or end <= nl:
                continue
            body = text[nl + 2:end]
            try:
                parsed = _json.loads(body)
            except (ValueError, TypeError):
                continue
            rows = parsed.get("results") if isinstance(parsed, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("kind") == "episode":
                    eid = str(row.get("id", "")).strip()
                    if eid:
                        episode_ids.add(eid)
        return episode_ids


    @staticmethod
    def _fetch_referenced_episodes(entries: list[dict[str, object]], db: "DatabaseService") -> list[dict[str, object]]:
        """Query tool_calls for memory skill invocations within the window.

        ``tool_name='memory'`` covers both auto-seed and LLM-invoked recall —
        both are dispatched under the same tool name.
        """

        t_ids = [e['id'] for e in entries if e.get('id')]
        if not t_ids:
            return []

        try:
            placeholders = ','.join('?' * len(t_ids))
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT result FROM tool_calls
                    WHERE transcript_id IN ({placeholders})
                      AND tool_name = 'memory'
                      AND result IS NOT NULL
                    """,
                    t_ids,
                )
                rows = cursor.fetchall()
                cursor.close()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} _fetch_referenced_episodes query failed: {exc}")
            return []

        # Parse episode IDs out of each memory recall envelope. recall renders a
        # structured JSON body — ``[memory(status=success, …)]\n{"results": [{"id":…,
        # "kind":"episode", …}, …]}\n[end:memory]`` — so we pull the JSON between the
        # open tag and ``[end:memory]`` and collect the ``id`` of every row whose
        # ``kind`` is ``episode`` (data-graph rows carry their own kind and are keyed
        # by data-graph key, not an episode id, so they are skipped).
        episode_ids = Transcript._parse_episode_ids_from_results([row[0] or '' for row in rows])

        if not episode_ids:
            return []

        try:
            from services.episodic_service import EpisodicService
            episodic_svc = EpisodicService(db)
            episodes = []
            for eid in episode_ids:
                ep = episodic_svc.get_episode_by_id(eid)
                if ep:
                    episodes.append(ep)
            return episodes
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} _fetch_referenced_episodes fetch failed: {exc}")
            return []


    @staticmethod
    def _format_episodes_for_prompt(episodes: list[dict[str, object]]) -> str:
        if not episodes:
            return ''
        lines = []
        for ep in episodes:
            eid = ep.get('id', '')
            gist = ep.get('gist', '')
            created_at = ep.get('created_at', '')
            lines.append(f"id: {eid} | gist: {gist} | created: {created_at}")
        return "\n".join(lines)


    @staticmethod
    def _strip_code_fence(text: str) -> str:
        open_end = text.find("```")
        if open_end == -1:
            return text
        # Skip the opening fence line (```json or ```)
        newline = text.find("\n", open_end)
        if newline == -1:
            return text
        close_start = text.rfind("```", newline)
        if close_start <= newline:
            return text
        return text[newline + 1 : close_start].strip()


    @staticmethod
    def _safe_json_load(text: str) -> list[object]:
        import json as _json

        if not text:
            return []
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = Transcript._strip_code_fence(text)
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, list):
                return parsed
            logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned non-list JSON")
            return []
        except ValueError:
            logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned unparseable JSON")
            return []


    @staticmethod
    def _is_delete_only(ep: dict[str, object]) -> bool:
        if not ep.get('delete_id'):
            return False
        # All other meaningful fields must be absent or null/empty
        meaningful = ('gist', 'transcript_ids', 'update_id')
        return not any(ep.get(f) for f in meaningful)


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
                                        location_lat, location_lon, location_name, turn_id)
                VALUES (?, ?, ?, 1, ?, ?, ?,
                        COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1
                                     FROM transcript WHERE channel = ?)))
                """,
                (channel, role, content, lat, lon, loc_name, turn_id, channel),
            )
            row_id = cursor.lastrowid
            cursor.close()

        Transcript._maybe_trigger_extraction(channel, row_id)

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
        browser-only blob: URL, so on reload the rebuild (api.conversation
        .serialize_turn) joins this table to re-render the image/file from
        /documents/<id>/preview.  INSERT OR IGNORE keeps it idempotent against the
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
                                        location_lat, location_lon, location_name, turn_id)
                VALUES (?, ?, ?, 1, ?, ?, ?,
                        COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1
                                     FROM transcript WHERE channel = ?)))
                """,
                (channel, 'assistant', content, lat, lon, loc_name, turn_id, channel),
            )
            row_id = cursor.lastrowid
            cursor.close()

        Transcript._maybe_trigger_extraction(channel, row_id)

        return cast("int", row_id)

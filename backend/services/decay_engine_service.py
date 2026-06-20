
import math
import logging
import sqlite3

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ._fts_delete import fts5_external_delete
from .time_utils import utc_now, parse_utc


logger = logging.getLogger(__name__)

# ── Relevance time-constants (τ) per hierarchy level ──────────────────────────
# rw(t) = (salience / 10) × exp(−Δt_hours / τ_level), Δt from last_relevant_at.
# Larger τ = slower decay.  Higher levels of the hierarchy are summaries of many
# leaves and are deliberately stickier; tombstoned rows decay fast toward
# deletion regardless of their original level.  Literature-informed defaults.
_TAU_HOURS_LEAF = 14 * 24          # level 0 — raw episodes, ~2 weeks
_TAU_HOURS_LEVEL_1 = 90 * 24       # level 1 — super-episodes, ~3 months
_TAU_HOURS_LEVEL_2 = 365 * 24      # level 2+ — era digests, ~1 year
_TAU_HOURS_TOMBSTONED = 7 * 24     # override for tombstoned rows, ~1 week

# ── Deletion windows ──────────────────────────────────────────────────────────
# Tombstoned episodes are hard-deleted once they have sat tombstoned this long.
_TOMBSTONE_DELETE_AFTER_DAYS = 30
# A never-consolidated leaf is hard-deleted when it is both old and weak: its
# weight has fallen below the floor, it is older than the age window, and its
# salience is at or below the junk threshold.
_LEAF_DELETE_RW_FLOOR = 0.05
_LEAF_DELETE_AGE_DAYS = 90
_LEAF_DELETE_SALIENCE_MAX = 3

# ── Janitor ───────────────────────────────────────────────────────────────────
# Leaf episodes on MUTED/legacy channels are fossils: nothing produces them on
# the current hot path, and they never get consolidated or re-confirmed.  The
# janitor tombstones any older than this so they enter the normal deletion path.
# HEAVY channels (user, dmn, external-agent:*) are protected via the shared
# source-profile allowlist — see _janitor_fossil_episodes.
# NOTE: the design plan leaves this threshold ("X") open; 7 days is chosen as a
# conservative default pending calibration.
_JANITOR_FOSSIL_AGE_DAYS = 7

# ── Tool-call retention ───────────────────────────────────────────────────────
# Durable tool_calls rows are time-capped: rows older than this are purged each
# decay cycle, replacing the old count-based (25 k) cap.
_TOOL_CALLS_RETENTION_DAYS = 7

# Below this absolute delta a recomputed weight is treated as unchanged, so an
# already-settled corpus produces zero UPDATEs on a repeat tick.
_RW_EPSILON = 0.0001

# Salience ceiling: salience is stored as INTEGER 1..10 (schema.sql CHECK and
# salience_service.compute_salience), so the salience→weight ratio divides by 10.
_MAX_SALIENCE = 10

# Legacy data_graph ``kind='moment'`` rows left by the now-deleted 6h moment
# worker. Moments moved to the dedicated ``moments`` table, so these are wiped on
# every cycle. Standing (not one-shot): after the wipe no producer remains, so
# subsequent passes match zero rows and the step is a no-op.
_LEGACY_MOMENT_KIND = "moment"
# Legacy moment rows were indexed in data_graph_fts (external-content over
# data_graph) with the column order (key, value, kind, search_queries); the
# FTS5 external-delete idiom needs those values, read BEFORE the base row goes.
_DG_FTS_TABLE = "data_graph_fts"

# parse_utc returns this exact value when an anchor string is unparseable; we
# detect it to skip corrupt rows loudly instead of decaying them to 0.0.
_PARSE_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

# Hierarchy levels used to pick τ.
_LEVEL_LEAF = 0
_LEVEL_1 = 1


@dataclass(frozen=True)
class _EpisodeDecayRow:

    episode_id: str
    retrieval_weight: float
    salience: int
    level: int
    anchor: str
    tombstoned_at: str


class DecayEngineService:

    def __init__(self) -> None:
        logger.info("[DECAY ENGINE] Initialized (absolute exponential decay)")

    def run_once(self) -> None:
        self.run_decay_cycle()

    def run_decay_cycle(self) -> None:
        fossils_tombstoned = self._janitor_fossil_episodes()
        episodic_count = self._decay_episodic()
        episodes_deleted = self._delete_expired_episodes()
        data_graph_count = self._decay_data_graph()
        legacy_moments_wiped = self._wipe_legacy_data_graph_moments()
        transcript_cleaned = self._cleanup_transcript()
        tool_calls_purged = self._purge_tool_calls()

        logger.info(
            f"[DECAY ENGINE] Cycle complete: "
            f"fossils_tombstoned={fossils_tombstoned}, "
            f"episodic={episodic_count} updated, "
            f"episodes_deleted={episodes_deleted}, "
            f"data_graph={data_graph_count} updated, "
            f"legacy_moments_wiped={legacy_moments_wiped}, "
            f"transcript_cleaned={transcript_cleaned}, "
            f"tool_calls_purged={tool_calls_purged}"
        )

    @staticmethod
    def _tau_hours_for(level: int, tombstoned_at: str | None) -> float:
        if tombstoned_at:
            return _TAU_HOURS_TOMBSTONED
        if level >= _LEVEL_1 + 1:
            return _TAU_HOURS_LEVEL_2
        if level == _LEVEL_1:
            return _TAU_HOURS_LEVEL_1
        return _TAU_HOURS_LEAF

    def _decay_episodic(self) -> int:
        try:
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()
            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, retrieval_weight, salience,
                               COALESCE(level, 0) AS level,
                               COALESCE(last_relevant_at, created_at) AS anchor,
                               tombstoned_at
                        FROM episodes
                        WHERE deleted_at IS NULL
                    """)
                    rows = cursor.fetchall()

                    now = utc_now()
                    updated = 0
                    for raw in rows:
                        row = _EpisodeDecayRow(*raw)
                        new_rw = self._absolute_weight(row, now)
                        if new_rw is None:
                            continue
                        current = row.retrieval_weight if row.retrieval_weight is not None else 0.0
                        if abs(new_rw - current) > _RW_EPSILON:
                            cursor.execute(
                                "UPDATE episodes SET retrieval_weight = ? WHERE id = ?",
                                (new_rw, row.episode_id),
                            )
                            updated += 1

                    cursor.close()
                    if updated > 0:
                        logger.info(f"[DECAY ENGINE] Recomputed {updated} episodic retrieval weights")
                    return updated
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Episodic decay failed: {e}")
            return 0

    def _absolute_weight(self, row: "_EpisodeDecayRow", now: datetime) -> float | None:
        if row.anchor is None:
            return None
        anchor_dt = parse_utc(row.anchor)
        if anchor_dt == _PARSE_SENTINEL:
            logger.warning(
                "[DECAY ENGINE] Unparseable anchor for episode %s (raw=%r); "
                "skipping decay/deletion eligibility",
                row.episode_id, row.anchor,
            )
            return None
        dt_hours = max(0.0, (now - anchor_dt).total_seconds() / 3600.0)
        tau_hours = self._tau_hours_for(int(row.level or 0), row.tombstoned_at)
        salience_ratio = float(row.salience or 0) / _MAX_SALIENCE
        return salience_ratio * math.exp(-dt_hours / tau_hours)

    def _delete_expired_episodes(self) -> int:
        try:
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()
            try:
                with db_service.connection() as conn:
                    now = utc_now()
                    tombstone_cutoff = (now - timedelta(days=_TOMBSTONE_DELETE_AFTER_DAYS)).isoformat()
                    leaf_age_cutoff = (now - timedelta(days=_LEAF_DELETE_AGE_DAYS)).isoformat()

                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id FROM episodes
                        WHERE deleted_at IS NULL
                          AND tombstoned_at IS NOT NULL
                          AND julianday(tombstoned_at) < julianday(?)
                        """,
                        (tombstone_cutoff,),
                    )
                    to_delete = {r[0] for r in cursor.fetchall()}

                    cursor.execute(
                        """
                        SELECT id FROM episodes
                        WHERE deleted_at IS NULL
                          AND consolidated_into IS NULL
                          AND COALESCE(level, 0) = ?
                          AND retrieval_weight < ?
                          AND salience <= ?
                          AND julianday(COALESCE(last_relevant_at, created_at)) < julianday(?)
                        """,
                        (_LEVEL_LEAF, _LEAF_DELETE_RW_FLOOR, _LEAF_DELETE_SALIENCE_MAX, leaf_age_cutoff),
                    )
                    to_delete.update(r[0] for r in cursor.fetchall())

                    deleted = 0
                    for episode_id in to_delete:
                        self._hard_delete_episode(conn, episode_id)
                        deleted += 1
                    cursor.close()

                    if deleted > 0:
                        logger.info(f"[DECAY ENGINE] Hard-deleted {deleted} expired episodes")
                    return deleted
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Episode deletion failed: {e}")
            return 0

    @staticmethod
    def _hard_delete_episode(conn: sqlite3.Connection, episode_id: str) -> None:
        row = conn.execute(
            "SELECT rowid, gist FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is not None:
            rowid, gist = row[0], row[1]
            fts5_external_delete(conn, "episodes_fts", rowid, {"gist": gist})
            conn.execute("DELETE FROM episodes_vec WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

    def _janitor_fossil_episodes(self) -> int:
        # Bidirectional dependency: the per-source allowlist lives in
        # services/source_profiles.py; this is the janitor-protection consumer.
        from .source_profiles import janitor_protected_sql

        try:
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()
            try:
                with db_service.connection() as conn:
                    now = utc_now()
                    fossil_cutoff = (now - timedelta(days=_JANITOR_FOSSIL_AGE_DAYS)).isoformat()
                    cursor = conn.execute(
                        f"""
                        UPDATE episodes
                        SET tombstoned_at = ?
                        WHERE deleted_at IS NULL
                          AND tombstoned_at IS NULL
                          AND consolidated_into IS NULL
                          AND COALESCE(level, 0) = ?
                          AND NOT {janitor_protected_sql()}
                          AND julianday(COALESCE(last_relevant_at, created_at)) < julianday(?)
                        """,
                        (now.isoformat(), _LEVEL_LEAF, fossil_cutoff),
                    )
                    tombstoned = cursor.rowcount or 0
                    if tombstoned > 0:
                        logger.info(f"[DECAY ENGINE] Janitor tombstoned {tombstoned} fossil episodes")
                    return tombstoned
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Fossil janitor failed: {e}")
            return 0

    def _decay_data_graph(self) -> int:
        try:
            from .database_service import get_shared_db_service
            from .data_graph_service import DataGraphService
            db = get_shared_db_service()
            try:
                svc = DataGraphService(db)
                return svc.decay_cycle()
            finally:
                db.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Data graph decay failed: {e}")
            return 0

    def _wipe_legacy_data_graph_moments(self) -> int:
        try:
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()
            try:
                with db_service.connection() as conn:
                    rows = conn.execute(
                        "SELECT rowid, key, value, kind, search_queries FROM data_graph "
                        "WHERE kind = ?",
                        (_LEGACY_MOMENT_KIND,),
                    ).fetchall()
                    for rowid, key, value, kind, search_queries in rows:
                        fts5_external_delete(conn, _DG_FTS_TABLE, rowid, {
                            "key": key,
                            "value": value,
                            "kind": kind,
                            "search_queries": search_queries,
                        })
                        conn.execute("DELETE FROM data_graph WHERE rowid = ?", (rowid,))
                        conn.execute("DELETE FROM data_graph_key_vec WHERE rowid = ?", (rowid,))
                        conn.execute("DELETE FROM data_graph_value_vec WHERE rowid = ?", (rowid,))
                    deleted = len(rows)
                if deleted:
                    logger.info(f"[DECAY ENGINE] Wiped {deleted} legacy data_graph kind='moment' rows")
                return deleted
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Legacy moment wipe failed: {e}")
            return 0

    def _cleanup_transcript(self) -> int:
        try:
            from services.transcript_service import Transcript
            return Transcript.cleanup_unlinked_entries()
        except Exception as e:
            logger.warning(f"[DECAY ENGINE] Transcript cleanup failed: {e}")
            return 0

    def _purge_tool_calls(self) -> int:
        cutoff = (utc_now() - timedelta(days=_TOOL_CALLS_RETENTION_DAYS)).isoformat()
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                result = conn.execute(
                    "DELETE FROM tool_calls WHERE julianday(created_at) < julianday(?)",
                    (cutoff,),
                )
                deleted = result.rowcount
            if deleted:
                logger.info(f"[DECAY ENGINE] Purged {deleted} tool_calls rows older than 7d")
            return deleted
        except Exception as e:
            logger.debug(f"[DECAY ENGINE] Tool calls retention purge non-fatal: {e}")
            return 0


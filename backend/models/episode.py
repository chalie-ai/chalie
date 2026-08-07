"""One ``episodes`` row — the active-record model for episodic memory.

Pure CRUD (Rule-3 depth): holds no ``mp``, never emits WS, never reaches
upstream. It only stores its fields, projects itself (inherited ``to_dict`` /
``to_json``), and runs its own tables' SQL on the connection bound onto the
:class:`Model` base. This model is the SOLE home of ``episodes`` +
``episodes_fts`` + ``episodes_vec`` SQL; the episodic services read and write
exclusively through it.

Rule-3 depth carve-out (precedent: ``data_graph`` imports ``time_utils``): the
only ``services`` reached are pure utilities — ``time_utils`` for the canonical
clock, ``_fts_delete`` for the external-content FTS removal idiom,
``_vec_upsert`` for the vec0 embedding-replace idiom, and
``episodic_constants`` for the novelty comparison-set sizes.

The retrieval lanes (:meth:`fts_search`, :meth:`nearest`) attach two transient
overlay signals — ``text_rank`` / ``vector_distance`` — onto each hit; the
service reranks on them. The decay methods run single autocommit statements and
never open a transaction: the episodic services group their multi-write cycles
through ``Database.transaction()``.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import ClassVar, Self

from contracts.search_config import EPISODE_SEARCH, SearchConfig
from models.model import Model
from models.query import Query
from services._fts_delete import FtsDelete
from services._vec_upsert import VecUpsert
from services.episodic_constants import NOVELTY_ACTIVATION_LIMIT, NOVELTY_RECENT_LIMIT
from services.search_expander_service import SearchExpanderService
from services.time_utils import PARSE_SENTINEL, parse_utc, utc_now

logger = logging.getLogger(__name__)


class Episode(Model):
    """One ``episodes`` row: field storage + CRUD + the retrieval lanes,
    embedding writes, consolidation markers, and the off-spine decay cycle."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "gist", "salience", "channel", "created_at", "updated_at",
        "last_accessed_at", "access_count", "deleted_at", "transcript_ids",
        "transcript_id_start", "transcript_id_end", "consolidated_from",
        "consolidated_into", "retrieval_weight", "location_lat", "location_lon",
        "location_name", "level", "last_relevant_at", "tombstoned_at",
        "facts_extracted_at", "indexed_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "episodes"

    #: The declared search-index footprint (the ``Searchable`` trait). Episodes
    #: carry their vectors synchronously (``EpisodicService`` writes
    #: ``episodes_vec`` on the store path); this config drives only the async FTS
    #: posting through ``SearchExpanderService``. Presence of a config IS the
    #: enablement signal (Rule 5: one authority, no flags) — the save path gates
    #: on it and the write-side engine reads it.
    __search__: ClassVar[SearchConfig | None] = EPISODE_SEARCH

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    gist: str
    salience: int
    channel: str
    created_at: str
    updated_at: str
    last_accessed_at: str | None
    access_count: int
    deleted_at: str | None
    transcript_ids: str | None
    transcript_id_start: int | None
    transcript_id_end: int | None
    consolidated_from: str | None
    consolidated_into: str | None
    retrieval_weight: float | None
    location_lat: float | None
    location_lon: float | None
    location_name: str | None
    level: int
    last_relevant_at: str | None
    tombstoned_at: str | None
    facts_extracted_at: str | None
    indexed_at: str | None

    # Transient retrieval overlays (NOT columns) — the lanes set these on the
    # hit and the service reranks on them; they never persist or project.
    text_rank: float | None
    vector_distance: float | None
    composite_score: float

    # ── Decay time-constants (τ, hours) per hierarchy level ───────────────
    _TAU_HOURS_LEAF: ClassVar[int] = 14 * 24
    _TAU_HOURS_LEVEL_1: ClassVar[int] = 90 * 24
    _TAU_HOURS_LEVEL_2: ClassVar[int] = 365 * 24
    _TAU_HOURS_TOMBSTONED: ClassVar[int] = 7 * 24

    # ── Deletion windows ──────────────────────────────────────────────────
    _TOMBSTONE_DELETE_AFTER_DAYS: ClassVar[int] = 30
    _LEAF_DELETE_RW_FLOOR: ClassVar[float] = 0.05
    _LEAF_DELETE_AGE_DAYS: ClassVar[int] = 90
    _LEAF_DELETE_SALIENCE_MAX: ClassVar[int] = 3
    _JANITOR_FOSSIL_AGE_DAYS: ClassVar[int] = 7

    # Below this absolute delta a recomputed weight is treated as unchanged, so
    # an already-settled corpus produces zero UPDATEs on a repeat tick.
    _RW_EPSILON: ClassVar[float] = 0.0001

    # Salience is stored INTEGER 1..10, so the salience→weight ratio divides by 10.
    _MAX_SALIENCE: ClassVar[int] = 10

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self) -> Self:
        """INSERT a new episode (TEXT-UUID PK generated here), or UPDATE the
        existing row.

        On insert, enqueue the freshly-inserted rowid for the async FTS + vec
        resync through ``SearchExpanderService`` — the same declaration-driven
        engine ``data_graph`` rows flow through — instead of an inline
        ``episodes_fts`` write. The vector lane is unaffected:
        ``EpisodicService`` still writes ``episodes_vec`` synchronously on the
        store path, so vector recall finds a new episode immediately while the
        FTS posting catches up on the worker. A queue-push failure never breaks
        the write; ``_self_heal`` re-indexes any row left with a NULL
        ``indexed_at``."""
        if self.id is not None:
            return super().save()
        self.id = str(uuid.uuid4())
        connection = self._bound_connection()
        columns = self._fields()
        values = [getattr(self, name) for name in columns]
        column_list = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO {self.get_table()} ({column_list}) VALUES ({placeholders})",
            values,
        )
        if self.__search__ is not None and cursor.lastrowid is not None:
            try:
                SearchExpanderService.enqueue(self.get_table(), cursor.lastrowid)
            except Exception:
                logger.warning(
                    "[EPISODE] search-index enqueue failed for id=%s", self.id, exc_info=True
                )
        return self

    # ── Live scope + id reads ─────────────────────────────────────────────

    @classmethod
    def live(cls) -> Query[Self]:
        """The live-episode scope — ``deleted_at IS NULL``. The one shared
        predicate every non-deleted read builds on."""
        return cls.filter("deleted_at", None, "IS")

    @classmethod
    def by_id(cls, episode_id: str) -> Self | None:
        """The single live episode for an exact id, or ``None``."""
        return cls.live().filter("id", episode_id).first()

    @classmethod
    def by_ids(cls, ids: list[str]) -> list[Self]:
        """Live episodes for a set of ids, ``salience DESC`` (mirrors
        ``_fetch_episodes_by_ids``). Empty input short-circuits to ``[]``."""
        if not ids:
            return []
        return cls.live().filter_in("id", ids).order_by("salience DESC").get()

    @classmethod
    def fact_extraction_backlog(cls, limit: int) -> list[Self]:
        """Up to ``limit`` live episodes still awaiting fact extraction,
        oldest-first (``created_at ASC, id ASC``) — the self-healing backlog."""
        return (
            cls.live()
            .filter("facts_extracted_at", None, "IS")
            .order_by("created_at ASC, id ASC")
            .limit(limit)
            .get()
        )

    @classmethod
    def watermark(cls, channel: str) -> int:
        """Highest transcript id already covered by a live episode of
        ``channel`` (0 when none) — the extraction floor."""
        row = cls._bound_connection().execute(
            "SELECT COALESCE(MAX(transcript_id_end), 0) FROM episodes "
            "WHERE channel = ? AND deleted_at IS NULL AND transcript_id_end IS NOT NULL",
            (channel,),
        ).fetchone()
        return int(row[0]) if row else 0

    @classmethod
    def recent_salient(
        cls, channel: str, *, weight_floor: float, lookback_days: int, limit: int
    ) -> list[Self]:
        """Recent, non-decayed episodes of ``channel`` — ``retrieval_weight``
        at or above ``weight_floor`` and touched or created within
        ``lookback_days`` — ranked ``retrieval_weight DESC, created_at DESC``.

        Raw SQL: the touched-or-created OR-across-columns predicate can't be
        expressed by the AND-only structured filter builder."""
        lookback = f"-{lookback_days} days"
        cursor = cls._bound_connection().execute(
            "SELECT * FROM episodes "
            "WHERE deleted_at IS NULL AND channel = ? AND retrieval_weight >= ? "
            "  AND (last_accessed_at >= datetime('now', ?) "
            "       OR created_at >= datetime('now', ?)) "
            "ORDER BY retrieval_weight DESC, created_at DESC LIMIT ?",
            (channel, weight_floor, lookback, lookback, limit),
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

    @classmethod
    def records_page(cls, q: str, limit: int, offset: int) -> list[dict[str, object]]:
        """One page of live episodes for the observability record browser —
        ``{created_at, last_relevant_at, gist, location_name}`` per row, most
        recently relevant first. ``q`` substring-filters the ``gist``; an empty
        ``q`` returns the unfiltered page.

        Raw SQL: the "unfiltered OR substring-match" predicate
        (``? = '' OR gist LIKE ?``) can't be expressed by the AND-only
        structured filter builder."""
        cursor = cls._bound_connection().execute(
            "SELECT created_at, last_relevant_at, gist, location_name "
            "FROM episodes "
            "WHERE deleted_at IS NULL AND (? = '' OR gist LIKE ?) "
            "ORDER BY COALESCE(last_relevant_at, created_at) DESC, created_at DESC "
            "LIMIT ? OFFSET ?",
            (q, f"%{q}%", limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Retrieval lanes ───────────────────────────────────────────────────

    @classmethod
    def fts_search(cls, query_text: str, channel: str | None, k: int) -> list[Self]:
        """FTS5 lane over the ``gist`` column. Sanitises the query to
        alphanumerics + spaces, quotes each term, and returns up to ``k`` hits
        ordered by FTS5 rank, each carrying a ``text_rank`` overlay. An empty
        sanitised query yields ``[]``. Raises on DB failure — the service holds
        the warn-guard."""
        safe = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text or '')
        safe = re.sub(r'\s+', ' ', safe).strip()
        terms = ' '.join(f'"{w}"' for w in safe.split() if w)
        if not terms:
            return []

        sql = (
            "SELECT e.*, episodes_fts.rank AS text_rank "
            "FROM episodes_fts "
            "JOIN episodes e ON e.rowid = episodes_fts.rowid "
            "WHERE episodes_fts MATCH ? AND e.deleted_at IS NULL"
        )
        params: list[object] = [terms]
        if channel is not None:
            sql += " AND e.channel = ?"
            params.append(channel)
        sql += " ORDER BY episodes_fts.rank LIMIT ?"
        params.append(k)

        cursor = cls._bound_connection().execute(sql, params)
        return [cls._lane_hit(row) for row in cursor.fetchall()]

    @classmethod
    def nearest(
        cls,
        blob: bytes,
        channel: str | None,
        k: int,
        *,
        level: int | None = None,
        apex_only: bool = False,
    ) -> list[Self]:
        """Cosine KNN lane via sqlite-vec. Every row filter (live, channel, and
        the optional level/apex ones) is pushed inside the KNN via a rowid
        pre-filter, so ``k`` means "k qualifying hits" — deleted, off-channel,
        or rolled-up rows never consume result slots, no matter how many sit
        nearer to the query. No distance ceiling (the service's relative floor
        decides survivors); returns hits ordered by distance, each carrying a
        ``vector_distance`` overlay. Raises on DB failure — the service holds
        the warn-guard."""
        where = ["deleted_at IS NULL"]
        params: list[object] = [blob, k]
        if channel is not None:
            where.append("channel = ?")
            params.append(channel)
        if level is not None:
            where.append("level = ?")
            params.append(level)
        if apex_only:
            where.append("consolidated_into IS NULL")
        sql = (
            "SELECT e.*, v.distance AS vector_distance "
            "FROM episodes e "
            "JOIN episodes_vec v ON v.rowid = e.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            f"AND v.rowid IN (SELECT rowid FROM episodes WHERE {' AND '.join(where)}) "
            "ORDER BY v.distance"
        )
        cursor = cls._bound_connection().execute(sql, params)
        return [cls._lane_hit(row) for row in cursor.fetchall()]

    @classmethod
    def _lane_hit(cls, row: sqlite3.Row) -> Self:
        """Hydrate one retrieval row: the lane's SELECT alias lands as its
        overlay attr automatically; default the other lane's attr to ``None``
        so every hit exposes both signals for the rerank."""
        episode = cls.hydrate(row)
        if not hasattr(episode, "text_rank"):
            episode.text_rank = None
        if not hasattr(episode, "vector_distance"):
            episode.vector_distance = None
        return episode

    @classmethod
    def located_at(cls, names: list[str], limit: int) -> list[Self]:
        """Live episodes whose ``location_name`` LIKE-matches any of ``names``
        (each wrapped ``%name%``), most-recent-first. Cross-channel by design —
        a location recall surfaces episodes from every producing channel."""
        if not names:
            return []
        like_clauses = " OR ".join(["e.location_name LIKE ?"] * len(names))
        params: list[object] = [f"%{name}%" for name in names]
        params.append(limit)
        cursor = cls._bound_connection().execute(
            "SELECT e.id, e.gist, e.location_name, e.created_at "
            "FROM episodes e "
            f"WHERE e.deleted_at IS NULL AND ({like_clauses}) "
            "AND e.location_name IS NOT NULL "
            "ORDER BY e.created_at DESC LIMIT ?",
            params,
        )
        return [cls.hydrate(row) for row in cursor.fetchall()]

    # ── Embedding / clustering reads ──────────────────────────────────────

    @classmethod
    def set_embedding(cls, episode_id: str, blob: bytes) -> None:
        """Upsert an episode's embedding blob into ``episodes_vec`` keyed by
        rowid. A missing episode is a silent no-op."""
        connection = cls._bound_connection()
        row = connection.execute(
            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row:
            VecUpsert.vec0_upsert(connection, "episodes_vec", row[0], blob)

    @classmethod
    def novelty_comparison_blobs(cls, channel: str) -> list[bytes]:
        """The novelty comparison set: dedup(last-``NOVELTY_RECENT_LIMIT`` by
        ``created_at``) ∪ (top-``NOVELTY_ACTIVATION_LIMIT`` by ``access_count``),
        apex-only, same channel, live — then their non-NULL embedding blobs.
        Never raises: ``[]`` on any failure (novelty degrades to 1.0)."""
        try:
            connection = cls._bound_connection()
            recent = connection.execute(
                "SELECT id FROM episodes "
                "WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (channel, NOVELTY_RECENT_LIMIT),
            ).fetchall()
            recent_ids = {row[0] for row in recent}

            top = connection.execute(
                "SELECT id FROM episodes "
                "WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL "
                "ORDER BY access_count DESC LIMIT ?",
                (channel, NOVELTY_ACTIVATION_LIMIT),
            ).fetchall()
            top_ids = {row[0] for row in top}

            all_ids = list(recent_ids | top_ids)
            if not all_ids:
                return []

            placeholders = ','.join('?' * len(all_ids))
            blob_rows = connection.execute(
                f"SELECT ev.embedding "
                f"FROM episodes_vec ev "
                f"JOIN episodes e ON e.rowid = ev.rowid "
                f"WHERE e.id IN ({placeholders}) "
                f"  AND ev.embedding IS NOT NULL",
                all_ids,
            ).fetchall()
            return [row[0] for row in blob_rows if row[0]]
        except Exception as exc:
            logging.warning(f"[NOVELTY] comparison-set fetch failed: {exc}")
            return []

    # ── Targeted writes ───────────────────────────────────────────────────

    @classmethod
    def update_fields(cls, episode_id: str, fields: dict[str, object]) -> None:
        """Targeted UPDATE of arbitrary columns plus ``updated_at``. Values must
        be sqlite-bindable — JSON columns are serialised by the service. Empty
        ``fields`` is a no-op."""
        if not fields:
            return
        set_clauses = [f"{key} = ?" for key in fields]
        set_clauses.append("updated_at = datetime('now')")
        values = list(fields.values())
        values.append(episode_id)
        cls._bound_connection().execute(
            f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )

    @classmethod
    def soft_delete(cls, episode_id: str) -> None:
        """Mark an episode deleted (idempotent — already-deleted rows unchanged)."""
        cls._bound_connection().execute(
            "UPDATE episodes SET deleted_at = datetime('now') "
            "WHERE id = ? AND deleted_at IS NULL",
            (episode_id,),
        )

    @classmethod
    def set_consolidated_into(cls, leaf_id: str, super_id: str) -> None:
        """Point a leaf at its super-episode and tombstone it in one statement:
        consolidation advances the leaf's relevance clock and collapses it onto
        the fast tombstone τ, so a child can never carry a back-pointer without a
        tombstone."""
        now_iso = utc_now().isoformat()
        cls.filter("id", leaf_id).update(
            consolidated_into=super_id, last_relevant_at=now_iso, tombstoned_at=now_iso
        )

    @classmethod
    def set_facts_extracted_at(cls, episode_id: str) -> None:
        """Stamp the fact-extraction cursor, removing the episode from the
        backlog — a durable progress marker across interrupted ticks."""
        cls.filter("id", episode_id).update(facts_extracted_at=utc_now().isoformat())

    @classmethod
    def hard_delete(cls, episode_id: str) -> None:
        """Physically remove an episode and its FTS + vec index rows. The FTS
        table is external-content over the ``gist`` column, so its posting must
        be deleted with the indexed value — in FTS column order — via the shared
        external-delete idiom (a NULL ``gist`` coerces to '' inside the
        helper)."""
        connection = cls._bound_connection()
        row = connection.execute(
            "SELECT rowid, gist, indexed_at FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is not None:
            rowid, gist, indexed_at = row[0], row[1], row[2]
            if indexed_at is not None:
                FtsDelete.fts5_external_delete(
                    connection, "episodes_fts", rowid,
                    {"gist": gist},
                )
            connection.execute("DELETE FROM episodes_vec WHERE rowid = ?", (rowid,))
        connection.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

    # ── Decay cycle ───────────────────────────────────────────────────────

    @classmethod
    def decay_weights(cls) -> int:
        """Recompute every live episode's ``retrieval_weight`` by absolute
        exponential decay, writing back only when a weight moves past
        ``_RW_EPSILON``. Returns the number of rows updated.

        Each change is a targeted single-column UPDATE (not ``save()``) so a
        concurrent consolidation write to other columns is never clobbered."""
        now = utc_now()
        updated = 0
        for episode in cls.live().get():
            new_weight = episode._decayed_weight(now)
            if new_weight is None:
                continue
            current = episode.retrieval_weight if episode.retrieval_weight is not None else 0.0
            if abs(new_weight - current) > cls._RW_EPSILON:
                cls.filter("id", episode.id).update(retrieval_weight=new_weight)
                updated += 1
        if updated > 0:
            logger.info(f"[DECAY ENGINE] Recomputed {updated} episodic retrieval weights")
        return updated

    def _decayed_weight(self, now: datetime) -> float | None:
        """Absolute decayed weight: ``(salience / 10) × exp(−Δh / τ)`` anchored
        on ``last_relevant_at`` (falling back to ``created_at``). ``None`` when
        the anchor is missing or unparseable — the row is then skipped."""
        anchor = self.last_relevant_at or self.created_at
        if anchor is None:
            return None
        anchor_dt = parse_utc(anchor)
        if anchor_dt == PARSE_SENTINEL:
            logger.warning(
                "[DECAY ENGINE] Unparseable anchor for episode %s (raw=%r); "
                "skipping decay/deletion eligibility",
                self.id, anchor,
            )
            return None
        dt_hours = max(0.0, (now - anchor_dt).total_seconds() / 3600.0)
        salience_ratio = float(self.salience or 0) / self._MAX_SALIENCE
        return salience_ratio * math.exp(-dt_hours / self._tau_hours())

    def _tau_hours(self) -> int:
        """The decay τ (hours) for this row: tombstoned rows collapse onto the
        fast tombstone τ regardless of level; otherwise higher hierarchy levels
        are stickier."""
        if self.tombstoned_at:
            return self._TAU_HOURS_TOMBSTONED
        level = self.level or 0
        if level >= 2:
            return self._TAU_HOURS_LEVEL_2
        if level == 1:
            return self._TAU_HOURS_LEVEL_1
        return self._TAU_HOURS_LEAF

    @classmethod
    def delete_expired(cls) -> int:
        """Hard-delete expired episodes: tombstoned rows aged past the tombstone
        window, plus never-consolidated leaves that are old, weight-floored and
        low-salience. Returns the number deleted."""
        connection = cls._bound_connection()
        now = utc_now()
        tombstone_cutoff = (now - timedelta(days=cls._TOMBSTONE_DELETE_AFTER_DAYS)).isoformat()
        leaf_age_cutoff = (now - timedelta(days=cls._LEAF_DELETE_AGE_DAYS)).isoformat()

        to_delete = {
            row[0]
            for row in connection.execute(
                "SELECT id FROM episodes "
                "WHERE deleted_at IS NULL "
                "  AND tombstoned_at IS NOT NULL "
                "  AND julianday(tombstoned_at) < julianday(?)",
                (tombstone_cutoff,),
            ).fetchall()
        }
        to_delete.update(
            row[0]
            for row in connection.execute(
                "SELECT id FROM episodes "
                "WHERE deleted_at IS NULL "
                "  AND consolidated_into IS NULL "
                "  AND COALESCE(level, 0) = 0 "
                "  AND retrieval_weight < ? "
                "  AND salience <= ? "
                "  AND julianday(COALESCE(last_relevant_at, created_at)) < julianday(?)",
                (cls._LEAF_DELETE_RW_FLOOR, cls._LEAF_DELETE_SALIENCE_MAX, leaf_age_cutoff),
            ).fetchall()
        )
        for episode_id in to_delete:
            cls.hard_delete(episode_id)
        if to_delete:
            logger.info(f"[DECAY ENGINE] Hard-deleted {len(to_delete)} expired episodes")
        return len(to_delete)

    @classmethod
    def tombstone_fossils(cls, protected_sql: str) -> int:
        """Tombstone stale leaf fossils on MUTED/legacy channels so they enter
        the normal deletion path. ``protected_sql`` is the caller's
        source-profile protection predicate (interpolated under ``NOT (...)``);
        the model never reaches into source profiles. Returns rows tombstoned."""
        now = utc_now()
        fossil_cutoff = (now - timedelta(days=cls._JANITOR_FOSSIL_AGE_DAYS)).isoformat()
        cursor = cls._bound_connection().execute(
            f"UPDATE episodes "
            f"SET tombstoned_at = ? "
            f"WHERE deleted_at IS NULL "
            f"  AND tombstoned_at IS NULL "
            f"  AND consolidated_into IS NULL "
            f"  AND COALESCE(level, 0) = 0 "
            f"  AND NOT {protected_sql} "
            f"  AND julianday(COALESCE(last_relevant_at, created_at)) < julianday(?)",
            (now.isoformat(), fossil_cutoff),
        )
        tombstoned = cursor.rowcount or 0
        if tombstoned > 0:
            logger.info(f"[DECAY ENGINE] Janitor tombstoned {tombstoned} fossil episodes")
        return tombstoned

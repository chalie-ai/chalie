"""The ``memory_map`` active-record model — episodic lineage searched by
situational cues (Memory v3).

:class:`MemoryMapRow` is a thin persistent shell for the ``memory_map``
table: field storage, CRUD, the post-commit vec write-sync, and a
pool-membership helper. Unlike :class:`~models.data_graph.DataGraphRow`, map
rows are not keyed by a business unique constraint — each row has a
surrogate ``id`` and carries a ``derived_from`` JSON array that references
other map rows (lineage).

Pure CRUD (Rule-3 depth): holds no ``mp``, never emits WS, never reaches
upstream. The only ``services`` seam is the module-level
:func:`services.search_expander_service.enqueue` — a lightweight queue push,
not a stateful service object — that drives the async vec resync (the model
owns the sync trigger, mirroring ``Episode.save``'s inline FTS write). This
model is the SOLE home of ``memory_map`` SQL (I6).
"""

from __future__ import annotations

import logging
from typing import ClassVar, Self, cast

from contracts.search_config import MAP_SEARCH, SearchConfig
from models.model import Model
from services.embedding_utils import pack_embedding
from services.search_expander_service import enqueue
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class MemoryMapRow(Model):
    """One ``memory_map`` row: field storage + CRUD + the post-commit vec
    write-sync and the searchable-pool helper."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "created_at", "generated_at", "contents", "source", "cues",
        "derived_from", "sourced_from", "iteration", "indexed_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "memory_map"

    #: The declared search-index footprint (the ``Searchable`` trait). Map rows
    #: are indexed purely by vector — no FTS column. Presence of a config IS
    #: the enablement signal: the save path gates on ``self.__search__ is not
    #: None`` and the write-side engine reads it.
    __search__: ClassVar[SearchConfig | None] = MAP_SEARCH

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    created_at: str
    generated_at: str
    contents: str
    #: The two columns a caller may legitimately omit, so they carry the same
    #: empty default the schema does. Class defaults are invisible to
    #: ``_fields()`` (which reads ``__dict__``), so an unset column is still
    #: omitted from the INSERT and the SQL default still fires — this only
    #: spares every read site a ``getattr`` guard against an attribute that a
    #: hand-built row never set.
    source: str = ""
    cues: str = ""
    derived_from: str
    sourced_from: str
    iteration: int
    indexed_at: str | None

    # ── Persistence + write-sync ─────────────────────────────────────────

    def save(self) -> Self:
        """INSERT a new row, or UPDATE the existing one, then — only on a
        content-creating insert (a fresh ``id`` assigned) — enqueue the async
        vec resync through ``SearchExpanderService``. A queue-push failure
        never breaks the write; ``_self_heal`` re-indexes any row left with a
        NULL ``indexed_at``."""
        is_insert = self.id is None
        if not getattr(self, "derived_from", ""):
            self.derived_from = "[]"
        if not getattr(self, "sourced_from", ""):
            self.sourced_from = "[]"
        if not getattr(self, "iteration", 0):
            self.iteration = 1
        if is_insert:
            self.created_at = utc_now().isoformat()
            self.generated_at = self.created_at
        super().save()
        if is_insert and self.id is not None:
            self._sync_search_index()
        return self

    def _sync_search_index(self) -> None:
        """Enqueue this freshly-inserted row for the async vec backfill. The
        model owns only the trigger; a queue-push failure never breaks the
        write."""
        if self.__search__ is None:
            return
        try:
            enqueue(self.get_table(), cast("int", self.id))
        except Exception:
            logger.warning(
                "[MEMORY MAP] search-index enqueue failed for id=%s",
                self.id, exc_info=True,
            )

    # ── Reads ─────────────────────────────────────────────────────────────

    @classmethod
    def generated_in_window(cls, start_iso: str, end_iso: str) -> list[Self]:
        """Rows whose ``generated_at`` falls in the half-open interval
        ``[start_iso, end_iso)`` — plain lexicographic comparison is correct
        because both bounds are ISO-8601 UTC text and the column is TEXT."""
        return (
            cls.all()
            .filter("generated_at", start_iso, operator=">=")
            .filter("generated_at", end_iso, operator="<")
            .order_by("generated_at ASC")
            .get()
        )

    @classmethod
    def earliest_created_at(cls) -> str | None:
        """The minimum ``created_at`` across the table, or ``None`` when the
        table is empty."""
        row = cls._bound_connection().execute(
            "SELECT MIN(created_at) FROM memory_map"
        ).fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    @classmethod
    def by_id(cls, row_id: int) -> Self | None:
        """The single map row for an exact ``id``."""
        return cls.filter("id", row_id).first()

    @classmethod
    def records_page(
        cls, q: str, limit: int, offset: int
    ) -> list[dict[str, object]]:
        """One page of map rows for the observability record browser —
        ``{created_at, generated_at, contents, source}`` per row, most
        recently generated first. ``q`` substring-filters ``contents``; an
        empty ``q`` returns the unfiltered page. ``cues`` stay out: the browser
        shows what a memory is, and the tags it is found by are not that.

        Raw SQL: the "unfiltered OR substring-match" predicate can't be
        expressed by the AND-only structured filter builder."""
        conn = cls._bound_connection()
        cursor = conn.execute(
            "SELECT created_at, generated_at, contents, source "
            "FROM memory_map "
            "WHERE (? = '' OR contents LIKE ?) "
            "ORDER BY generated_at DESC, created_at DESC "
            "LIMIT ? OFFSET ?",
            (q, f"%{q}%", limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Pool membership ───────────────────────────────────────────────────

    @classmethod
    def searchable_pool(cls) -> list[Self]:
        """Rows that are SEARCHABLE — i.e. no other row's ``derived_from``
        JSON array contains this row's ``id``. A row referenced as a lineage
        parent by another row is considered a dependency, not a top-level
        entry in the searchable corpus."""
        conn = cls._bound_connection()
        rows = conn.execute(
            "SELECT m.* FROM memory_map m "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM memory_map other, json_each(other.derived_from) "
            "  WHERE json_each.value = m.id"
            ")"
            " ORDER BY generated_at DESC"
        ).fetchall()
        return [cls.hydrate(row) for row in rows]

    @classmethod
    def vec_knn(cls, query_embedding: list[float] | None, k: int) -> dict[int, float]:
        """Vec0 KNN over ``memory_map_cues_vec`` for recall. Returns
        ``{rowid: distance}`` for the top-``k`` nearest CUE embeddings — the
        episode text and its ``source`` are not indexed, so a match means the
        query situation resembles the situations the memory is about. Callers
        restrict to the searchable pool (:meth:`searchable_pool`) and re-rank by
        iteration DESC, distance ASC. Non-fatal -> empty dict."""
        if k <= 0:
            return {}
        blob = pack_embedding(query_embedding) if query_embedding else None
        if not blob:
            return {}
        try:
            cursor = cls._bound_connection().execute(
                "SELECT v.rowid, v.distance FROM memory_map_cues_vec v "
                "WHERE v.embedding MATCH ? AND k = ? "
                "ORDER BY v.distance",
                (blob, k),
            )
            return {int(rowid): float(dist) for rowid, dist in cursor.fetchall()}
        except Exception:
            logger.debug(
                "[MEMORY MAP] vec KNN failed (non-fatal)", exc_info=True,
            )
            return {}

    @classmethod
    def max_iteration(cls, ids: list[int]) -> int:
        """Highest ``iteration`` among the given map ids (0 if none/missing).
        ``save_map`` uses this to set a new row's iteration = max(parents) + 1."""
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        row = cls._bound_connection().execute(
            f"SELECT COALESCE(MAX(iteration), 0) FROM memory_map WHERE id IN ({placeholders})",
            ids,
        ).fetchone()
        return int(row[0]) if row is not None else 0

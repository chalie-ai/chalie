"""The ``memory_graph`` active-record model — subject-keyed living facts
(Memory v3).

:class:`MemoryGraphRow` is a thin persistent shell for the ``memory_graph``
table: field storage, CRUD, the post-commit FTS write-sync, and the subject-
keyed upsert path. Unlike :class:`~models.data_graph.DataGraphRow`, graph rows
are keyed by ``subject`` (unique) rather than by ``(kind, key)`` — an upsert on
``subject`` appends new provenance to ``sourced_from`` and refreshes
``last_updated_at`` on each re-write.

Pure CRUD (Rule-3 depth): holds no ``mp``, never emits WS, never reaches
upstream. The only ``services`` seam is the module-level
:func:`services.search_expander_service.enqueue` — a lightweight queue push,
not a stateful service object — that drives the async FTS resync (the model
owns the sync trigger, mirroring ``Episode.save``'s inline FTS write). This
model is the SOLE home of ``memory_graph`` SQL (I6).
"""

from __future__ import annotations

import json
import logging
from typing import ClassVar, Self, cast

from contracts.search_config import GRAPH_SEARCH, SearchConfig
from models.model import Model
from models.query import Query
from services.search_expander_service import enqueue
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class MemoryGraphRow(Model):
    """One ``memory_graph`` row: field storage + CRUD + the subject-keyed
    upsert path and the post-commit FTS write-sync."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "created_at", "last_updated_at", "subject", "contents",
        "sourced_from", "indexed_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "memory_graph"

    #: The declared search-index footprint (the ``Searchable`` trait).
    __search__: ClassVar[SearchConfig | None] = GRAPH_SEARCH

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    created_at: str
    last_updated_at: str
    subject: str
    contents: str
    sourced_from: str
    indexed_at: str | None

    # ── Persistence + write-sync ─────────────────────────────────────────

    def save(self) -> Self:
        """Subject-keyed upsert: if a row with this ``subject`` exists, UPDATE
        it (refresh ``contents`` and ``last_updated_at``, merge new provenance
        into ``sourced_from``); otherwise INSERT a new row.

        The FTS resync is enqueued ONLY on a genuine fresh INSERT — the FTS
        index is over ``subject`` (which never changes on update), so updates
        need no resync. A queue-push failure never breaks the write;
        ``_self_heal`` re-indexes any row left with a NULL ``indexed_at``.
        """
        if not getattr(self, "sourced_from", ""):
            self.sourced_from = "[]"
        conn = self._bound_connection()
        now = utc_now().isoformat()

        existing = conn.execute(
            "SELECT id, sourced_from FROM memory_graph WHERE subject = ?",
            (self.subject,),
        ).fetchone()

        if existing is not None:
            rowid, existing_sourced = existing
            try:
                base_ids = json.loads(str(existing_sourced or "[]"))
            except (json.JSONDecodeError, TypeError):
                base_ids = []
            try:
                new_ids = json.loads(str(self.sourced_from or "[]"))
            except (json.JSONDecodeError, TypeError):
                new_ids = []
            seen: set[object] = set(base_ids)
            merged = base_ids + [x for x in new_ids if x not in seen]
            merged_json = json.dumps(merged)
            conn.execute(
                "UPDATE memory_graph SET contents = ?, sourced_from = ?, "
                "last_updated_at = ? WHERE subject = ?",
                (self.contents, merged_json, now, self.subject),
            )
            self.id = rowid
        else:
            if self.sourced_from == "":
                self.sourced_from = "[]"
            self.created_at = now
            self.last_updated_at = now
            cursor = conn.execute(
                "INSERT INTO memory_graph "
                "(subject, contents, sourced_from, created_at, last_updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.subject, self.contents, self.sourced_from,
                 self.created_at, self.last_updated_at),
            )
            self.id = cursor.lastrowid
            if self.__search__ is not None:
                try:
                    enqueue(self.get_table(), cast("int", self.id))
                except Exception:
                    logger.warning(
                        "[MEMORY GRAPH] search-index enqueue failed for id=%s",
                        self.id, exc_info=True,
                    )
        return self

    # ── Reads ─────────────────────────────────────────────────────────────

    @classmethod
    def by_subject(cls, subject: str) -> Self | None:
        """The single live row for an exact ``subject`` (the store lookup)."""
        return cls.filter("subject", subject).first()

    @classmethod
    def fts_subject_search(cls, terms: str, k: int) -> list[Self]:
        """FTS5 recall over ``subject``. ``terms`` is an FTS5 query string
        (callers build it from extracted keywords). Returns the top ``k`` rows
        by BM25 rank (FTS5 ``rank`` — lower is better). Non-fatal on FTS
        errors -> empty list."""
        if not terms or k <= 0:
            return []
        try:
            cursor = cls._bound_connection().execute(
                "SELECT m.* FROM memory_graph_fts f "
                "JOIN memory_graph m ON m.rowid = f.rowid "
                "WHERE memory_graph_fts MATCH ? "
                "ORDER BY f.rank LIMIT ?",
                (terms, k),
            )
            return [cls.hydrate(row) for row in cursor.fetchall()]
        except Exception:
            logger.debug(
                "[MEMORY GRAPH] FTS subject search failed (non-fatal)",
                exc_info=True,
            )
            return []

    @classmethod
    def delete_by_subject(cls, subject: str) -> int:
        """Hard-delete the live row for ``subject``. Returns rows deleted (0/1).
        FTS external-content postings are orphaned harmlessly — recall JOINs the
        base table, so a dangling posting surfaces nothing."""
        cur = cls._bound_connection().execute(
            "DELETE FROM memory_graph WHERE subject = ?", (subject,)
        )
        return int(cur.rowcount or 0)

    @classmethod
    def records_page(
        cls, q: str, limit: int, offset: int
    ) -> list[dict[str, object]]:
        """One page of graph rows for the observability record browser —
        ``{created_at, last_updated_at, subject, contents}`` per row, most
        recently updated first. ``q`` substring-filters ``subject`` or
        ``contents``; an empty ``q`` returns the unfiltered page.

        Raw SQL: the "unfiltered OR substring-match" predicate across two
        columns can't be expressed by the AND-only structured filter builder."""
        conn = cls._bound_connection()
        cursor = conn.execute(
            "SELECT created_at, last_updated_at, subject, contents "
            "FROM memory_graph "
            "WHERE (? = '' OR subject LIKE ? OR contents LIKE ?) "
            "ORDER BY last_updated_at DESC, created_at DESC "
            "LIMIT ? OFFSET ?",
            (q, f"%{q}%", f"%{q}%", limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def iterate(cls) -> Query[Self]:
        """Bulk reader: all graph rows, ordered by insertion time. Returns a
        lazy :class:`~models.query.Query` — callers drive iteration themselves
        to avoid loading the whole table into memory at once."""
        return cls.all().order_by("created_at ASC")

    # ── Projection ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        d = super().to_dict()
        # Ensure sourced_from is always a valid JSON array string.
        sourced = d.get("sourced_from")
        if sourced is not None:
            try:
                json.loads(str(sourced))  # validate
            except (json.JSONDecodeError, TypeError):
                d["sourced_from"] = "[]"
        return d

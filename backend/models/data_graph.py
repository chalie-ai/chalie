"""The ``data_graph`` active-record models — one shared shell, two supersede
shapes.

:class:`DataGraphRow` is the abstract base every kind subclasses: it owns
the table binding, the column set, the kind-bound live/FTS reads (scoped to
the subclass's :attr:`~DataGraphRow.KIND`) and the post-commit FTS/vec
write-sync. Whether a kind supersedes a contradicting value at the same key
is structural rather than a membership test:
:class:`~models.exact_key.ExactKeyRow` is an abstract intermediate rung that
adds the exact-key supersede machinery (``store``/``demote``) on top of the
shared shell, and a superseding kind (e.g.
:class:`~models.machine_state.MachineStateRow`) subclasses it; a
non-superseding kind (e.g. :class:`~models.contact.ContactRow`) subclasses
:class:`DataGraphRow` directly instead and carries its own non-superseding
``store``.

Pure CRUD (Rule-3 depth): holds no ``mp``, never emits WS, never reaches
upstream. The single sanctioned ``services`` seam is the module-level
:func:`services.search_expander_service.enqueue` — a lightweight queue push,
not a stateful service object — that drives the async FTS + key/value-vec
resync (the model owns the sync *trigger*, mirroring ``Episode.save``'s
inline FTS write). These models are the SOLE home of ``data_graph`` SQL
(I6); the services read and write exclusively through them.

The live lane reads apply ``active = 1 AND deleted_at IS NULL``. The
exact-key store lookup — in :class:`~models.exact_key.ExactKeyRow` and
:class:`~models.contact.ContactRow`, not this class — is ``active = 1``
only, deliberately not ``deleted_at``-filtered.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from typing import ClassVar, Self, cast

from contracts.search_config import DATA_GRAPH_SEARCH, SearchConfig, register_kind
from models.model import Model
from models.query import Query
from services._fts_delete import fts5_external_delete
from services.search_expander_service import enqueue
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class DataGraphRow(Model):
    """The shared ``data_graph`` shell: field storage + CRUD + the reinforce
    path, the kind-bound reads a concrete vertical inherits, and the post-commit
    FTS/vec write-sync. Concrete verticals set :attr:`KIND` and read their own
    lane with a bare ``live()`` / ``search()``; :meth:`live` also accepts an
    explicit ``kind``, for a cross-kind read or a class that never set
    :attr:`KIND`."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "kind", "key", "value", "storage_strength", "retrieval_weight",
        "salience_score", "evidence_count", "first_seen_at", "last_confirmed_at",
        "last_accessed_at", "source", "deleted_at", "active", "indexed_at",
        "valid_from", "valid_to",
    )

    @classmethod
    def get_table(cls) -> str:
        return "data_graph"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    kind: str
    key: str
    value: str | None
    storage_strength: float
    retrieval_weight: float
    salience_score: float
    evidence_count: int
    first_seen_at: str
    last_confirmed_at: str
    last_accessed_at: str | None
    source: str | None
    deleted_at: str | None
    active: int
    indexed_at: str | None
    valid_from: str | None
    valid_to: str | None

    #: The single ``data_graph.kind`` a concrete vertical owns — abstract on the
    #: base (subclasses set it); the kind-bound reads below scope to it.
    KIND: ClassVar[str]

    #: The declared search-index footprint (the ``Searchable`` trait). Full
    #: ``data_graph`` config by default — inherited by every searchable vertical;
    #: non-searchable kinds (``contact``, ``machine_state``) override
    #: to ``None``. Presence of a config IS the enablement signal: the save path
    #: gates on ``self.__search__ is not None`` and the write-side engine reads
    #: it (no flags, no policy table — one authority per Rule 5).
    __search__: ClassVar[SearchConfig | None] = DATA_GRAPH_SEARCH

    # Evidence-diminishing reinforcement boost numerator (``_reinforce_row``).
    _REINFORCE_BOOST: ClassVar[float] = 0.05

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register each concrete vertical's ``KIND → __search__`` mapping so the
        write-side engine can resolve a raw ``kind`` string (from a SQL row, with
        no model instance) to its declared config. Only classes that directly
        define ``KIND`` register — abstract rungs (``ExactKeyRow``) are skipped;
        ``__search__`` resolves through the MRO, so an override or an inherited
        default is picked up correctly."""
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("KIND")
        if kind is not None:
            register_kind(kind, cls.__search__)

    # ── persistence + write-sync ─────────────────────────────────────────

    def save(self) -> Self:
        """INSERT a new row or UPDATE the existing one, then — only on a
        content-creating insert (a fresh ``id`` assigned, including a
        superseding successor) — enqueue the async FTS + key/value-vec
        resync.

        A pure :meth:`reinforce` of an identical value keeps the existing ``id``
        and unchanged FTS content, so it skips the sync — mirroring the way
        ``Episode.save`` writes its FTS row only on insert."""
        is_insert = self.id is None
        super().save()
        if is_insert and self.id is not None:
            self._sync_search_index()
        return self

    def _sync_search_index(self) -> None:
        """Enqueue this freshly-inserted row for the async FTS + key/value-vec
        backfill. The model owns only the trigger; a queue-push failure never
        breaks the write.

        Enablement is the ``Searchable`` trait: a kind with ``__search__ = None``
        (behavioural patterns, ``machine_state``) declares no search footprint
        and never enters the index. The write-side ``_self_heal`` consults the
        same declaration via the kind→config registry, so neither enqueue path
        indexes a non-searchable row."""
        if self.__search__ is None:
            return
        try:
            enqueue("data_graph", cast("int", self.id))
        except Exception:
            logger.warning(
                "[DATA GRAPH] search-index enqueue failed for id=%s", self.id, exc_info=True
            )

    @classmethod
    def _purge_search_index(
        cls, conn: sqlite3.Connection, rowid: int, indexed: dict[str, object]
    ) -> None:
        """Tear down one row's search-index footprint before its base
        ``data_graph`` row is deleted: the external-content FTS posting (FTS5
        must be told the indexed values *before* the source row disappears) and
        every declared key/value-vec shadow.

        Config-driven via the ``Searchable`` trait (``cls.__search__``) — the
        same declaration the save path gates on — so teardown reads the fts
        table + column order and the vec-lane tables from one authority instead
        of hard-coded literals duplicated per vertical. A non-searchable kind
        holds no postings, so this is a no-op. ``indexed`` supplies the FTS
        content-column values captured from the row before deletion, keyed by
        column name; a missing column is coerced to empty inside the delete."""
        config = cls.__search__
        if config is None:
            return
        fts5_external_delete(
            conn, config.fts_table, rowid,
            {col: cast("str | None", indexed.get(col)) for col in config.fts_columns},
        )
        for lane in config.vec_lanes:
            conn.execute(f"DELETE FROM {lane.table} WHERE rowid = ?", (rowid,))

    # ── the shared live predicate + kind-bound reads ─────────────────────

    @classmethod
    def live(cls, kind: str | None = None) -> Query[Self]:
        """The live-fact scope — ``active = 1 AND deleted_at IS NULL`` for one
        ``kind``. The one shared live predicate every lane builds on. ``kind``
        defaults to this class's :attr:`KIND` so a concrete vertical reads its
        own lane with a bare ``live()``; passing ``kind`` explicitly instead
        reads a specific kind independent of the class it's called on — for a
        cross-kind read or a class that never set :attr:`KIND`."""
        resolved = kind if kind is not None else cls.KIND
        return cls.filter("kind", resolved).filter("active", 1).filter("deleted_at", None, "IS")

    @classmethod
    def search(cls, query: str, k: int) -> list[Self]:
        """Up to ``k`` FTS5 candidate rows of this kind for ``query``, ranked by
        FTS rank — the per-vertical single-kind read the cross-kind recall
        service fuses across verticals later. The query is sanitised to word
        characters and each term prefix-matched; an empty query, no candidates,
        or any DB error yields ``[]``. Calling this on an abstract rung that
        never set :attr:`KIND` raises loudly (a misuse), never a silent empty
        result."""
        kind = cls.KIND  # loud AttributeError on an abstract rung that never
        # set KIND, before the try below can swallow it into a "no results" []
        safe = re.sub(r"[^\w\s]", " ", query or "")
        terms = " OR ".join(f'"{w}"*' for w in safe.split() if w)
        if not terms:
            return []
        try:
            cursor = cls._bound_connection().execute(
                "SELECT d.* FROM data_graph_fts "
                "JOIN data_graph d ON d.rowid = data_graph_fts.rowid "
                "WHERE data_graph_fts MATCH ? AND d.kind = ? "
                "AND d.active = 1 AND d.deleted_at IS NULL "
                "ORDER BY data_graph_fts.rank LIMIT ?",
                (terms, kind, k),
            )
            return [cls.hydrate(row) for row in cursor.fetchall()]
        except Exception:
            logger.debug(
                "[DATA GRAPH] FTS search failed (non-fatal) for kind=%s", kind, exc_info=True
            )
            return []

    @classmethod
    def records_page(
        cls, kind: str, q: str, limit: int, offset: int
    ) -> list[dict[str, object]]:
        """One page of live facts of ``kind`` for the observability record
        browser — ``{created, last_accessed, key, value}`` per row, most
        recently accessed first (NULL access last). ``q`` substring-filters
        ``key`` or ``value``; an empty ``q`` returns the unfiltered page.

        Raw SQL: the "unfiltered OR substring-match" predicate across two
        columns (``? = '' OR key LIKE ? OR value LIKE ?``) plus the
        NULL-last ordering can't be expressed by the AND-only structured
        filter builder."""
        cursor = cls._bound_connection().execute(
            "SELECT first_seen_at AS created, last_accessed_at AS last_accessed, "
            "key, value "
            "FROM data_graph "
            "WHERE kind = ? AND active = 1 AND deleted_at IS NULL "
            "AND (? = '' OR key LIKE ? OR value LIKE ?) "
            "ORDER BY last_accessed_at IS NULL, last_accessed_at DESC, first_seen_at DESC "
            "LIMIT ? OFFSET ?",
            (kind, q, f"%{q}%", f"%{q}%", limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── reinforce ────────────────────────────────────────────────────────

    def reinforce(self) -> Self:
        """Re-confirm this row on a repeated identical fact (``_reinforce_row``):
        bump ``evidence_count``, apply the evidence-diminishing storage boost,
        reset ``retrieval_weight`` to 1.0, and stamp the confirm/access times."""
        now = utc_now().isoformat()
        self.evidence_count += 1
        boost = self._REINFORCE_BOOST / math.log2(self.evidence_count + 1)
        self.storage_strength = min(1.0, self.storage_strength + boost)
        self.retrieval_weight = 1.0
        self.last_confirmed_at = now
        self.last_accessed_at = now
        return self.save()

"""The ``data_graph`` active-record models — one shared shell, two shapes.

:class:`DataGraphRow` is the abstract base every per-kind vertical subclasses:
it owns the table binding, the column set, the reinforce path, the kind-bound
live/exact-key/FTS reads (scoped to the subclass's :attr:`~DataGraphRow.KIND`)
and the post-commit FTS/vec write-sync. :class:`DataGraph` is the thin generic
gateway that still spans every not-yet-ported kind: it leaves ``KIND`` unset and
overrides the kind-bound reads with kind-parametrised ones, and carries the
supersede machinery (``store``/``demote``/``_SUPERSEDING_KINDS``) and the
``system``/``user_specific`` lanes.

Pure CRUD (Rule-3 depth): holds no ``mp``, never emits WS, never reaches
upstream. The single sanctioned ``services`` seam is the module-level
:func:`services.search_expander_service.enqueue` — a lightweight queue push, not
a stateful service object — that drives the async FTS + key/value-vec + doc2query
resync (the model owns the sync *trigger*, mirroring ``Episode.save``'s inline
FTS write). These models are the SOLE home of ``data_graph`` SQL (I6); the
services read and write exclusively through them.

Live-fact filters mirror today's code verbatim: the lane reads mirror
``DataGraphService.fetch`` (``active = 1 AND deleted_at IS NULL``); the exact-key
store lookup mirrors ``_SELECT_ACTIVE_BY_KIND_KEY_SQL`` (``active = 1`` only, no
``deleted_at``).
"""

from __future__ import annotations

import logging
import math
import re
from typing import ClassVar, Self, cast

from models.model import Model
from models.query import Query
from services.search_expander_service import enqueue
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class DataGraphRow(Model):
    """The shared ``data_graph`` shell: field storage + CRUD + the reinforce
    path, the kind-bound reads a concrete vertical inherits, and the post-commit
    FTS/vec write-sync. Concrete verticals set :attr:`KIND` and read their own
    lane with a bare ``live()`` / ``search()``; the generic :class:`DataGraph`
    gateway leaves ``KIND`` unset and passes the kind explicitly to
    :meth:`live`."""

    __table__: ClassVar[str] = "data_graph"
    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "kind", "key", "value", "storage_strength", "retrieval_weight",
        "salience_score", "evidence_count", "first_seen_at", "last_confirmed_at",
        "last_accessed_at", "source", "deleted_at", "active", "search_queries",
        "valid_from", "valid_to",
    )

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
    search_queries: str | None
    valid_from: str | None
    valid_to: str | None

    #: The single ``data_graph.kind`` a concrete vertical owns — abstract on the
    #: base (subclasses set it); the kind-bound reads below scope to it.
    KIND: ClassVar[str]

    # Evidence-diminishing reinforcement boost numerator (``_reinforce_row``).
    _REINFORCE_BOOST: ClassVar[float] = 0.05

    # ── persistence + write-sync ─────────────────────────────────────────

    def save(self) -> Self:
        """INSERT a new row or UPDATE the existing one, then — only on a
        content-creating insert (a fresh ``id`` assigned, including a
        superseding successor) — enqueue the async FTS + key/value-vec +
        doc2query resync.

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
        backfill + doc2query variants (the resync the old
        ``DataGraphService.store`` fired after commit — regression R2). The
        model owns only the trigger; a queue-push failure never breaks the
        write."""
        try:
            enqueue("data_graph", cast("int", self.id))
        except Exception:
            logger.warning(
                "[DATA GRAPH] search-index enqueue failed for id=%s", self.id, exc_info=True
            )

    # ── the shared live predicate + kind-bound reads ─────────────────────

    @classmethod
    def live(cls, kind: str | None = None) -> Query[Self]:
        """The live-fact scope — ``active = 1 AND deleted_at IS NULL`` for one
        ``kind`` (mirroring ``DataGraphService.fetch``). The one shared live
        predicate every lane builds on. ``kind`` defaults to this class's
        :attr:`KIND` so a concrete vertical reads its own lane with a bare
        ``live()``; the generic :class:`DataGraph` gateway (which spans kinds)
        passes the kind explicitly."""
        resolved = kind if kind is not None else cls.KIND
        return cls.filter("kind = ?", resolved).filter("active = 1").filter("deleted_at IS NULL")

    @classmethod
    def search(cls, query: str, k: int) -> list[Self]:
        """Up to ``k`` FTS5 candidate rows of this kind for ``query``, ranked by
        FTS rank — the per-vertical single-kind read the cross-kind recall
        service fuses across verticals later. The query is sanitised to word
        characters and each term prefix-matched; an empty query, no candidates,
        or any DB error yields ``[]``. Calling this on the generic KIND-less
        :class:`DataGraph` gateway raises loudly (a misuse), never a silent
        empty result."""
        kind = cls.KIND  # loud AttributeError on the KIND-less gateway, before
        # the try below can swallow it into a "no results" []
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


class DataGraph(DataGraphRow):
    """The thin generic ``data_graph`` gateway spanning every not-yet-ported
    kind: the structured user-context lanes (``system``/``user_specific``) and
    the exact-``(kind, key)`` upsert/reinforce/supersede write path.

    It has no single :attr:`~DataGraphRow.KIND`, so its callers pass ``kind``
    explicitly to the inherited :meth:`~DataGraphRow.live`, and it carries the
    kind-parametrised :meth:`active_by_key`; the shared live predicate, reinforce
    path, write-sync and column set come from the base."""

    # The kinds whose exact-key value change reconciles by temporal supersession
    # (``DataGraphService._KIND_POLICY`` contradiction 'cosine_supersede' /
    # 'lut_canonicalize'); with the embedding path out of scope both collapse to
    # the exact-key supersede fallback the service takes on a LUT/vec miss. Every
    # other kind (contradiction ``None``) instead inserts a fresh row.
    _SUPERSEDING_KINDS: ClassVar[frozenset[str]] = frozenset({"system", "user_specific"})
    # Retrieval-weight haircut on the demoted row (``_SUPERSEDE_RW_FACTOR``).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5

    # ── lane reads — live scopes, late-binding ──────────────────────────

    @classmethod
    def traits(cls) -> Query[Self]:
        """Live ``user_specific`` traits, ``retrieval_weight DESC`` — the
        user-summary channel's facts section (``UserSummaryConfig`` section 1)."""
        return cls.live("user_specific").order_by("retrieval_weight DESC")

    # ── write path (§3.11 C) ────────────────────────────────────────────

    @classmethod
    def active_by_key(cls, kind: str, key: str) -> Self | None:
        """The single active row for an exact ``(kind, key)`` — the store
        lookup, mirroring ``_SELECT_ACTIVE_BY_KIND_KEY_SQL`` (``active = 1``
        only; deliberately NOT ``deleted_at``-filtered). Kind-parametrised
        because the gateway spans kinds."""
        return cls.filter("kind = ?", kind).filter("key = ?", key).filter("active = 1").first()

    @classmethod
    def store(cls, kind: str, key: str, value: str, source: str | None = None) -> Self:
        """Exact-``(kind, key)`` upsert, ported from ``DataGraphService.store``.

        No active row → insert new. Same value (case-folded, trimmed) →
        :meth:`reinforce`. Contradicting value → temporal supersede for a
        superseding kind (demote old, insert successor), else a fresh insert
        (``contradiction = None`` kinds). Returns the resulting live row."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(kind, key)
        if existing is None:
            return cls(
                kind=kind, key=key, value=value, source=source,
                first_seen_at=now, last_confirmed_at=now,
            ).save()
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce()
        if kind in cls._SUPERSEDING_KINDS:
            existing.demote()
            return cls(
                kind=kind, key=key, value=value, source=source,
                first_seen_at=now, last_confirmed_at=now, valid_from=now,
            ).save()
        return cls(
            kind=kind, key=key, value=value, source=source,
            first_seen_at=now, last_confirmed_at=now,
        ).save()

    def demote(self) -> Self:
        """Close this row out on supersession (``_apply_temporal_supersession``):
        mark ``active = 0``, halve ``retrieval_weight``, and stamp ``valid_to``
        so it drops out of every live lane and enters the fast-decay window."""
        self.active = 0
        self.retrieval_weight *= self._SUPERSEDE_RW_FACTOR
        self.valid_to = utc_now().isoformat()
        return self.save()

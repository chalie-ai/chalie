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
from services.embedding_utils import pack_embedding
from services.search_expander_service import enqueue, should_fts_index
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
        write.

        Kinds excluded by
        :func:`services.search_expander_service.should_fts_index` (behavioural
        patterns, machine-written system rows) never enter the index — the same
        authority ``_self_heal`` consults, so neither enqueue path indexes them."""
        if not should_fts_index(self.kind, self.source):
            return
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

    # ── cross-kind recall primitives (E7 — ruling 1, layer 1) ────────────
    #
    # These are raw DB primitives: vec-KNN and the supersession-neighbour walk,
    # both kind-scoped and RECALL-STRICT (``valid_to IS NULL`` on top of the
    # ``live()`` predicate) so a superseded row never surfaces as a primary
    # candidate. ``superseded_predecessors`` is the one place that deliberately
    # reaches the ``valid_to IS NOT NULL`` rows this filter excludes elsewhere.
    # ``services/memory_recall_service.py`` (layer 3) is the only caller these
    # exist for; a vertical's own :meth:`gather_candidates` (layer 2) is the
    # ORM composition in between.

    @classmethod
    def vec_candidates(cls, query_embedding: list[float] | None, k: int) -> dict[int, dict[str, float]]:
        """Kind-scoped key_vec + value_vec KNN candidates (ported from
        ``_recall_vec_search``, f035ebc0:services/data_graph_service.py — the
        JOIN-then-filter idiom against a vec0 table is precedented in that same
        commit's ``_recall_variant_search``). Each vec0 table is joined back to
        ``data_graph`` and filtered to this kind under the RECALL-STRICT live
        predicate (``active = 1 AND deleted_at IS NULL AND valid_to IS NULL`` —
        stricter than :meth:`live`, which omits ``valid_to``). Returns
        ``{rowid: {'key_cos', 'value_cos'}}`` — raw per-lane cosine signals, not
        a final score."""
        blob = pack_embedding(query_embedding) if query_embedding else None
        if not blob:
            return {}
        candidates: dict[int, dict[str, float]] = {}
        conn = cls._bound_connection()
        for table, signal in (("data_graph_key_vec", "key_cos"), ("data_graph_value_vec", "value_cos")):
            try:
                cursor = conn.execute(
                    f"SELECT v.rowid, v.distance FROM {table} v "
                    "JOIN data_graph d ON d.id = v.rowid "
                    "WHERE v.embedding MATCH ? AND k = ? "
                    "AND d.kind = ? AND d.active = 1 AND d.deleted_at IS NULL AND d.valid_to IS NULL "
                    "ORDER BY v.distance",
                    (blob, k, cls.KIND),
                )
                for rowid, dist in cursor.fetchall():
                    cos = max(0.0, 1.0 - (dist ** 2 / 2.0))
                    row_sig = candidates.setdefault(rowid, {"key_cos": 0.0, "value_cos": 0.0})
                    row_sig[signal] = max(row_sig[signal], cos)
            except Exception:
                logger.debug(
                    "[DATA GRAPH] %s KNN failed (non-fatal) for kind=%s", table, cls.KIND, exc_info=True
                )
        return candidates

    @classmethod
    def variant_candidates(cls, query_embedding: list[float] | None, k: int) -> dict[int, float]:
        """Kind-scoped ``expanded_semantic_vec`` (doc2query variant) KNN
        candidates (ported from ``_recall_variant_search``,
        f035ebc0:services/data_graph_service.py): joins the variant vec table
        through ``expanded_semantic`` back to ``data_graph``, filtered to this
        kind under the RECALL-STRICT live predicate. Returns
        ``{rowid: variant_cos}``."""
        blob = pack_embedding(query_embedding) if query_embedding else None
        if not blob:
            return {}
        try:
            cursor = cls._bound_connection().execute(
                "SELECT CAST(es.related_to_id AS INTEGER) AS source_id, v.distance "
                "FROM expanded_semantic_vec v "
                "JOIN expanded_semantic es ON es.id = v.rowid "
                "JOIN data_graph d ON d.id = CAST(es.related_to_id AS INTEGER) "
                "WHERE v.embedding MATCH ? AND k = ? "
                "AND es.relates_to_table = 'data_graph' "
                "AND d.kind = ? AND d.active = 1 AND d.deleted_at IS NULL AND d.valid_to IS NULL "
                "ORDER BY v.distance",
                (blob, k, cls.KIND),
            )
            candidates: dict[int, float] = {}
            for rowid, dist in cursor.fetchall():
                cos = max(0.0, 1.0 - (dist ** 2 / 2.0))
                candidates[rowid] = max(candidates.get(rowid, 0.0), cos)
            return candidates
        except Exception:
            logger.debug(
                "[DATA GRAPH] variant vec KNN failed (non-fatal) for kind=%s", cls.KIND, exc_info=True
            )
            return {}

    @classmethod
    def superseded_predecessors(cls, key: str, limit: int = 5) -> list[Self]:
        """Historical predecessor rows for this kind's exact ``key`` — the
        supersession-neighbour walk (ruling 3) that REPLACES the deleted
        ``data_graph_edges`` walk: rows sharing ``(kind, key)`` closed out by a
        later supersession (``valid_to IS NOT NULL``), most-recently-valid
        first. A recall seed is always a LIVE row (the newest version), so its
        only supersession neighbours are its own predecessors — read backward
        off the temporal chain rather than an edge table (out of scope; the
        richer relatedness graph stays deleted)."""
        return (cls.filter("kind = ?", cls.KIND).filter("key = ?", key)
                   .filter("valid_to IS NOT NULL")
                   .order_by("valid_from DESC")
                   .limit(limit).get())

    @classmethod
    def _hydrate_candidates(cls, signals: dict[int, dict[str, float]]) -> dict[int, dict[str, object]]:
        """Fetch the full row for every candidate id in one query and pair it
        with its signals — the ORM hydration step every ``gather_candidates``
        below shares."""
        if not signals:
            return {}
        ids = list(signals.keys())
        placeholders = ",".join("?" * len(ids))
        rows = {row.id: row for row in cls.filter(f"id IN ({placeholders})", *ids).get()}
        return {rowid: {"row": rows[rowid], **sig} for rowid, sig in signals.items() if rowid in rows}

    @classmethod
    def gather_candidates(
        cls, query: str, query_embedding: list[float] | None, k: int
    ) -> dict[int, dict[str, object]]:
        """This vertical's raw per-lane recall signals — vec (key/value) +
        doc2query variant + FTS bonus — composed from the primitives above
        (ruling 1, layer 2). Shared here since six of the seven verticals
        compose the identical set of lanes;
        :class:`~models.behavioral_pattern.BehavioralPattern` overrides this to
        be vec-only (TKT-1455 — no FTS posting, no variants for that kind).
        Each entry is ``{'row': <hydrated Self>, 'key_cos', 'value_cos',
        'fts_bonus', 'variant_cos'}``; fusing these into one composite score is
        the recall service's job (ruling 2), not this layer's."""
        signals = cls.vec_candidates(query_embedding, k)
        for rowid, variant_cos in cls.variant_candidates(query_embedding, k).items():
            row_sig = signals.setdefault(rowid, {"key_cos": 0.0, "value_cos": 0.0})
            row_sig["variant_cos"] = max(row_sig.get("variant_cos", 0.0), variant_cos)
        for hit in cls.search(query, 30) if query else []:
            # search() filters active=1/deleted_at only (no valid_to) — guard
            # here so a superseded row's FTS posting can never surface as a
            # live candidate (the recall-strict filter every other lane keeps).
            if hit.valid_to is not None:
                continue
            row_sig = signals.setdefault(cast("int", hit.id), {"key_cos": 0.0, "value_cos": 0.0})
            row_sig["fts_bonus"] = 1.0
        return cls._hydrate_candidates(signals)

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

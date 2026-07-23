"""Cross-kind fused recall over the data graph (E7 — ruling 1, layer 3):
composes each in-span vertical's ORM candidate-gather, fuses the raw per-lane
signals into one composite score (ruling 2), expands top hits with their
supersession predecessors (ruling 3), optionally merges in episodic recall,
ranks, and returns the top-k.

MODELLESS: holds no SQL of its own. Every read reaches the DB exclusively
through a vertical model's ORM primitives —
:meth:`~models.data_graph.DataGraphRow.gather_candidates` and
:meth:`~models.data_graph.DataGraphRow.superseded_predecessors` — never a raw
query against ``data_graph`` or its shadow tables.

Recall span (ruling 4): ``user_specific``, ``system``, ``misc``,
``place``, ``discovery``, ``document`` — deliberately NOT ``contact`` (no
recall need identified) and deliberately NOT ``behavioral_pattern`` (surfaced
deterministically, never semantically — see the note below). ``document`` is in
``_VERTICALS`` so an explicit ``kinds=["document"]`` call (document search,
``abilities/document.py`` / ``api/documents.py``) reaches it through this
service; it is deliberately absent from every default span
(``memory_service._DG_RECALL_KINDS``, ``api/memory.py._RECALL_KINDS``),
which pass their own explicit ``kinds=`` lists that omit it — so it never
enters generic cross-kind recall uninvited.

Embedding backfill — why this service does NOT port
``_backfill_missing_embeddings`` (f035ebc0:services/data_graph_service.py):
that legacy helper synchronously embedded orphan rows at recall time using
raw SQL against ``data_graph_key_vec``/``data_graph_value_vec``/the FTS table,
from within the old monolithic ``DataGraphService``, which held its own SQL
freely. This service is deliberately modelless and SQL-less (ruling 1), so it
cannot reimplement that pattern without violating the ruling, and doing the
same work through the ORM would mean writing vec rows from a service — a
second write path alongside ``DataGraphRow._sync_search_index``'s ``enqueue``,
violating "one source of truth per concept" for search-index writes. Write-side
embedding population is owned exclusively by
``services/search_expander_service.py``'s async ``enqueue()`` pipeline.

``behavioral_pattern`` is intentionally NOT a recall vertical. Patterns reach
the model deterministically every turn — ``PromptService`` injects them via
``BehavioralPatternService.patterns()`` (user-summary channel, most-recent
first) and ``top_patterns()`` (pattern/geo channels, top by confidence). Those
reads carry the full active set (capped, not query-filtered), so a semantic
recall lane over patterns is redundant with what is already in context; and
because a pattern's value is rewritten every turn as its confidence decays,
vec-indexing it would be per-turn embedding churn for no gain.
``BehavioralPattern`` therefore declares no search config (``__search__ =
None``, the ``Searchable`` trait off), excluding the kind from the entire
write-side pipeline, and this span omits it — the two now agree instead of one
indexing gate starving a span that lists it.
"""

from __future__ import annotations

import logging
import math
from typing import cast

from contracts.tuples.recall_signals import RecallSignals
from models.data_graph import DataGraphRow
from models.discovery import DiscoveryRow
from models.document import DocumentRow
from models.fact import FactRow
from models.misc import MiscRow
from models.place import PlaceRow
from models.system_memory import SystemMemoryRow
from services.time_utils import parse_utc, utc_now

logger = logging.getLogger(__name__)

# Recall span (ruling 4) — every kind fused into cross-kind recall. `document`
# only surfaces for a caller that explicitly asks (`kinds=["document"]`); every
# default-span caller passes its own explicit `kinds=` list that omits it.
_VERTICALS: tuple[type[DataGraphRow], ...] = (
    FactRow, SystemMemoryRow, MiscRow, PlaceRow, DiscoveryRow, DocumentRow,
)

# Fusion weights (ruling 2, ported VERBATIM from the deleted
# ``_recall_score_candidates``, f035ebc0:services/data_graph_service.py).
_KEY_COS_WEIGHT = 2.0
_VALUE_COS_WEIGHT = 1.0
_FTS_BONUS_WEIGHT = 0.3
_VARIANT_COS_WEIGHT = 0.8
_ACTR_SIGMOID_WEIGHT = 0.3

# Historical-edge multiplier for a supersession predecessor (ruling 3 — the
# supersedes/superseded_by weight from the deleted ``_EDGE_TYPE_MULTIPLIER``).
_PREDECESSOR_MULTIPLIER = 1.8

# Candidate KNN width per vertical, and the top-scored window (relative to the
# caller's `limit`) whose supersession predecessors get a chance to compete
# for a final slot — bounded so expansion never means one DB round trip per
# raw candidate (cheap: `superseded_predecessors` is only called for this
# top slice, not the whole candidate pool).
_K_MULTIPLIER = 3
_MAX_K = 50
_EXPANSION_WINDOW_MULTIPLIER = 2


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _composite_score(row: DataGraphRow, signals: RecallSignals) -> float:
    """The weighted-composite fusion (ruling 2, VERBATIM): a cosine/FTS base
    scaled by ``retrieval_weight`` and an ACT-R-style recency/evidence boost."""
    base = (
        _KEY_COS_WEIGHT * signals.key_cos
        + _VALUE_COS_WEIGHT * signals.value_cos
        + _FTS_BONUS_WEIGHT * signals.fts_bonus
        + _VARIANT_COS_WEIGHT * signals.variant_cos
    )
    now_ts = utc_now().timestamp()
    ref_ts_str = row.last_accessed_at or row.last_confirmed_at
    try:
        ref_ts = parse_utc(ref_ts_str).timestamp() if ref_ts_str else now_ts - 3600.0
    except Exception:
        ref_ts = now_ts - 3600.0
    age_seconds = max(1.0, now_ts - ref_ts)
    evidence = max(1, row.evidence_count or 1)
    actr_boost = math.log(evidence) - 0.5 * math.log(age_seconds)
    retrieval_weight = row.retrieval_weight if row.retrieval_weight is not None else 1.0
    return base * retrieval_weight * (1 + _ACTR_SIGMOID_WEIGHT * _sigmoid(actr_boost))


def _cos_score(key_cos: float, value_cos: float) -> float:
    return max(key_cos, value_cos)


def _project(row: DataGraphRow, composite: float, signals: RecallSignals) -> dict[str, object]:
    """The data-graph recall hit shape (mirrors the deleted ``recall()``'s
    return dict) — the one contract both ``MemoryService._search_data_graph``
    and ``api/memory.py`` adapt from independently."""
    return {
        "id": row.id,
        "kind": row.kind,
        "key": row.key,
        "value": row.value,
        "source": row.source,
        "first_seen_at": row.first_seen_at,
        "retrieval_weight": row.retrieval_weight,
        "evidence_count": row.evidence_count,
        "composite_score": composite,
        "cos_score": _cos_score(signals.key_cos, signals.value_cos),
    }


def _expand_supersession(
    scored: list[tuple[DataGraphRow, float, RecallSignals]], limit: int
) -> list[tuple[DataGraphRow, float, RecallSignals]]:
    """Ruling 3: for each of the top-scored hits, walk its supersession
    predecessors (off ``valid_to``/``valid_from``, not an edge table) and add
    them as candidates at the historical-edge multiplier — the replacement for
    the deleted ``data_graph_edges`` graph-expansion lane. Only the top window
    is expanded (not every raw candidate) since each predecessor lookup is one
    ORM round trip (this service holds no SQL to batch them)."""
    window = scored[: max(limit, 1) * _EXPANSION_WINDOW_MULTIPLIER]
    seen_ids = {row.id for row, _composite, _sig in scored}
    expanded = list(scored)
    for row, composite, _sig in window:
        for predecessor in type(row).superseded_predecessors(row.key, limit=3):
            if predecessor.id in seen_ids:
                continue
            expanded.append((predecessor, composite * _PREDECESSOR_MULTIPLIER, RecallSignals()))
            seen_ids.add(predecessor.id)
    expanded.sort(key=lambda item: item[1], reverse=True)
    return expanded


def _episode_candidates(query: str, query_embedding: list[float] | None, limit: int) -> list[dict[str, object]]:
    """Episode hits (:meth:`~services.episodic_service.EpisodicService.retrieve`)
    projected into the same recall-dict shape, for a caller that opts into
    ``include_episodes=True``."""
    from services.episodic_service import EpisodicService  # noqa: PLC0415
    from models.episode import Episode  # noqa: PLC0415

    # retrieve() returns `list[Episode] | tuple[list[Episode], dict]`, gated by
    # `return_telemetry` (omitted here, so it's always the bare list) — cast
    # narrows the static type to match the guaranteed runtime shape.
    episodes = cast(
        "list[Episode]", EpisodicService().retrieve(query, query_embedding=query_embedding, k=limit)
    )
    return [
        {
            "id": ep.id, "kind": "episode", "key": None, "value": ep.gist,
            "source": None, "first_seen_at": ep.created_at,
            "retrieval_weight": ep.retrieval_weight, "evidence_count": None,
            "composite_score": ep.composite_score, "cos_score": None,
        }
        for ep in episodes
    ]


def recall(
    query: str,
    *,
    kinds: list[str] | None = None,
    limit: int = 10,
    query_embedding: list[float] | None = None,
    include_episodes: bool = False,
) -> list[dict[str, object]]:
    """Cross-kind fused recall (ruling 1, layer 3): compose the span's
    vertical gathers, fuse per-lane signals into one composite score
    (ruling 2), expand the top hits with their supersession predecessors
    (ruling 3), optionally merge in :func:`_episode_candidates`, rank, and
    return the top ``limit``.

    ``kinds`` narrows the span (``None`` = every in-span kind). Never raises:
    any failure degrades to ``[]`` (mirrors the legacy ``recall()``'s
    swallow-and-log, since a recall miss must never break the caller's turn).

    ``include_episodes`` defaults False: both current callers
    (``MemoryService._search_data_graph``, ``api/memory.py``) already run
    their own separate episode lane at the call site — merging episodes here
    too would double-count them. Pass ``include_episodes=True`` only from a
    caller that wants ONE unified data-graph-plus-episode list."""
    try:
        if query_embedding is None and query:
            from services.embedding_service import get_embedding_service  # noqa: PLC0415

            query_embedding = get_embedding_service().generate_embedding(query)

        verticals = [v for v in _VERTICALS if kinds is None or v.KIND in kinds]
        k = min(limit * _K_MULTIPLIER, _MAX_K)

        scored: list[tuple[DataGraphRow, float, RecallSignals]] = []
        for vertical in verticals:
            for row, signals in vertical.gather_candidates(query, query_embedding, k).values():
                scored.append((row, _composite_score(row, signals), signals))
        scored.sort(key=lambda item: item[1], reverse=True)

        expanded = _expand_supersession(scored, limit)
        results = [_project(row, composite, signals) for row, composite, signals in expanded[:limit]]

        if include_episodes:
            results = (results + _episode_candidates(query, query_embedding, limit))[:limit]
        return results
    except Exception:
        logger.error("[MEMORY RECALL] recall failed for query=%r", query, exc_info=True)
        return []

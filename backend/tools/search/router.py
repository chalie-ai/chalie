"""
Search query router — semantic routing via pre-built example embeddings.

Embeds the query, runs k-NN against example_embeddings (vec0), scores
providers by top-3 mean similarity, returns winners above threshold.

Regenerate embeddings with:  cd backend && python -m utils.generate_search_cache
"""

import logging
import sqlite3
from typing import cast

import numpy as np

from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_DB = str(FileMapperService.get_search_providers_db_path())

_GAP = 0.10        # max distance from top score to still be selected
_MAX = 3            # max providers returned
_K = 30             # k-NN depth
_TOP_N = 3          # top-N examples per provider for mean score
_MIN_SCORE = 0.50   # below this = routing miss
_WEAK_SCORE = 0.60  # below this (but >= _MIN_SCORE) = DDG supplement fires


def rank_results(query: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Score each result by cosine similarity to *query* and return best-first.

    Each result's score is the cosine similarity in [0, 1] between the query
    embedding and the result's ``(title + " " + summary)`` embedding.  This is
    a per-HIT relevance signal — distinct from the per-provider routing score
    used by :func:`route_query` — so the global top-5 cut in
    ``SearchAbility.run`` selects the 5 most query-relevant hits across all
    providers after merging their result sets.

    Fail-open: any embedding error is logged at WARNING level and the original
    *results* list is returned unchanged with no ``score`` key added.  Search
    is never aborted by a ranking failure.

    Args:
        query: The user's plain-language search query.
        results: List of result dicts with at least ``title`` and ``summary``.

    Returns:
        A new list sorted descending by ``score`` (each result gains a
        ``score: float`` key), or the original *results* unchanged on error.
    """
    if not results:
        return results

    try:
        from services.embedding_service import get_embedding_service
        svc = get_embedding_service()

        query_vec: np.ndarray = svc.generate_embedding_np(query)

        texts = [
            (cast("str", r.get("title")) or "") + " " + (cast("str", r.get("summary")) or "")
            for r in results
        ]
        result_vecs: list[np.ndarray] = svc.generate_embeddings_batch(texts)

        # Cosine similarity: dot(q, r) / (‖q‖ · ‖r‖), clamped to [0, 1].
        q_norm = np.linalg.norm(query_vec)
        scored: list[dict[str, object]] = []
        for r, vec in zip(results, result_vecs):
            r_norm = np.linalg.norm(vec)
            if q_norm == 0.0 or r_norm == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(query_vec, vec) / (q_norm * r_norm))
            sim = max(0.0, min(1.0, sim))
            scored.append({**r, "score": sim})

        return sorted(scored, key=lambda x: cast("float", x["score"]), reverse=True)

    except Exception as exc:
        logger.warning("[SEARCH] rank_results embedding error, returning unranked: %s", exc)
        return results


def route_query(query: str) -> list[dict[str, object]]:
    """Route query to best providers. Returns [{"name", "score"}, ...] or []."""
    try:
        from services.embedding_service import get_embedding_service
        vec = get_embedding_service().generate_embedding(query)
    except Exception as e:
        logger.warning('[SEARCH] embedding failed: %s', e)
        return []

    try:
        conn = sqlite3.connect(_DB)
        try:
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                conn.load_extension('vec0')

            # k-NN search
            hits = conn.execute(
                "SELECT rowid, distance FROM example_embeddings "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (pack_embedding(vec), _K),
            ).fetchall()

            # rowid → provider name (single query, same connection)
            id_to_provider = dict(conn.execute(
                "SELECT pe.id, p.name FROM provider_examples pe "
                "JOIN providers p ON p.id = pe.provider_id"
            ).fetchall())
        finally:
            conn.close()
    except Exception as e:
        logger.warning('[SEARCH] routing query failed: %s', e)
        return []

    if not hits:
        return []

    # Group similarities by provider
    scores: dict[str, list[float]] = {}
    for rowid, distance in hits:
        name = id_to_provider.get(rowid)
        if name:
            scores.setdefault(name, []).append(max(0.0, 1.0 - distance / 2.0))

    if not scores:
        return []

    # Top-N mean per provider, ranked
    ranked: list[dict[str, object]] = sorted(
        [{'name': n, 'score': round(sum(sorted(s, reverse=True)[:_TOP_N]) / min(len(s), _TOP_N), 4)}
         for n, s in scores.items()],
        key=lambda p: cast("float", p['score']), reverse=True,
    )

    top = cast("float", ranked[0]['score'])
    if top < _MIN_SCORE:
        logger.info('[SEARCH] routing_miss query="%s" top=%.3f', query, top)
        return []

    selected = [p for p in ranked if cast("float", p['score']) >= top - _GAP][:_MAX]
    logger.info('[SEARCH] routed "%s" → %s', query, [(p['name'], p['score']) for p in selected])
    return selected

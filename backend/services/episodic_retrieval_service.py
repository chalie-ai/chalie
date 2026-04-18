"""
EpisodicRetrievalService — hybrid FTS + vector retrieval with apex traversal.

Retrieval pipeline:
  1. FTS lane   — literal/keyword match against the gist column.
  2. Vector lane — cosine KNN via sqlite-vec.
  3. Union + dedup by episode ID.
  4. Apex promotion — each hit walks the consolidated_into chain to the apex
     episode.  Duplicate apexes are deduplicated so a super-episode only
     appears once regardless of how many leaves matched.
  5. Composite rerank — vector-sim + FTS-rank + emotional_congruence +
     arousal_salience + recency + salience.  Entity/goal/outcome components
     are intentionally absent (columns dropped in episodic simplification).

Plan: /Volumes/llm/chalie-plans/v0.3.3/episodic-simplification.md § Retrieval
"""

import logging
import math
import re
import struct
from typing import Optional

from services.episodic_constants import APEX_TRAVERSAL_MAX_DEPTH
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# ── Module-level constant ─────────────────────────────────────────────────────

_MAX_TRAVERSAL_DEPTH = APEX_TRAVERSAL_MAX_DEPTH


# ── Apex traversal ────────────────────────────────────────────────────────────


def walk_up_to_apex(episode_id: str) -> Optional[dict]:
    """Walk the consolidated_into chain from *episode_id* to its apex.

    Returns the apex episode dict.  If the episode has no consolidated_into,
    it is its own apex and is returned directly.

    Cycle-safe: tracks visited IDs in a ``seen`` set and logs an error if a
    cycle is detected, returning the current episode rather than looping.

    Depth-safe: stops after _MAX_TRAVERSAL_DEPTH hops and logs a warning,
    returning whatever episode is current at that point.

    Args:
        episode_id: UUID string of the starting episode.

    Returns:
        Episode dict at the apex (or the current episode on cycle/depth-guard),
        or None if the starting episode does not exist.
    """
    from services.episodic_service import EpisodicService
    from services.database_service import get_shared_db_service

    def _fetch(eid: str) -> Optional[dict]:
        try:
            db = get_shared_db_service()
            episodic_svc = EpisodicService(db)
            return episodic_svc.get_episode_by_id(eid)
        except Exception as exc:
            logger.warning(f"[RETRIEVAL] walk_up_to_apex fetch failed for id={eid}: {exc}")
            return None

    seen: set[str] = set()
    current_id = episode_id

    for _ in range(_MAX_TRAVERSAL_DEPTH):
        if current_id in seen:
            logger.error(f"[retrieval] consolidation cycle at id={current_id}")
            return _fetch(current_id)
        seen.add(current_id)

        row = _fetch(current_id)
        if not row:
            return None
        if not row.get('consolidated_into'):
            return row
        current_id = str(row['consolidated_into'])

    logger.warning(f"[retrieval] apex traversal hit max depth {_MAX_TRAVERSAL_DEPTH}")
    return _fetch(current_id)


# ── Retrieval helpers ─────────────────────────────────────────────────────────


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list of floats into a sqlite-vec binary blob."""
    if embedding is None:
        return None
    try:
        from services.embedding_utils import pack_embedding
        return pack_embedding(embedding)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _pack_embedding failed: {exc}")
        return None


def _unpack_blob(blob: bytes) -> list[float]:
    """Unpack a sqlite-vec binary blob into a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def _cosine_sim(embedding_a, blob_b: bytes) -> float:
    """Cosine similarity between a float-list embedding and a packed blob."""
    try:
        import numpy as np
        vec_a = np.array(embedding_a, dtype=np.float32)
        vec_b = np.array(_unpack_blob(blob_b), dtype=np.float32)
        if vec_a.shape != vec_b.shape or vec_a.shape[0] == 0:
            return 0.0
        norm_a = float(np.linalg.norm(vec_a))
        if norm_a > 0:
            vec_a = vec_a / norm_a
        return float(np.dot(vec_a, vec_b))
    except Exception:
        return 0.0


def _fts_search(query_text: str, channel: str, k: int) -> list[dict]:
    """Full-text search against episodes_fts (gist column only).

    Returns episode dicts with a ``text_rank`` key (negative float; lower =
    better match in FTS5 rank semantics).

    Args:
        query_text: Free-text query string.
        channel:    Episode channel filter.
        k:          Maximum number of results to return.

    Returns:
        List of episode dicts (may be empty on failure or no matches).
    """
    from services.database_service import get_shared_db_service

    # Sanitise for FTS5 — only alphanumeric + spaces, then quote each term.
    safe = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text)
    safe = re.sub(r'\s+', ' ', safe).strip()
    terms = ' '.join(f'"{w}"' for w in safe.split() if w)
    if not terms:
        return []

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.id, e.gist, e.salience, e.channel, e.created_at,
                       e.last_accessed_at, e.retrieval_weight,
                       e.emotional_valence, e.emotional_arousal,
                       e.consolidated_into,
                       episodes_fts.rank AS text_rank
                FROM episodes_fts
                JOIN episodes e ON e.rowid = episodes_fts.rowid
                WHERE episodes_fts MATCH ?
                  AND e.channel = ?
                  AND e.deleted_at IS NULL
                ORDER BY episodes_fts.rank
                LIMIT ?
                """,
                (terms, channel, k),
            )
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                'id': str(r[0]),
                'gist': r[1],
                'salience': r[2],
                'channel': r[3],
                'created_at': r[4],
                'last_accessed_at': r[5],
                'retrieval_weight': r[6] if r[6] is not None else 1.0,
                'emotional_valence': r[7],
                'emotional_arousal': r[8],
                'consolidated_into': r[9],
                'text_rank': r[10],
                'vector_distance': None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] FTS search failed: {exc}")
        return []


def _vector_search(query_embedding, channel: str, k: int) -> list[dict]:
    """Cosine KNN search via sqlite-vec.

    Args:
        query_embedding: Query embedding as a list of floats.
        channel:         Episode channel filter.
        k:               Maximum number of results to return.

    Returns:
        List of episode dicts with a ``vector_distance`` key.
    """
    from services.database_service import get_shared_db_service

    blob = _pack_embedding(query_embedding)
    if blob is None:
        return []

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.id, e.gist, e.salience, e.channel, e.created_at,
                       e.last_accessed_at, e.retrieval_weight,
                       e.emotional_valence, e.emotional_arousal,
                       e.consolidated_into,
                       v.distance AS vector_distance
                FROM episodes e
                JOIN episodes_vec v ON v.rowid = e.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND e.channel = ?
                  AND e.deleted_at IS NULL
                ORDER BY v.distance
                """,
                (blob, k, channel),
            )
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                'id': str(r[0]),
                'gist': r[1],
                'salience': r[2],
                'channel': r[3],
                'created_at': r[4],
                'last_accessed_at': r[5],
                'retrieval_weight': r[6] if r[6] is not None else 1.0,
                'emotional_valence': r[7],
                'emotional_arousal': r[8],
                'consolidated_into': r[9],
                'text_rank': None,
                'vector_distance': r[10],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] Vector search failed: {exc}")
        return []


def _dedup_by_id(hits: list[dict]) -> list[dict]:
    """Deduplicate hits preserving first-seen order.  Merges text_rank and
    vector_distance from duplicate entries onto the first occurrence."""
    seen: dict[str, dict] = {}
    ordered: list[dict] = []
    for hit in hits:
        eid = hit['id']
        if eid not in seen:
            seen[eid] = dict(hit)
            ordered.append(seen[eid])
        else:
            # Merge: fill in the lane the first copy was missing.
            if seen[eid]['text_rank'] is None and hit.get('text_rank') is not None:
                seen[eid]['text_rank'] = hit['text_rank']
            if seen[eid]['vector_distance'] is None and hit.get('vector_distance') is not None:
                seen[eid]['vector_distance'] = hit['vector_distance']
    return ordered


def _rerank_composite(query_embedding, episodes: list[dict]) -> list[dict]:
    """Composite rerank — simplified per plan.

    Score = vector_sim + fts_rank_norm + emotional_congruence + arousal_salience
            + recency + salience_norm

    No entity_overlap, goal_tag_overlap, or outcome_relevance (columns gone).
    No super-episode boost — apex traversal already promoted supers.

    Args:
        query_embedding: The query embedding (list of floats) or None.
        episodes:        List of apex episode dicts (may already be deduped).

    Returns:
        Episodes sorted by composite score (highest first), with
        ``composite_score`` key added.
    """
    now = utc_now()

    for ep in episodes:
        # 1. Vector similarity — convert distance to [0, 1] similarity.
        vd = ep.get('vector_distance')
        if vd is not None and query_embedding is not None:
            # sqlite-vec cosine distance is 1 - cosine_sim, clamped to [0, 2].
            vector_sim = max(0.0, 1.0 - float(vd))
        else:
            vector_sim = 0.0

        # 2. FTS rank normalised — FTS5 rank is a negative float; better = lower.
        #    We invert and clamp to [0, 1] using an empirical range of [-50, 0].
        tr = ep.get('text_rank')
        if tr is not None:
            fts_rank_norm = max(0.0, min(1.0, 1.0 - abs(float(tr)) / 50.0))
        else:
            fts_rank_norm = 0.0

        # 3. Emotional congruence — no query emotion available at module level;
        #    treat as neutral (0.5) so the signal is a non-zero soft bias.
        ep_valence = ep.get('emotional_valence')
        emotional_congruence = 0.5  # neutral default

        # 4. Arousal-salience — raw arousal in [0, 1].
        ep_arousal = ep.get('emotional_arousal')
        arousal_salience = float(ep_arousal) if ep_arousal is not None else 0.0

        # 5. Recency — exponential decay, half-life ≈ 14 days.
        created_str = ep.get('created_at')
        try:
            ref_time = parse_utc(ep.get('last_accessed_at') or created_str)
            hours = (now - ref_time).total_seconds() / 3600.0
            recency = math.exp(-0.002 * hours)  # half-life ≈ 347 h ≈ 14 days
        except Exception:
            recency = 0.5

        # 6. Salience normalised to [0, 1].
        salience_norm = float(ep.get('salience') or 5) / 10.0

        composite = (
            vector_sim * 4.0
            + fts_rank_norm * 2.0
            + emotional_congruence * 1.0
            + arousal_salience * 1.0
            + recency * 1.0
            + salience_norm * 1.0
        )
        ep['composite_score'] = composite

    episodes.sort(key=lambda e: e.get('composite_score', 0.0), reverse=True)
    return episodes


# ── Public entry point ────────────────────────────────────────────────────────


def retrieve(
    query_text: str,
    query_embedding,
    channel: str,
    k: int = 10,
) -> list[dict]:
    """Hybrid FTS + vector retrieval with apex promotion and composite rerank.

    Args:
        query_text:      Raw text query (for FTS lane).
        query_embedding: Pre-computed query embedding (for vector lane).
        channel:         Episode channel to restrict search to.
        k:               Number of results to return.

    Returns:
        Up to *k* apex episode dicts, sorted by composite score.
    """
    fts_hits = _fts_search(query_text, channel, k=k * 2)
    vector_hits = _vector_search(query_embedding, channel, k=k * 2)

    union = _dedup_by_id(fts_hits + vector_hits)

    promoted: list[dict] = []
    seen_apex: set[str] = set()

    for hit in union:
        apex = walk_up_to_apex(hit['id'])
        if apex is None:
            continue
        apex_id = apex['id']
        if apex_id in seen_apex:
            # Carry over lane signals from the hit onto the already-promoted apex.
            continue
        seen_apex.add(apex_id)
        # Merge hit signals (text_rank, vector_distance) onto the apex dict so
        # the composite reranker has both lane scores even when the hit was a leaf.
        merged = dict(apex)
        if merged.get('text_rank') is None:
            merged['text_rank'] = hit.get('text_rank')
        if merged.get('vector_distance') is None:
            merged['vector_distance'] = hit.get('vector_distance')
        promoted.append(merged)

    return _rerank_composite(query_embedding, promoted)[:k]

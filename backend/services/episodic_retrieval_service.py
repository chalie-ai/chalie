"""
EpisodicRetrievalService — hybrid FTS + vector retrieval with apex traversal.

Retrieval pipeline:
  1. FTS lane   — literal/keyword match against the gist column.
  2. Vector lane — cosine KNN via sqlite-vec.
  3. Union + dedup by episode ID.
  4. Apex promotion — each hit walks the consolidated_into chain to the apex
     episode.  Duplicate apexes are deduplicated so a super-episode only
     appears once regardless of how many leaves matched.
  5. Composite rerank — vector-sim + FTS-rank + arousal_salience + recency
     + salience + retrieval_weight.  Entity/goal/outcome/emotional_congruence
     components are intentionally absent (dropped in episodic simplification).
  6. Reconsolidation bump on the returned episodes (salience + activation).

Single production entry point: ``retrieve()``.  No class — this is a module-
level function API so there is no per-call EpisodicService construction cost.

"""

import logging
import math
import re
from typing import Optional

from services.episodic_constants import APEX_TRAVERSAL_MAX_DEPTH
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# ── Module-level constants ───────────────────────────────────────────────────

_MAX_TRAVERSAL_DEPTH = APEX_TRAVERSAL_MAX_DEPTH

# Reconsolidation debounce window — skip activation bump if the episode
# was touched more recently than this.
_RECONSOLIDATION_DEBOUNCE_SECONDS = 10 * 60  # 10 minutes
_RECONSOLIDATION_SALIENCE_BOOST = 2  # 0.2 × 10 — matches legacy default


# ── Apex traversal ────────────────────────────────────────────────────────────


def _get_episode_raw(episode_id: str, db=None) -> Optional[dict]:
    """Fetch a single episode row WITHOUT bumping its activation score.

    Used for apex traversal so intermediate hops don't accidentally boost
    leaves that the caller will never see.  The final apex gets its bump
    via the normal reconsolidation pass in ``retrieve``.

    Args:
        episode_id: The episode UUID.
        db:         Optional DatabaseService; resolves via shared service if None.

    Returns:
        Episode dict or None if not found.
    """
    if db is None:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, gist, salience, channel, created_at, updated_at,
                       last_accessed_at, access_count, transcript_ids,
                       transcript_id_start, transcript_id_end,
                       emotional_valence, emotional_arousal,
                       consolidated_from, consolidated_into,
                       storage_strength, retrieval_weight
                FROM episodes
                WHERE id = ? AND deleted_at IS NULL
                """,
                (episode_id,),
            )
            row = cursor.fetchone()
            cursor.close()
        if not row:
            return None
        return {
            'id': str(row[0]),
            'gist': row[1],
            'salience': row[2],
            'channel': row[3],
            'created_at': row[4],
            'updated_at': row[5],
            'last_accessed_at': row[6],
            'access_count': row[7],
            'transcript_ids': row[8] if row[8] is not None else '[]',
            'transcript_id_start': row[9],
            'transcript_id_end': row[10],
            'emotional_valence': row[11],
            'emotional_arousal': row[12],
            'consolidated_from': row[13] if row[13] is not None else '[]',
            'consolidated_into': row[14],
            'storage_strength': row[15] if row[15] is not None else 1.0,
            'retrieval_weight': row[16] if row[16] is not None else 1.0,
        }
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _get_episode_raw failed for id={episode_id}: {exc}")
        return None


def walk_up_to_apex(episode_id: str, db=None) -> Optional[dict]:
    """Walk the consolidated_into chain from *episode_id* to its apex.

    Returns the apex episode dict.  If the episode has no consolidated_into,
    it is its own apex and is returned directly.

    Does NOT bump activation on intermediate hops — see ``_get_episode_raw``.
    The caller (``retrieve``) is responsible for activating the final apex
    it actually surfaces to the user.

    Cycle-safe: tracks visited IDs in a ``seen`` set and logs an error if a
    cycle is detected, returning the current episode rather than looping.

    Depth-safe: stops after _MAX_TRAVERSAL_DEPTH hops and logs a warning,
    returning whatever episode is current at that point.

    Args:
        episode_id: UUID string of the starting episode.
        db:         Optional DatabaseService for testing; resolves the shared
                    service when None.

    Returns:
        Episode dict at the apex (or the current episode on cycle/depth-guard),
        or None if the starting episode does not exist.
    """
    seen: set[str] = set()
    current_id = episode_id

    for _ in range(_MAX_TRAVERSAL_DEPTH):
        if current_id in seen:
            logger.exception(f"[retrieval] consolidation cycle at id={current_id}")
            return _get_episode_raw(current_id, db=db)
        seen.add(current_id)

        row = _get_episode_raw(current_id, db=db)
        if not row:
            return None
        if not row.get('consolidated_into'):
            return row
        current_id = str(row['consolidated_into'])

    logger.warning(f"[retrieval] apex traversal hit max depth {_MAX_TRAVERSAL_DEPTH}")
    return _get_episode_raw(current_id, db=db)


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


def _generate_embedding(text: str) -> Optional[list[float]]:
    """Resolve an embedding via the shared embedding service."""
    try:
        from services.embedding_service import get_embedding_service
        return get_embedding_service().generate_embedding(text)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _generate_embedding failed: {exc}")
        return None


def _count_episodes(channel: Optional[str] = None) -> int:
    """Return the live episode count, optionally scoped to a channel."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            if channel is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL AND channel = ?",
                    (channel,),
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
                )
            return cursor.fetchone()[0]
    except Exception:
        return 0


def _fts_search(query_text: str, channel: Optional[str], k: int) -> list[dict]:
    """Full-text search against episodes_fts (gist column only).

    Args:
        query_text: Free-text query string.
        channel:    Episode channel filter. ``None`` means no channel filter.
        k:          Maximum number of results.

    Returns:
        List of episode dicts with ``text_rank`` key (negative float).
    """
    from services.database_service import get_shared_db_service

    # Sanitise for FTS5 — only alphanumeric + spaces, then quote each term.
    safe = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text or '')
    safe = re.sub(r'\s+', ' ', safe).strip()
    terms = ' '.join(f'"{w}"' for w in safe.split() if w)
    if not terms:
        return []

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            if channel is not None:
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
            else:
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
                      AND e.deleted_at IS NULL
                    ORDER BY episodes_fts.rank
                    LIMIT ?
                    """,
                    (terms, k),
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


def _vector_search(
    query_embedding,
    channel: Optional[str],
    k: int,
    radius: Optional[float] = None,
) -> list[dict]:
    """Cosine KNN search via sqlite-vec.

    Args:
        query_embedding: Query embedding as a list of floats.
        channel:         Channel filter (``None`` = no filter).
        k:               Max results to pull from sqlite-vec.
        radius:          If supplied, drop hits with ``vector_distance > radius``.

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
            if channel is not None:
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
            else:
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
                      AND e.deleted_at IS NULL
                    ORDER BY v.distance
                    """,
                    (blob, k),
                )
            rows = cursor.fetchall()
            cursor.close()

        hits = [
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
        if radius is not None:
            hits = [h for h in hits if h['vector_distance'] is not None
                    and h['vector_distance'] <= radius]
        return hits
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
            if seen[eid]['text_rank'] is None and hit.get('text_rank') is not None:
                seen[eid]['text_rank'] = hit['text_rank']
            if seen[eid]['vector_distance'] is None and hit.get('vector_distance') is not None:
                seen[eid]['vector_distance'] = hit['vector_distance']
    return ordered


def _rerank_composite(episodes: list[dict]) -> list[dict]:
    """Composite rerank — simplified per plan.

    Score components (all normalised to ~[0, 1] before weighting):
      - vector_sim      (w=4)  — 1 - vector_distance
      - fts_rank_norm   (w=2)  — inverted + clamped FTS5 rank
      - arousal         (w=1)  — raw emotional_arousal
      - recency         (w=1)  — exponential decay with ~14-day half-life
      - salience        (w=1)  — salience / 10
      - retrieval       (w=3)  — retrieval_weight

    No entity_overlap / goal_tag_overlap / outcome_relevance (columns gone).
    No emotional_congruence (no query emotion available here; was hardcoded
    constant which is zero-discrimination — dropped).
    No super-episode boost — apex traversal already promoted supers.
    """
    now = utc_now()

    for ep in episodes:
        vd = ep.get('vector_distance')
        if vd is not None:
            vector_sim = max(0.0, 1.0 - float(vd))
        else:
            vector_sim = 0.0

        tr = ep.get('text_rank')
        if tr is not None:
            fts_rank_norm = max(0.0, min(1.0, 1.0 - abs(float(tr)) / 50.0))
        else:
            fts_rank_norm = 0.0

        arousal = ep.get('emotional_arousal')
        arousal_norm = float(arousal) if arousal is not None else 0.0

        created_str = ep.get('created_at')
        try:
            ref_time = parse_utc(ep.get('last_accessed_at') or created_str)
            hours = (now - ref_time).total_seconds() / 3600.0
            recency = math.exp(-0.002 * hours)  # half-life ≈ 14 days
        except Exception:
            recency = 0.5

        salience_norm = float(ep.get('salience') or 5) / 10.0
        retrieval_w = float(ep.get('retrieval_weight') or 1.0)

        composite = (
            vector_sim * 4.0
            + fts_rank_norm * 2.0
            + arousal_norm * 1.0
            + recency * 1.0
            + salience_norm * 1.0
            + retrieval_w * 3.0
        )
        # Scale into the 0-100-ish space memory_skill uses for its
        # confidence label bucketing (hits/100).
        ep['composite_score'] = composite * 10.0

    episodes.sort(key=lambda e: e.get('composite_score', 0.0), reverse=True)
    return episodes


# ── Reconsolidation ───────────────────────────────────────────────────────────


def _is_debounced(ep: dict, store, now) -> bool:
    """Return True if this episode should be skipped (debounce window active).

    When a MemoryStore connection is available the debounce key is set on the
    store; otherwise ``last_accessed_at`` is used as a fallback clock.
    Side-effect: sets the debounce key in *store* when the episode is NOT
    debounced and *store* is available.
    """
    eid = ep.get('id')
    if store is not None:
        key = f"reconsolidation:{eid}"
        if store.get(key):
            return True
        store.set(key, "1", ex=_RECONSOLIDATION_DEBOUNCE_SECONDS)
        return False

    last = ep.get('last_accessed_at')
    if not last:
        return False
    try:
        last_dt = parse_utc(last)
        return (now - last_dt).total_seconds() < _RECONSOLIDATION_DEBOUNCE_SECONDS
    except Exception:
        return False


def _write_reconsolidation(ep: dict, db, now) -> None:
    """Persist activation bump for one episode and update the dict in place."""
    new_salience = min(10, int(ep.get('salience') or 5) + _RECONSOLIDATION_SALIENCE_BOOST)
    new_access_count = int(ep.get('access_count') or 0) + 1
    iso_now = now.isoformat()

    with db.connection() as conn:
        conn.execute(
            """
            UPDATE episodes
            SET salience = ?,
                last_accessed_at = ?,
                access_count = ?,
                storage_strength = MIN(COALESCE(storage_strength, 1.0) + 0.1, 10.0),
                retrieval_weight = MIN(COALESCE(retrieval_weight, 1.0) + 0.3, 1.0),
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (new_salience, iso_now, new_access_count, ep['id']),
        )

    ep['salience'] = new_salience
    ep['last_accessed_at'] = iso_now
    ep['access_count'] = new_access_count


def _apply_reconsolidation(episodes: list[dict]) -> None:
    """Bump activation_score + salience for retrieved episodes.

    Debounced to avoid runaway boosts on rapid re-queries: an episode is
    only reconsolidated once per ``_RECONSOLIDATION_DEBOUNCE_SECONDS`` window.
    Falls back to ``last_accessed_at`` as the debounce clock.

    Mutates the episode dicts in place — ``salience`` and
    ``last_accessed_at`` reflect the new values on return.
    """
    if not episodes:
        return

    from services.database_service import get_shared_db_service

    store = None
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
    except Exception:
        store = None

    now = utc_now()

    try:
        db = get_shared_db_service()
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _apply_reconsolidation db resolve failed: {exc}")
        return

    for ep in episodes:
        eid = ep.get('id')
        if not eid:
            continue
        try:
            if _is_debounced(ep, store, now):
                continue
            _write_reconsolidation(ep, db, now)
        except Exception as exc:
            logger.warning(f"[RETRIEVAL] reconsolidation failed for id={eid}: {exc}")


def _promote_to_apex(union: list) -> list:
    """Walk each hit up to its apex and deduplicate, merging lane signals."""
    promoted: list[dict] = []
    seen_apex: set[str] = set()
    for hit in union:
        apex = walk_up_to_apex(hit['id'])
        if apex is None:
            continue
        apex_id = apex['id']
        if apex_id in seen_apex:
            continue
        seen_apex.add(apex_id)
        merged = dict(apex)
        if merged.get('text_rank') is None:
            merged['text_rank'] = hit.get('text_rank')
        if merged.get('vector_distance') is None:
            merged['vector_distance'] = hit.get('vector_distance')
        promoted.append(merged)
    return promoted


def _collect_top_distances(ranked: list) -> list:
    """Return rounded vector distances for the top-5 ranked episodes."""
    top_dists = []
    for ep in ranked[:5]:
        vd = ep.get('vector_distance')
        if vd is not None:
            try:
                top_dists.append(round(float(vd), 4))
            except (TypeError, ValueError):
                pass
    return top_dists


# ── Public entry point ────────────────────────────────────────────────────────


def retrieve(
    query_text: str,
    *,
    query_embedding=None,
    channel: Optional[str] = None,
    radius: float = 0.3,
    k: int = 10,
    return_telemetry: bool = False,
):
    """Hybrid FTS + vector retrieval with apex promotion and composite rerank.

    This is the single production retrieval entry point for episodes.  It is
    a drop-in replacement for the legacy ``EpisodicService.retrieve_episodes``
    and carries the same radius/telemetry contract.

    Adaptive radius: applies a population-aware shrink
    ``effective = radius / (1 + 0.1 × log2(N + 2))`` before hitting sqlite-vec
    so retrieval tightens as the corpus grows.

    Args:
        query_text:      Raw text query (FTS lane).
        query_embedding: Pre-computed embedding.  If None, one is generated
                         from ``query_text`` via the embedding service.
        channel:         Episode channel filter.  ``None`` = all channels.
        radius:          Input vector-distance ceiling (pre adaptive-shrink).
        k:               Maximum number of results.
        return_telemetry: If True, return ``(episodes, telemetry_dict)``.

    Returns:
        List of up to *k* apex episode dicts (highest composite_score first).
        When ``return_telemetry`` is True, returns ``(list, telemetry)``.
    """
    telemetry: dict = {
        'episode_count': 0,
        'input_radius': radius,
        'adaptive_shrink_divisor': 1.0,
        'effective_radius': radius,
        'vector_candidates': 0,
        'survivors_after_radius': 0,
        'fts_candidates': 0,
        'final_rrf_count': 0,
        'top_distances': [],
    }

    try:
        if query_embedding is None and query_text:
            query_embedding = _generate_embedding(query_text)

        episode_count = _count_episodes(channel)
        adaptive_divisor = 1.0 + 0.1 * math.log2(episode_count + 2)
        effective_radius = radius / adaptive_divisor

        telemetry['episode_count'] = episode_count
        telemetry['adaptive_shrink_divisor'] = adaptive_divisor
        telemetry['effective_radius'] = effective_radius

        # Pull 2× k candidates per lane so post-dedup/apex-promotion still has
        # room to fill k slots.
        fts_hits = _fts_search(query_text or '', channel, k=k * 2)
        all_vector_hits = _vector_search(query_embedding, channel, k=200)
        telemetry['vector_candidates'] = len(all_vector_hits)

        vector_hits = [
            h for h in all_vector_hits
            if h['vector_distance'] is not None
            and h['vector_distance'] <= effective_radius
        ]
        telemetry['survivors_after_radius'] = len(vector_hits)
        telemetry['fts_candidates'] = len(fts_hits)

        union = _dedup_by_id(fts_hits + vector_hits)
        if not union:
            if return_telemetry:
                return [], telemetry
            return []

        ranked = _rerank_composite(_promote_to_apex(union))[:k]
        _apply_reconsolidation(ranked)

        telemetry['final_rrf_count'] = len(ranked)
        telemetry['top_distances'] = _collect_top_distances(ranked)

        if return_telemetry:
            return ranked, telemetry
        return ranked

    except Exception as exc:
        logger.error(f"[RETRIEVAL] retrieve failed: {exc}")
        if return_telemetry:
            return [], telemetry
        return []

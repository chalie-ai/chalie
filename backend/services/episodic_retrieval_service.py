"""
EpisodicRetrievalService — hybrid FTS + vector retrieval over a collapsed tree.

Retrieval pipeline:
  1. FTS lane    — literal/keyword match against the gist column.
  2. Vector lane — cosine KNN via sqlite-vec (no distance ceiling; the relative
     score floor below decides what survives).
  3. Union + dedup by episode ID, merging the two lanes' signals onto one row.
  4. Collapsed-tree rerank — episodes at ALL hierarchy levels (leaf, super,
     era) compete in one candidate pool. Each lane signal is min-max normalised
     within the pool, the composite ``score = relevance + recency + importance``
     is computed, then a RELATIVE score floor drops weak candidates outright —
     results are never padded back up to *k*.

There is NO radius / adaptive-shrink apparatus: a hard vector-distance ceiling
silently muted the vector lane (the radius-0 bug). Lane quality is now governed
entirely by per-lane normalisation plus the relative floor.

Retrieval is a pure read: it never mutates the episodes it returns.  Relevance
(``last_relevant_at``, which anchors both the recency rerank term and the
decay engine) advances only on write-relevant events such as episode creation
and consolidation — never on access.

Single production entry point: ``retrieve()``.  No class — this is a module-
level function API so there is no per-call EpisodicService construction cost.

"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, cast

from services.database_service import DatabaseService
from services.episodic_constants import APEX_TRAVERSAL_MAX_DEPTH
from services.time_utils import utc_now, parse_utc

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# ── Module-level constants ───────────────────────────────────────────────────

_MAX_TRAVERSAL_DEPTH = APEX_TRAVERSAL_MAX_DEPTH

# parse_utc never raises — it returns this sentinel on unparseable/legacy input
# so a bad timestamp degrades the recency term loudly instead of crashing.
_PARSE_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

# Vector KNN depth pulled per query (collapsed-tree KNN k=50).
_VECTOR_KNN_K = 50

# Composite scores are scaled into a 0-100-ish band for the memory skill's
# confidence-label bucketing (composite_score / 100).
_COMPOSITE_DISPLAY_SCALE = 10.0

# Recency half-life: exp(-_RECENCY_DECAY_PER_HOUR × hours) ≈ 0.5 at ~14 days.
_RECENCY_DECAY_PER_HOUR = 0.002

# Recency fallback when the relevance anchor cannot be parsed (sentinel/legacy).
_RECENCY_FALLBACK = 0.5

# Salience normalisation denominator (salience is stored on a 0-10 scale).
_SALIENCE_SCALE = 10.0

# Relative score floor: a candidate must score at least this fraction of the
# top candidate's score to survive. Below it, it is DROPPED (never padded to k).
# Tuned for precision-over-recall on the turn-0 hot path.
_RELATIVE_SCORE_FLOOR = 0.5


# ── Apex traversal ────────────────────────────────────────────────────────────


def _get_episode_raw(episode_id: str, db: Optional[DatabaseService] = None) -> Optional[dict[str, object]]:
    """Fetch a single episode row for apex traversal or final-apex surfacing."""
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
                       storage_strength, retrieval_weight,
                       location_lat, location_lon, location_name,
                       last_relevant_at
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
            'location_lat': row[17],
            'location_lon': row[18],
            'location_name': row[19],
            'last_relevant_at': row[20],
        }
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _get_episode_raw failed for id={episode_id}: {exc}")
        return None


def walk_up_to_apex(episode_id: str, db: Optional[DatabaseService] = None) -> Optional[dict[str, object]]:
    """Walk the consolidated_into chain from *episode_id* to its apex."""
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


def _pack_embedding(embedding: object) -> Optional[bytes]:
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
            return cast(int, cast("sqlite3.Row", cursor.fetchone())[0])
    except Exception:
        return 0


def _fts_search(query_text: str, channel: Optional[str], k: int) -> list[dict[str, object]]:
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
                           episodes_fts.rank AS text_rank,
                           e.location_lat, e.location_lon, e.location_name
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
                           episodes_fts.rank AS text_rank,
                           e.location_lat, e.location_lon, e.location_name
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
                'location_lat': r[11],
                'location_lon': r[12],
                'location_name': r[13],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] FTS search failed: {exc}")
        return []


def _vector_search(
    query_embedding: object,
    channel: Optional[str],
    k: int,
) -> list[dict[str, object]]:
    """Cosine KNN search via sqlite-vec.

    No distance ceiling is applied here: the relative score floor in
    ``_rerank_composite`` decides what survives. This is what kills the
    radius-0 bug where a hard ceiling muted the entire vector lane.

    Args:
        query_embedding: Query embedding as a list of floats.
        channel:         Channel filter (``None`` = no filter).
        k:               Max results to pull from sqlite-vec.

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
                           v.distance AS vector_distance,
                           e.location_lat, e.location_lon, e.location_name
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
                           v.distance AS vector_distance,
                           e.location_lat, e.location_lon, e.location_name
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
                'location_lat': r[11],
                'location_lon': r[12],
                'location_name': r[13],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] Vector search failed: {exc}")
        return []


def _dedup_by_id(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate hits preserving first-seen order.  Merges text_rank and
    vector_distance from duplicate entries onto the first occurrence."""
    seen: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    for hit in hits:
        eid = cast(str, hit['id'])
        if eid not in seen:
            seen[eid] = dict(hit)
            ordered.append(seen[eid])
        else:
            if seen[eid]['text_rank'] is None and hit.get('text_rank') is not None:
                seen[eid]['text_rank'] = hit['text_rank']
            if seen[eid]['vector_distance'] is None and hit.get('vector_distance') is not None:
                seen[eid]['vector_distance'] = hit['vector_distance']
    return ordered


def _vector_sim(ep: dict[str, object]) -> float:
    """Raw vector-lane similarity (1 - distance), 0 when the lane did not hit."""
    vd = ep.get('vector_distance')
    if vd is None:
        return 0.0
    return max(0.0, 1.0 - float(cast(float, vd)))


def _fts_strength(ep: dict[str, object]) -> float:
    """Raw FTS-lane strength from the (negative) FTS5 rank, 0 when no hit.

    FTS5 ``rank`` is negative and more negative = better; the magnitude is the
    raw strength fed into per-lane min-max normalisation (no fixed /50 clamp —
    normalisation is now relative to the candidate pool).
    """
    tr = ep.get('text_rank')
    if tr is None:
        return 0.0
    return abs(float(cast(float, tr)))


def _recency(ep: dict[str, object], now: datetime) -> float:
    """Exponential recency term anchored on ``last_relevant_at`` (half-life 14d).

    The clock is the relevance anchor (last write-relevant event), falling back
    to creation time — reads never advance it. ``parse_utc`` never raises; a
    sentinel/legacy value yields a flat fallback rather than a crash.
    """
    anchor = ep.get('last_relevant_at') or ep.get('created_at')
    if not anchor:
        return _RECENCY_FALLBACK
    ref_time = parse_utc(cast(str, anchor))
    if ref_time == _PARSE_SENTINEL:
        logger.warning(
            "[RETRIEVAL] unparseable relevance anchor for id=%s: %r",
            ep.get('id'), anchor,
        )
        return _RECENCY_FALLBACK
    hours = (now - ref_time).total_seconds() / 3600.0
    return math.exp(-_RECENCY_DECAY_PER_HOUR * hours)


def _importance(ep: dict[str, object]) -> float:
    """Importance term: salience/10 × retrieval_weight."""
    salience_norm = float(cast(float, ep.get('salience') or 5)) / _SALIENCE_SCALE
    retrieval_w = float(cast(float, ep.get('retrieval_weight') or 1.0))
    return salience_norm * retrieval_w


def _min_max_normalise(values: list[float]) -> list[float]:
    """Min-max normalise a lane's raw signals into [0, 1].

    A degenerate spread (all equal, including all-zero) maps every entry to 0.0
    so a lane that fired uniformly adds no discrimination rather than inflating
    every candidate to 1.0.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def _rerank_composite(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapsed-tree composite rerank with a relative score floor.

    Episodes at all hierarchy levels (leaf, super, era) compete in one pool.
    The two retrieval lanes are min-max normalised WITHIN the pool, then

        score = relevance + recency + importance

    where ``relevance`` is the stronger of the two normalised lane signals,
    ``recency`` is an exp half-life ≈ 14d term on ``last_relevant_at``, and
    ``importance`` is ``salience/10 × retrieval_weight``.

    A RELATIVE floor then drops every candidate scoring below
    ``_RELATIVE_SCORE_FLOOR × top_score`` — survivors only, never padded to *k*.
    Returns the survivors sorted by score descending.
    """
    if not episodes:
        return []

    now = utc_now()

    vector_norm = _min_max_normalise([_vector_sim(ep) for ep in episodes])
    fts_norm = _min_max_normalise([_fts_strength(ep) for ep in episodes])

    for ep, v_norm, f_norm in zip(episodes, vector_norm, fts_norm):
        relevance = max(v_norm, f_norm)
        score = relevance + _recency(ep, now) + _importance(ep)
        # Scale into the 0-100-ish space the memory skill buckets confidence
        # labels in (composite_score / 100).
        ep['composite_score'] = score * _COMPOSITE_DISPLAY_SCALE

    episodes.sort(key=lambda e: cast(float, e.get('composite_score', 0.0)), reverse=True)

    top_score = cast(float, episodes[0].get('composite_score', 0.0))
    if top_score <= 0.0:
        return episodes
    floor = top_score * _RELATIVE_SCORE_FLOOR
    return [ep for ep in episodes if cast(float, ep.get('composite_score', 0.0)) >= floor]


def _promote_to_apex(union: list[dict[str, object]]) -> list[dict[str, object]]:
    """Hydrate each matched hit into a full episode row for the collapsed tree.

    Collapsed-tree retrieval (RAPTOR collapsed-traversal): episodes
    at ALL hierarchy levels — leaves, super-episodes, era digests — compete in
    one candidate pool. A matched row is therefore surfaced AS-IS at its own
    level; it is no longer walked up to its apex (which discarded leaves and
    muted the lower levels). The full row is fetched so the rerank has
    ``last_relevant_at``/``salience``/``retrieval_weight`` available, with the
    matching hit's lane signals (text_rank / vector_distance) merged on.

    The apex-walk utility (``walk_up_to_apex``) is retained for the hierarchy
    consolidation path; it is intentionally not used here.
    """
    promoted: list[dict[str, object]] = []
    seen: set[str] = set()
    for hit in union:
        eid = cast(str, hit['id'])
        if eid in seen:
            continue
        row = _get_episode_raw(eid)
        if row is None:
            continue
        seen.add(eid)
        merged = dict(row)
        merged['text_rank'] = hit.get('text_rank')
        merged['vector_distance'] = hit.get('vector_distance')
        promoted.append(merged)
    return promoted


def _collect_top_distances(ranked: list[dict[str, object]]) -> list[float]:
    """Return rounded vector distances for the top-5 ranked episodes."""
    top_dists = []
    for ep in ranked[:5]:
        vd = ep.get('vector_distance')
        if vd is not None:
            try:
                top_dists.append(round(float(cast(float, vd)), 4))
            except (TypeError, ValueError):
                pass
    return top_dists


# ── Public entry point ────────────────────────────────────────────────────────


def retrieve(
    query_text: str,
    *,
    query_embedding: Optional[list[float]] = None,
    channel: Optional[str] = None,
    k: int = 10,
    return_telemetry: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, object]]:
    """Hybrid FTS + vector retrieval over the collapsed episode tree.

    Single production retrieval entry point for episodes. Both lanes pull their
    full candidate sets (no vector-distance ceiling); episodes at all levels
    (leaf / super / era) compete in one pool, are scored by the normalised
    composite rerank, and the weak tail is dropped by the relative score floor
    (it is never padded back up to *k*).

    Args:
        query_text:      Raw text query (FTS lane).
        query_embedding: Pre-computed embedding.  If None, one is generated
                         from ``query_text`` via the embedding service.
        channel:         Episode channel filter.  ``None`` = all channels.
        k:               Maximum number of results returned.
        return_telemetry: If True, return ``(episodes, telemetry_dict)``.

    Returns:
        Up to *k* episode dicts (highest composite_score first) that survived
        the relative floor.  When ``return_telemetry`` is True, returns
        ``(list, telemetry)``.
    """
    telemetry: dict[str, object] = {
        'episode_count': 0,
        'vector_candidates': 0,
        'fts_candidates': 0,
        'floor_cut_count': 0,
        'final_rrf_count': 0,
        'top_distances': [],
    }

    try:
        if query_embedding is None and query_text:
            query_embedding = _generate_embedding(query_text)

        telemetry['episode_count'] = _count_episodes(channel)

        fts_hits = _fts_search(query_text or '', channel, k=k * 2)
        vector_hits = _vector_search(query_embedding, channel, k=_VECTOR_KNN_K)
        telemetry['vector_candidates'] = len(vector_hits)
        telemetry['fts_candidates'] = len(fts_hits)

        union = _dedup_by_id(fts_hits + vector_hits)
        if not union:
            if return_telemetry:
                return [], telemetry
            return []

        promoted = _promote_to_apex(union)
        scored = _rerank_composite(promoted)
        ranked = scored[:k]
        # Candidates dropped by the relative score floor: scored candidates that
        # did not survive the floor (before the final top-k truncation).
        telemetry['floor_cut_count'] = len(promoted) - len(scored)
        # Retrieval is a pure read: relevance advances only on write-relevant
        # events (episode creation / consolidation), never on access, so no
        # weight or salience bump is applied to the returned episodes here.

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

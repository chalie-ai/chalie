import hashlib
import json
import logging
import math
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"


# ── Dynamic memory radius — tuning constants ────────────────────────────────
#
# Composition: effective_input = BASELINE × narrow_factor × expand_factor
# `EpisodicService.retrieve_episodes` then applies its own population-aware
# adaptive shrink on top. All eight constants are tuned by the meta-harness
# loop (loop_improve.sh) against the d1-context-recall benchmark suite —
# see /Volumes/llm/chalie-plans/v0.3.2/memory-dynamic-radius.md.
#
# Do NOT read these values from config/env at import time. They are literal
# module-level floats so the meta-harness can diff-patch them mechanically.
# ─────────────────────────────────────────────────────────────────────────────

RECALL_RADIUS_BASELINE: float = 0.5
SEED_RADIUS_BASELINE: float = 0.2

NARROW_MIN_DIST: float = 0.25
NARROW_MAX_DIST: float = 0.05
NARROW_FACTOR_FLOOR: float = 0.35

EXPAND_MIN_DIST: float = 0.30
EXPAND_MAX_DIST: float = 0.55
EXPAND_FACTOR_CEILING: float = 2.2

TOOL_SCHEMA = {
    "name": "memory",
    "description": (
        "Store or recall knowledge about the user. Store personal facts "
        "the moment they're disclosed. Recall before recommending anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "recall"],
                "description": "store: save a fact. recall: search memory.",
            },
            "kind": {
                "type": "string",
                "enum": ["user_specific", "system", "misc"],
                "description": (
                    "user_specific: about the human (traits, preferences, "
                    "relationships, secrets, goals). system: about how Chalie "
                    "operates (rules, decisions, analysis). misc: short-lived "
                    "scratchpad."
                ),
            },
            "key": {
                "type": "string",
                "description": "For store: short identifier (e.g. 'user_name', 'favourite_food').",
            },
            "value": {
                "type": "string",
                "description": "For store: the fact itself.",
            },
            "query": {
                "type": "string",
                "description": "For recall: what to search for.",
            },
        },
        "required": ["action"],
    },
}


# ── Entry point ─────────────────────────────────────────────────────


def handle_memory(channel: str, params: dict) -> str:
    action = params.get("action", "recall")

    try:
        if action == "store":
            return _handle_store(channel, params)
        elif action == "recall":
            return _handle_recall(channel, params)
        else:
            return (
                f"{LOG_PREFIX} Unknown action: {action}. "
                f"Valid: store, recall"
            )
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Error in {action}: {e}", exc_info=True)
        return f"{LOG_PREFIX} Error: {e}"


# ── Store ────────────────────────────────────────────────────────────


def _handle_store(channel: str, params: dict) -> str:
    key = params.get("key")
    value = params.get("value")
    kind = params.get("kind", "user_specific")

    if not key:
        return f"{LOG_PREFIX} Error: 'key' is required for store."
    if value is None:
        return f"{LOG_PREFIX} Error: 'value' is required for store."

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.store(kind=kind, key=key, value=str(value), source=f"skill:memory:store:{channel}")

    if result is None:
        return f"{LOG_PREFIX} Store failed — invalid kind '{kind}' or internal error."

    if result.get("conflict"):
        classification = result.get("classification", "ambiguous")
        existing = result.get("existing", {})
        old_value = existing.get("value", "")
        proposed_value = result.get("proposed_value", value)
        proposed_key = result.get("proposed_key", key)

        if classification == "true_contradiction":
            return (
                f"{LOG_PREFIX} Conflict detected: existing '{proposed_key}' says "
                f"'{old_value}' but new claim is '{proposed_value}'. Which is correct?"
            )
        return (
            f"{LOG_PREFIX} Not sure if this conflicts with existing '{proposed_key}': "
            f"'{old_value}'. Should I store '{proposed_value}'?"
        )

    return f"{LOG_PREFIX} Stored '{key}'."


# ── Recall ───────────────────────────────────────────────────────────


def _handle_recall(channel: str, params: dict) -> str:
    query = params.get("query", "")
    if not query:
        return f"{LOG_PREFIX} Error: no query specified for recall."

    limit = 10

    results: List[Dict] = []
    layer_status: Dict[str, str] = {}

    hits, status = _search_data_graph(query, limit)
    layer_status["knowledge"] = status
    results.extend(hits)

    hits, status = _search_episodes(channel, query, limit)
    layer_status["episodes"] = status
    results.extend(hits)

    hits, status = _search_transcript(channel, query, limit)
    layer_status["transcript"] = status
    results.extend(hits)

    partial = sum(1 for r in results if r.get("confidence", 0) < 0.5)
    _store_fok_signal(channel, partial)

    if not results:
        return _format_empty(["knowledge", "episodes", "transcript"], layer_status, query)

    return _format_results(results, query)


def _salience_label(retrieval_weight: float) -> str:
    if retrieval_weight >= 0.7:
        return "high"
    if retrieval_weight >= 0.4:
        return "medium"
    return "low"


def _search_data_graph(query: str, limit: int) -> Tuple[List[Dict], str]:
    try:
        from services.data_graph_service import get_data_graph_service

        dgs = get_data_graph_service()
        rows = dgs.recall(query=query, limit=limit)

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            key = row.get("key", "")
            value = row.get("value", "")
            rw = row.get("retrieval_weight", 1.0)
            salience = _salience_label(rw)
            hits.append({
                "layer": "knowledge",
                "content": f"**{key}** (salience: {salience}): {value}"[:250],
                "confidence": rw,
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Data graph search failed: {e}")
        return [], f"error: {e}"


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Return cosine DISTANCE (0..2). Returns 1.0 on dimension mismatch or empty."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 1.0
    sim = max(-1.0, min(1.0, dot / math.sqrt(na * nb)))
    return 1.0 - sim


def _compute_narrow_factor(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float]:
    if not history:
        return 1.0, float("inf")

    min_dist = float("inf")
    for entry in history:
        emb = entry.get("embedding")
        if not emb:
            continue
        d = _cosine_distance(q_embedding, emb)
        if d < min_dist:
            min_dist = d

    if min_dist == float("inf") or min_dist >= NARROW_MIN_DIST:
        return 1.0, min_dist

    if min_dist <= NARROW_MAX_DIST:
        return NARROW_FACTOR_FLOOR, min_dist

    span = NARROW_MIN_DIST - NARROW_MAX_DIST
    if span <= 0:
        return NARROW_FACTOR_FLOOR, min_dist
    t = (NARROW_MIN_DIST - min_dist) / span
    factor = 1.0 - t * (1.0 - NARROW_FACTOR_FLOOR)
    return max(NARROW_FACTOR_FLOOR, min(1.0, factor)), min_dist


def _compute_expand_factor(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float]:
    if not history:
        return 1.0, 0.0

    max_drift = 0.0
    for entry in history:
        emb = entry.get("embedding")
        if not emb:
            continue
        d = _cosine_distance(q_embedding, emb)
        if d > max_drift:
            max_drift = d

    if max_drift <= EXPAND_MIN_DIST:
        return 1.0, max_drift

    if max_drift >= EXPAND_MAX_DIST:
        return EXPAND_FACTOR_CEILING, max_drift

    span = EXPAND_MAX_DIST - EXPAND_MIN_DIST
    if span <= 0:
        return EXPAND_FACTOR_CEILING, max_drift
    t = (max_drift - EXPAND_MIN_DIST) / span
    factor = 1.0 + t * (EXPAND_FACTOR_CEILING - 1.0)
    return min(EXPAND_FACTOR_CEILING, max(1.0, factor)), max_drift


def _embedding_hash(embedding: List[float]) -> str:
    if not embedding:
        return "empty"
    try:
        h = hashlib.md5()
        for x in embedding[:16]:
            h.update(f"{x:.6f}".encode())
        return h.hexdigest()[:16]
    except Exception:
        return "err"


def _write_recall_telemetry(
    db_service,
    *,
    turn_uid: str,
    transcript_id,
    channel: str,
    caller: str,
    query: str,
    embedding_hash: str,
    input_radius: float,
    narrow_factor: float,
    expand_factor: float,
    telemetry: Dict,
) -> None:
    try:
        with db_service.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_recall_log (
                    turn_uid, transcript_id, channel, caller, query,
                    query_embedding_hash, input_radius, narrow_factor,
                    expand_factor, adaptive_shrink_divisor, effective_radius,
                    episode_count, vector_candidates, fts_candidates,
                    survivors_after_radius, final_rrf_count, top_distances
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_uid,
                    transcript_id,
                    channel,
                    caller,
                    query,
                    embedding_hash,
                    input_radius,
                    narrow_factor,
                    expand_factor,
                    telemetry.get("adaptive_shrink_divisor", 1.0),
                    telemetry.get("effective_radius", input_radius),
                    telemetry.get("episode_count", 0),
                    telemetry.get("vector_candidates", 0),
                    telemetry.get("fts_candidates", 0),
                    telemetry.get("survivors_after_radius", 0),
                    telemetry.get("final_rrf_count", 0),
                    json.dumps(telemetry.get("top_distances", [])),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to write memory_recall_log row: {e}")


def recall_episodes(
    channel: str,
    query: str,
    *,
    caller: str = "llm_recall",
    baseline_radius: float | None = None,
    limit: int = 10,
    return_raw: bool = False,
):
    """Public entry point for episode recall with dynamic radius.

    Used by both the `memory` skill's recall action (``caller='llm_recall'``)
    and the pre-turn seed path (``caller='seed'``).
    """
    try:
        from services.episodic_service import EpisodicService
        from services.database_service import get_shared_db_service
        from services.message_processor import current_processor
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Episode recall imports failed: {exc}")
        return [], f"error: {exc}"

    if baseline_radius is None:
        baseline_radius = (
            SEED_RADIUS_BASELINE if caller == "seed" else RECALL_RADIUS_BASELINE
        )

    try:
        db = get_shared_db_service()
        service = EpisodicService(db)

        q_embedding = service._generate_embedding(query)

        proc = current_processor()
        history: List[Dict] = []
        turn_uid = "ephemeral"
        transcript_id = None
        if proc is not None:
            history = list(proc._memory_query_history or [])
            if proc._memory_seed and not any(
                h.get("caller") == "seed" for h in history
            ):
                try:
                    seed_emb = service._generate_embedding(proc._memory_seed)
                    history.insert(
                        0,
                        {
                            "query": proc._memory_seed,
                            "embedding": seed_emb,
                            "caller": "seed",
                            "effective_radius": SEED_RADIUS_BASELINE,
                        },
                    )
                except Exception as _seed_exc:
                    logger.debug(
                        f"{LOG_PREFIX} Could not embed seed for drift calc: {_seed_exc}"
                    )
            turn_uid = getattr(proc, "_uid", None) or proc.CHANNEL or "unbound"
            turn_uid = str(turn_uid)
            transcript_id = getattr(proc, "_uid", None)

        narrow_factor, _min_dist = _compute_narrow_factor(q_embedding, history)
        expand_factor, _max_drift = _compute_expand_factor(q_embedding, history)
        input_radius = baseline_radius * narrow_factor * expand_factor

        episodes, telemetry = service.retrieve_episodes(
            query_text=query,
            radius=input_radius,
            query_embedding=q_embedding,
            return_telemetry=True,
        )

        if proc is not None:
            proc._memory_query_history.append(
                {
                    "query": query,
                    "embedding": q_embedding,
                    "caller": caller,
                    "effective_radius": telemetry.get("effective_radius", input_radius),
                }
            )

        _write_recall_telemetry(
            db,
            turn_uid=turn_uid,
            transcript_id=transcript_id,
            channel=channel,
            caller=caller,
            query=query,
            embedding_hash=_embedding_hash(q_embedding),
            input_radius=input_radius,
            narrow_factor=narrow_factor,
            expand_factor=expand_factor,
            telemetry=telemetry,
        )

        if not episodes:
            candidates = _count_episode_candidates(db, channel)
            status = f"0 matches ({candidates} candidates evaluated)"
            return ([], status) if return_raw else ([], status)

        if return_raw:
            return episodes[:limit], f"{len(episodes[:limit])} matches"

        hits = []
        for ep in episodes[:limit]:
            gist = ep.get("gist", "")
            hits.append({
                "layer": "episodes",
                "content": gist[:200],
                "confidence": min(1.0, ep.get("composite_score", 0) / 100.0),
                "freshness": str(ep.get("created_at", "")),
                "salience": ep.get("salience", 0),
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Episode search failed: {e}", exc_info=True)
        return [], f"error: {e}"


def _search_episodes(
    channel: str, query: str, limit: int
) -> Tuple[List[Dict], str]:
    return recall_episodes(
        channel=channel,
        query=query,
        caller="llm_recall",
        limit=limit,
    )


def _search_transcript(
    channel: str, query: str, limit: int
) -> Tuple[List[Dict], str]:
    try:
        from services import transcript_service

        results = transcript_service.search(channel, query, limit=limit)

        if not results:
            return [], f"0 matches (scope: {channel})"

        hits = []
        for r in results:
            content = r.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            role = r.get("role", "unknown")
            tool_tag = f" [{r['tool_name']}]" if r.get("tool_name") else ""
            entry_channel = r.get("channel", channel)

            hits.append({
                "layer": "transcript",
                "content": f"[{role}{tool_tag}] {content}",
                "confidence": r.get("similarity", 0.5),
                "freshness": str(r.get("created_at", "")),
                "channel": entry_channel,
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Transcript search failed: {e}")
        return [], f"error: {e}"


def _count_episode_candidates(db_service, channel: str) -> int:
    try:
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL AND channel = ?",
                (channel,),
            )
            count = cursor.fetchone()[0]
            cursor.close()
        return count
    except Exception:
        return 0


# ── Formatting helpers ───────────────────────────────────────────────


def _format_results(results: List[Dict], query: str) -> str:
    by_layer: Dict[str, List[Dict]] = {}
    for r in results:
        layer = r["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(r)

    lines = [f"{LOG_PREFIX} {len(results)} results for '{query}':"]
    for layer, hits in by_layer.items():
        lines.append(f"\n  [{layer}]")
        for hit in hits:
            conf = hit.get("confidence", 0)
            content = hit["content"]
            extra = ""
            if "salience" in hit:
                extra = f", salience={hit['salience']}"
            if "channel" in hit:
                extra += f", channel={hit['channel']}"
            lines.append(f"    - {content} (confidence={conf:.2f}{extra})")

    return "\n".join(lines)


def _format_empty(
    searched: List[str], layer_status: Dict[str, str], query: str
) -> str:
    lines = [f"{LOG_PREFIX} No matches found for '{query}' across {searched}:"]
    for layer in searched:
        status = layer_status.get(layer, "not searched")
        lines.append(f"  - {layer}: {status}")
    lines.append(
        "Suggestion: Try broader query terms or use associate "
        "to explore related concepts."
    )
    return "\n".join(lines)


def _store_fok_signal(channel: str, partial_match_count: int) -> None:
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{channel}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")

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

# Baseline input radius for LLM-driven recall calls (memory skill recall).
# Higher than SEED_RADIUS_BASELINE because the LLM is deliberately probing
# for context, not pre-seeding from a compressed prior query.
RECALL_RADIUS_BASELINE: float = 0.5

# Baseline for the pre-turn seed recall. Tighter than RECALL_RADIUS_BASELINE
# because the seed query is a raw user turn and we want precision, not
# exploration.
SEED_RADIUS_BASELINE: float = 0.3

# Redundancy-narrow — fires when a new recall query is semantically very
# close to a query already made earlier in the same turn (seed or prior
# llm_recall). Narrowing contracts scope to avoid re-surfacing duplicates.
#
# NARROW_MIN_DIST  — if min_dist_to_history >= this, no narrowing (factor=1.0)
# NARROW_MAX_DIST  — if min_dist_to_history <= this, full narrowing applies
#                     (anything below is clamped to NARROW_FACTOR_FLOOR)
# NARROW_FACTOR_FLOOR — smallest multiplicative factor allowed (e.g. 0.35
#                     means recall radius shrinks to at most 35% of baseline)
NARROW_MIN_DIST: float = 0.25
NARROW_MAX_DIST: float = 0.05
NARROW_FACTOR_FLOOR: float = 0.35

# Drift-expand — fires when a new recall query is semantically FAR from the
# seed query and the prior recall query. Models exploratory multi-topic
# turns where one cluster was already covered and the LLM is now pivoting
# to an unrelated topic.
#
# DRIFT_MIN_DIST — below this, no expansion (factor=1.0)
# DRIFT_MAX_DIST — at/above this, full expansion applies (clamped to CEILING)
# EXPAND_FACTOR_CEILING — largest multiplicative factor allowed
EXPAND_MIN_DIST: float = 0.30
EXPAND_MAX_DIST: float = 0.55
EXPAND_FACTOR_CEILING: float = 2.2

TOOL_SCHEMA = {
    "name": "memory",
    "description": (
        "Store, recall or update knowledge. Use this for any memory "
        "operation — facts, preferences, traits, concepts, procedures, metrics, "
        "or arbitrary data. Never recommend anything before reading their "
        "preferences & constraints."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "recall", "update"],
                "description": (
                    "store: Save new knowledge. recall: Search memory. "
                    "update: Modify existing entry."
                ),
            },
            "entries": {
                "type": "array",
                "description": "For store action: list of entries to store.",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "Unique identifier (e.g., 'user_name', "
                                "'python:list_comprehension', 'meeting_time')."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": "The knowledge to store.",
                        },
                        "kind": {
                            "type": "string",
                            "enum": [
                                "trait", "concept", "fact", "procedure",
                                "preference", "relationship", "rule", "metric",
                            ],
                            "description": "Type of knowledge (auto-classified if omitted).",
                        },
                        "decay_class": {
                            "type": "string",
                            "enum": [
                                "permanent", "slow", "standard", "fast", "ephemeral",
                            ],
                            "description": "How quickly this decays (auto-classified if omitted).",
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional structured data (optional).",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            "query": {
                "type": "string",
                "description": "For recall action: what to search for.",
            },
            "key": {
                "type": "string",
                "description": "For update: the key to modify.",
            },
            "value": {
                "type": "string",
                "description": "For update: new value.",
            },
            "confidence": {
                "type": "number",
                "description": "For update: new confidence (0-1).",
            },
            "entity": {
                "type": "string",
                "description": "Who this is about (default: 'user').",
            },
        },
        "required": ["action"],
    },
}


# ── Kind auto-classification rules ──────────────────────────────────

_TRAIT_KEYS = {"name", "birthday", "age", "email", "phone", "address"}
_PREFERENCE_KEYS = {"prefer", "like", "dislike", "favorite"}
_PROCEDURE_KEYS = {"how_to", "steps", "process", "workflow"}


def _auto_classify(key: str) -> Tuple[str, str]:
    """Return (kind, decay_class) inferred from key name."""
    key_lower = key.lower()
    for fragment in _TRAIT_KEYS:
        if fragment in key_lower:
            return "trait", "permanent"
    for fragment in _PREFERENCE_KEYS:
        if fragment in key_lower:
            return "preference", "slow"
    for fragment in _PROCEDURE_KEYS:
        if fragment in key_lower:
            return "procedure", "slow"
    return "fact", "standard"


# ── Entry point ─────────────────────────────────────────────────────


def handle_memory(channel: str, params: dict) -> str:
    action = params.get("action", "recall")

    try:
        if action == "store":
            return _handle_store(channel, params)
        elif action == "recall":
            return _handle_recall(channel, params)
        elif action == "update":
            return _handle_update(channel, params)
        else:
            return (
                f"{LOG_PREFIX} Unknown action: {action}. "
                f"Valid: store, recall, update"
            )
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Error in {action}: {e}", exc_info=True)
        return f"{LOG_PREFIX} Error: {e}"


# ── Store ────────────────────────────────────────────────────────────


def _check_trait_contradiction(ks, new_id: int, key: str, value: str, confidence: float, thread_id: str, source: str = 'chat'):
    """
    Runs synchronously after a trait is stored.
    - temporal_change + source=chat → hard-delete old trait
    - true_contradiction OR source=ambient → reduce confidences, create pending record, push question
    """
    try:
        from services.embedding_service import get_embedding_service
        from services.contradiction_classifier_service import ContradictionClassifierService
        from services.pending_contradiction_service import PendingContradictionService
        from services.database_service import get_shared_db_service
        from services.output_service import OutputService

        emb_svc = get_embedding_service()
        new_text = f"{key}: {value}"
        embedding = emb_svc.generate_embedding(new_text)
        if embedding is None:
            return

        similar = ks.find_similar_traits(embedding, exclude_id=new_id)
        if not similar:
            return

        existing = similar[0]  # top candidate only
        existing_text = f"{existing['key']}: {existing['value']}"

        classifier = ContradictionClassifierService()
        result = classifier.check_new_trait(new_text, existing_text, source=source)
        if result is None:
            return

        classification = result.get('classification', 'compatible')

        if classification == 'temporal_change' and source == 'chat':
            # Auto-overwrite: hard-delete old trait
            ks.hard_delete_by_id(existing['id'])
            logger.info(f"[CONTRADICTION] temporal_change: deleted old trait id={existing['id']} key={existing['key']}")

        elif classification in ('true_contradiction', 'ambiguous') or source == 'ambient':
            # Ask user: reduce confidences, create pending record, push message
            ks.update_confidence(existing['id'], existing['confidence'] * 0.75)
            ks.update_confidence(new_id, 0.5)

            question = (
                f"I'm seeing two conflicting things about {key}: "
                f"'{existing['value']}' (existing) and '{value}' (just mentioned). "
                f"Which is correct? Just tell me and I'll update."
            )

            db = get_shared_db_service()
            pending_svc = PendingContradictionService(db)
            pending_svc.create(new_id, existing['id'], question, source)

            if thread_id:
                OutputService().enqueue_proactive(
                    topic=thread_id,
                    response=question,
                    source='contradiction',
                )
                logger.info(f"[CONTRADICTION] Surfaced to user: {question[:80]}")

    except Exception as e:
        logger.debug(f"[CONTRADICTION] Check failed: {e}")


def _handle_store(channel: str, params: dict) -> str:
    """Store one or more knowledge entries."""
    entries = params.get("entries", [])
    if not entries:
        return f"{LOG_PREFIX} Error: no entries specified for store."

    entity = params.get("entity", "user")
    stored = 0

    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        for entry in entries:
            key = entry.get("key")
            value = entry.get("value")
            if not key or value is None:
                continue

            kind = entry.get("kind")
            decay_class = entry.get("decay_class")

            # Auto-classify if not provided
            if not kind or not decay_class:
                auto_kind, auto_decay = _auto_classify(key)
                kind = kind or auto_kind
                decay_class = decay_class or auto_decay

            data = entry.get("data")

            stored_entry = ks.store(
                entity=entity,
                key=key,
                value=str(value),
                kind=kind,
                decay_class=decay_class,
                data=data,
                source=f"skill:memory:store:{channel}",
            )
            stored += 1

            if stored_entry and kind == 'trait':
                try:
                    new_id = stored_entry.get('rowid') or stored_entry.get('id')
                    if new_id:
                        _check_trait_contradiction(
                            ks, new_id, key, str(value), 1.0,
                            thread_id=channel,
                            source='chat',
                        )
                except Exception as _ctc_exc:
                    logger.debug(f"{LOG_PREFIX} Contradiction check failed: {_ctc_exc}")

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Store failed: {e}", exc_info=True)
        if stored:
            return f"{LOG_PREFIX} Partially stored {stored}/{len(entries)} entries (error: {e})."
        return f"{LOG_PREFIX} Store failed: {e}"

    if stored:
        keys = [e.get("key", "?") for e in entries[:5]]
        key_list = ", ".join(keys)
        suffix = f" (+{len(entries) - 5} more)" if len(entries) > 5 else ""
        return f"{LOG_PREFIX} Stored {stored} entries: {key_list}{suffix}."
    return f"{LOG_PREFIX} Nothing stored — check entry format."


# ── Recall ───────────────────────────────────────────────────────────


def _handle_recall(channel: str, params: dict) -> str:
    query = params.get("query", "")
    if not query:
        return f"{LOG_PREFIX} Error: no query specified for recall."

    limit = 10
    entity = params.get("entity")

    results: List[Dict] = []
    layer_status: Dict[str, str] = {}

    hits, status = _search_knowledge(entity, query, limit)
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


def _search_knowledge(
    entity: str, query: str, limit: int
) -> Tuple[List[Dict], str]:
    """Search the knowledge store via KnowledgeService."""
    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        rows = ks.recall(entity=entity, query=query, limit=limit)

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            kind = row.get("kind", "")
            key = row.get("key", "")
            value = row.get("value", "")
            conf = row.get("confidence", 0.5)
            conf_label = (
                "well established" if conf >= 0.7
                else "likely" if conf >= 0.4
                else "uncertain"
            )
            hits.append({
                "layer": "knowledge",
                "content": f"[{kind}] {key}: {value}"[:250],
                "confidence": conf,
                "freshness": conf_label,
                "meta": {
                    "kind": kind,
                    "entity": row.get("entity", "user"),
                    "decay_class": row.get("decay_class", "standard"),
                    "evidence_count": row.get("evidence_count", 1),
                    "confidence_label": conf_label,
                },
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Knowledge search failed: {e}")
        return [], f"error: {e}"


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Return cosine DISTANCE in the same scale sqlite-vec uses (0..2).

    Matches the distance domain used by ``EpisodicService`` radius filters
    so narrow/expand thresholds are comparable with the values that come
    back from ``top_distances`` in retrieval telemetry. Returns 1.0 (a
    neutral "unknown, definitely not matched") when vectors are empty or
    dimension-mismatched — never raises, never returns NaN.
    """
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
    sim = dot / math.sqrt(na * nb)
    # Clamp for numerical safety, then map [-1,1] similarity → [0,2] distance.
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def _compute_narrow_factor(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float]:
    """Return ``(narrow_factor, min_dist_to_history)``.

    Fires when the new query is semantically close to any prior query
    made earlier in the current turn. Closer match → tighter narrowing.
    History entries without an embedding are skipped (can't score).
    """
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

    # Linear interpolation between (NARROW_MIN_DIST → 1.0) and
    # (NARROW_MAX_DIST → NARROW_FACTOR_FLOOR). NARROW_MAX_DIST < NARROW_MIN_DIST
    # because smaller distance = stronger narrowing.
    span = NARROW_MIN_DIST - NARROW_MAX_DIST
    if span <= 0:
        return NARROW_FACTOR_FLOOR, min_dist
    t = (NARROW_MIN_DIST - min_dist) / span  # 0 at MIN_DIST, 1 at MAX_DIST
    factor = 1.0 - t * (1.0 - NARROW_FACTOR_FLOOR)
    return max(NARROW_FACTOR_FLOOR, min(1.0, factor)), min_dist


def _compute_expand_factor(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float]:
    """Return ``(expand_factor, max_drift)``.

    Drift = max distance between q_now and any prior query in history
    (including the seed). Large drift ⇒ LLM has pivoted to a fresh topic
    that the baseline radius may be too tight for.
    """
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
    """Short stable fingerprint for telemetry — not a crypto hash."""
    if not embedding:
        return "empty"
    try:
        h = hashlib.md5()
        # Only hash first 16 components; good enough for dedup in a
        # telemetry table, cheap to compute.
        for x in embedding[:16]:
            h.update(f"{x:.6f}".encode())
        return h.hexdigest()[:16]
    except Exception:
        return "err"


def _write_recall_telemetry(
    db_service,
    *,
    turn_uid: str,
    transcript_id: int | None,
    channel: str,
    caller: str,
    query: str,
    embedding_hash: str,
    input_radius: float,
    narrow_factor: float,
    expand_factor: float,
    telemetry: Dict,
) -> None:
    """Insert one row into memory_recall_log. Never raises."""
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
    and the pre-turn seed path (``caller='seed'``). Composes the effective
    input radius from baseline × narrow_factor × expand_factor using the
    current MessageProcessor's ``_memory_query_history`` and ``_memory_seed``
    if available, caches q_now in the history for future calls within the
    same turn, and emits one ``memory_recall_log`` row per call.

    Parameters
    ----------
    return_raw
        When False (default), returns ``(hits, status)`` in the memory-skill
        hit format — used by the `memory` skill recall action.
        When True, returns ``(raw_episodes, status)`` where ``raw_episodes``
        are the untouched ranked dicts from ``EpisodicService`` — used by
        the seed path which needs them for ``format_for_prompt``.
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

        # 1. Compute q_now embedding once, reuse for retrieval + history.
        q_embedding = service._generate_embedding(query)

        # 2. Reach into the current processor for per-turn memory state.
        proc = current_processor()
        history: List[Dict] = []
        turn_uid = "ephemeral"
        transcript_id: int | None = None
        if proc is not None:
            history = list(proc._memory_query_history or [])
            # Seed query is conceptually the 0th entry; fold it in for
            # narrow/expand scoring even if the seed path did not already
            # populate history.
            if proc._memory_seed and not any(
                h.get("caller") == "seed" for h in history
            ):
                # We don't have the seed embedding cached in that case,
                # so we recompute it — cheap and one-time per turn.
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

        # 3. Compose dynamic radius.
        narrow_factor, _min_dist = _compute_narrow_factor(q_embedding, history)
        expand_factor, _max_drift = _compute_expand_factor(q_embedding, history)
        input_radius = baseline_radius * narrow_factor * expand_factor

        # 4. Retrieve with pre-computed embedding + telemetry.
        episodes, telemetry = service.retrieve_episodes(
            query_text=query,
            radius=input_radius,
            query_embedding=q_embedding,
            return_telemetry=True,
        )

        # 5. Append this call to per-turn history (in live order).
        if proc is not None:
            proc._memory_query_history.append(
                {
                    "query": query,
                    "embedding": q_embedding,
                    "caller": caller,
                    "effective_radius": telemetry.get("effective_radius", input_radius),
                }
            )

        # 6. Emit telemetry row — every call, even empty.
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
    """Search episodic memory via hybrid retrieval with dynamic radius."""
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
    """Count total episodes to distinguish empty from no-match."""
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


# ── Update ───────────────────────────────────────────────────────────


def _handle_update(channel: str, params: dict) -> str:
    """Update an existing knowledge entry."""
    entity = params.get("entity", "user")
    key = params.get("key")
    if not key:
        return f"{LOG_PREFIX} Error: 'key' is required for update."

    new_value = params.get("value")
    new_confidence = params.get("confidence")

    if new_value is None and new_confidence is None:
        return f"{LOG_PREFIX} Error: provide 'value' and/or 'confidence' to update."

    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        updated = ks.update(
            entity=entity,
            key=key,
            value=new_value,
            confidence=new_confidence,
        )

        if updated:
            parts = []
            if new_value is not None:
                parts.append(f"value='{new_value[:60]}'")
            if new_confidence is not None:
                parts.append(f"confidence={new_confidence}")
            return f"{LOG_PREFIX} Updated '{key}': {', '.join(parts)}."
        return f"{LOG_PREFIX} No entry found for key='{key}', entity='{entity}'."

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Update failed: {e}", exc_info=True)
        return f"{LOG_PREFIX} Update failed: {e}"


# ── Formatting helpers ───────────────────────────────────────────────


def _format_results(results: List[Dict], query: str) -> str:
    """Format retrieval results with layer headers."""
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
            if "meta" in hit:
                m = hit["meta"]
                extra = f", certainty={m.get('confidence_label', '')}"
                if m.get("evidence_count", 1) > 1:
                    extra += f", evidence={m['evidence_count']}"
            if "channel" in hit:
                extra += f", channel={hit['channel']}"
            lines.append(f"    - {content} (confidence={conf:.2f}{extra})")

    return "\n".join(lines)


def _format_empty(
    searched: List[str], layer_status: Dict[str, str], query: str
) -> str:
    """Format structured empty results."""
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
    """Store partial match count in MemoryStore for introspect's FOK signal."""
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{channel}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")

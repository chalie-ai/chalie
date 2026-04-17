import hashlib
import json
import logging
import math
from typing import Dict, List, Optional, Tuple

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
SEED_RADIUS_BASELINE: float = 0.4

NARROW_MIN_DIST: float = 0.25
NARROW_MAX_DIST: float = 0.05
NARROW_FACTOR_FLOOR: float = 0.35

EXPAND_MIN_DIST: float = 0.30
EXPAND_MAX_DIST: float = 0.55
EXPAND_FACTOR_CEILING: float = 2.2

TOOL_SCHEMA = {
    "name": "memory",
    "description": (
        "Store, recall, or forget knowledge about the user. "
        "STORE EVERY personal fact the user discloses — no judgment, no "
        "filtering. Quirky habits, streaks, hobbies, niche preferences, "
        "throwaway details: store them ALL. The user told you because it "
        "matters to them. Recall before recommending anything. "
        "Use forget to remove a specific memory when asked.\n\n"
        "STORE RULES — read carefully:\n"
        "1. One fact per call. Never summarize multiple facts into one value. "
        "If the user mentions pasta AND pizza, call store twice.\n"
        "2. Use the canonical key from the list below when the fact fits. "
        "Don't invent variants like 'favorite_food', 'current_role', "
        "'home_city' — pick the canonical key.\n"
        "3. Atomic values only. 'Valletta' not 'lives in Valletta, Malta'. "
        "'CTO' not 'recently promoted to CTO at Acme'.\n"
        "4. When the user corrects a fact, store the new value under the SAME "
        "canonical key — the system will supersede the old one automatically.\n"
        "5. Store on FIRST mention. Don't wait for a second confirmation. "
        "If the user says 'My favorite food is pasta', store it now — don't "
        "wait to see if they repeat it.\n\n"
        "CANONICAL KEYS (kind=user_specific):\n"
        "  Immutable (set once, never change): birth_date, birth_place, "
        "biological_parents\n"
        "  Temporal (latest replaces previous): residence, gender, religion, "
        "timezone, partner, employment, financial_status\n"
        "  Coexist (multiple values accumulate): name, heritage, family, "
        "relationships, education, health, food_and_drink, entertainment, "
        "sports, style, skills_and_interests, contact_info, tech_setup, "
        "personality, life_events, goals_and_projects, routines_and_habits\n\n"
        "NICHE FACTS WITHOUT A CANONICAL KEY: still store them. Pick the "
        "shortest descriptive snake_case key (e.g. 'dryer_streak' for 'my "
        "dryer streak is 47 loads', 'coffee_order' for 'I always get oat "
        "milk lattes'). The system records the miss for later LUT curation. "
        "Never skip a fact just because no canonical key fits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "recall", "reflect", "forget"],
                "description": (
                    "store: save a fact. recall: search memory - fast. "
                    "reflect: deep search about a topic. "
                    "forget: permanently remove a memory."
                ),
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
                "description": (
                    "For store/forget: the canonical key from the list in the "
                    "tool description (e.g. 'residence', 'food_and_drink', "
                    "'employment'). Use the exact canonical key when the fact "
                    "fits one of the 27 concepts."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "For store: the fact itself, atomic — a single value. "
                    "'Valletta' (not 'lives in Valletta'). 'pasta' (not "
                    "'loves pasta and pizza'). "
                    "For forget: required when removing one value from a "
                    "multi-value key."
                ),
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
        elif action == "reflect":
            return _handle_reflect(channel, params)
        elif action == "forget":
            return _handle_forget(channel, params)
        else:
            return (
                f"{LOG_PREFIX} Unknown action: {action}. "
                f"Valid: store, recall, reflect, forget"
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

    return _format_store_response(result)


def _format_store_response(result: dict) -> str:
    status = result.get("status", "")
    canonical = result.get("canonical_key", "")
    provided = result.get("provided_key", "")
    value = result.get("value", "")
    date = result.get("date")

    if canonical != provided:
        key_display = f"'{canonical}' (canonical of '{provided}')"
    else:
        key_display = f"'{canonical}'"

    if status == "created":
        rule = result.get("rule")
        if rule == "coexist":
            return f"{key_display} saved as ['{value}']."
        return f"{key_display} saved as '{value}'."

    if status == "reinforced":
        return f"{key_display} was already set on {date}. Memory reinforced."

    if status == "superseded":
        old = result.get("old_value", "")
        return f"{key_display} updated to '{value}'. Supersedes '{old}' (previously set on {date})."

    if status == "conflict":
        old = result.get("old_value", "")
        return (
            f"{key_display} is immutable. Existing value '{old}' (set {date}) kept. "
            f"New value '{value}' rejected. Use 'forget' first if you're sure."
        )

    if status == "appended":
        all_vals = result.get("all_values") or []
        vals_str = ", ".join(f"'{v}'" for v in all_vals)
        return f"{key_display} updated. Values now: [{vals_str}] (previously updated on {date})."

    if status == "lut_miss_created":
        return f"'{provided}' saved as '{value}'."

    if status == "lut_miss_reinforced":
        return f"'{provided}' already set to '{value}'. Memory reinforced."

    if status == "lut_miss_appended":
        all_vals = result.get("all_values") or []
        vals_str = ", ".join(f"'{v}'" for v in all_vals)
        return f"'{provided}' updated. Values now: [{vals_str}]."

    return f"'{provided}' stored."


# ── Forget ───────────────────────────────────────────────────────────


def _handle_forget(channel: str, params: dict) -> str:
    key = params.get("key")
    value = params.get("value")
    kind = params.get("kind", "user_specific")

    if not key:
        return f"{LOG_PREFIX} Error: 'key' is required for forget."

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.forget(kind=kind, key=key, value=value, source=f"skill:memory:forget:{channel}")

    if result is None:
        return f"{LOG_PREFIX} Forget failed — invalid kind or internal error."

    return _format_forget_response(result)


def _format_forget_response(result: dict) -> str:
    status = result.get("status", "")
    canonical = result.get("canonical_key", "")
    provided = result.get("provided_key", "")
    value = result.get("value")
    date = result.get("date")

    if canonical and provided and canonical != provided:
        key_display = f"'{canonical}' (canonical of '{provided}')"
    else:
        key_display = f"'{canonical or provided}'"

    if status == "forgotten":
        rule = result.get("rule")
        if rule == "coexist":
            remaining = result.get("remaining_values") or []
            vals_str = ", ".join(f"'{v}'" for v in remaining)
            return f"'{value}' removed from {key_display}. Remaining: [{vals_str}]."
        old = result.get("old_value") or value
        return f"{key_display} forgotten (was '{old}', set {date})."

    if status == "forgotten_all":
        n = result.get("versions_removed", 0)
        return f"{key_display} forgotten. All {n} versions removed."

    if status == "forgotten_empty":
        return f"'{value}' removed from {key_display}. No values remain — key fully forgotten."

    if status == "value_not_found":
        remaining = result.get("remaining_values") or []
        vals_str = ", ".join(f"'{v}'" for v in remaining)
        return f"'{value}' not found in {key_display}. Currently stored: [{vals_str}]."

    if status == "not_found":
        return f"No memory stored under {key_display}. Nothing to forget."

    if status == "error":
        return f"{LOG_PREFIX} {result.get('message', 'Unknown error')}"

    return f"{key_display} forget operation completed."


# ── Recall ───────────────────────────────────────────────────────────


def _handle_recall(channel: str, params: dict) -> str:
    query = params.get("query", "")
    if not query:
        return f"{LOG_PREFIX} Error: no query specified for recall."

    limit = 10

    results: List[Dict] = []

    hits, _ = _search_data_graph(query, limit)
    results.extend(hits)

    hits, _ = _search_episodes(channel, query, limit)
    results.extend(hits)

    doc_hits = _search_document_artifacts(query, limit=3)
    results.extend(doc_hits)

    partial = sum(1 for r in results if r.get("confidence", 0) < 0.5)
    _store_fok_signal(channel, partial)

    if not results:
        return f'No memories found for the query "{query}"'

    return _format_results(results)


# ── Reflect ──────────────────────────────────────────────────────────


def _handle_reflect(channel: str, params: dict) -> str:
    query = params.get("query", "")
    if not query:
        return f"{LOG_PREFIX} Error: no query specified for reflect."

    # 1. Top episode (raw dict with transcript_ids / consolidated_from)
    raw_episodes, _ = recall_episodes(
        channel=channel,
        query=query,
        caller="llm_recall",
        limit=1,
        return_raw=True,
    )

    if not raw_episodes:
        # Fall back to data_graph only
        dg_hits, _ = _search_data_graph(query, 2)
        if not dg_hits:
            return f'No memories found for the query "{query}"'
        return _format_reflect(query, None, [], dg_hits)

    top = raw_episodes[0]

    # 2. Expand up to 3 layers deep
    supporting = _expand_episode_layers(top)

    # 3. Data graph (limit 2)
    dg_hits, _ = _search_data_graph(query, 2)

    return _format_reflect(query, top, supporting, dg_hits)


def _expand_episode_layers(episode: Dict) -> List[Dict]:
    """Recursively expand an episode graph up to 3 levels deep.

    Walks consolidated_from (child episodes) recursively. Transcript
    entries are leaf nodes. Depth 0 is the top episode itself (not
    emitted here — it's the Main Memory), so children start at depth 1.
    """
    from services.database_service import get_shared_db_service

    try:
        db = get_shared_db_service()
        results: List[Dict] = []
        _expand_recursive(db, episode, results, depth=0, max_depth=3)
        return results
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Reflect layer expansion failed: {exc}")
        return []


def _expand_recursive(
    db, episode: Dict, results: List[Dict], depth: int, max_depth: int
) -> None:
    """Recursive walker for episode graph expansion."""
    if depth >= max_depth:
        return

    consolidated_from = _parse_json_list(episode.get("consolidated_from"))
    transcript_ids = _parse_json_list(episode.get("transcript_ids"))

    # Leaf: episode has transcript entries — emit them and stop
    if transcript_ids and not consolidated_from:
        results.extend(_fetch_transcript_entries(db, transcript_ids))
        return

    # Branch: episode consolidated from child episodes — recurse
    if consolidated_from:
        children = _fetch_episodes_by_ids(db, consolidated_from)
        for child in children:
            results.append({
                "type": "episode",
                "content": child.get("gist", ""),
                "salience": child.get("salience", 0),
            })
            _expand_recursive(db, child, results, depth + 1, max_depth)


def _parse_json_list(raw) -> list:
    """Safely parse a JSON list from a string or return as-is if already a list."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _fetch_episodes_by_ids(db_service, episode_ids: list) -> List[Dict]:
    """Fetch episode rows by ID, ordered by salience descending."""
    if not episode_ids:
        return []
    try:
        with db_service.connection() as conn:
            placeholders = ','.join('?' for _ in episode_ids)
            cursor = conn.execute(
                f"SELECT id, gist, salience, transcript_ids, consolidated_from "
                f"FROM episodes WHERE id IN ({placeholders}) AND deleted_at IS NULL "
                f"ORDER BY salience DESC",
                list(episode_ids),
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Episode fetch by IDs failed: {exc}")
        return []


def _fetch_transcript_entries(db_service, transcript_ids: List) -> List[Dict]:
    """Fetch transcript rows by ID, return as supporting layer entries."""
    if not transcript_ids:
        return []
    try:
        with db_service.connection() as conn:
            placeholders = ','.join('?' for _ in transcript_ids)
            cursor = conn.execute(
                f"SELECT id, role, content, tool_name, created_at FROM transcript "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                list(transcript_ids),
            )
            rows = cursor.fetchall()

        entries = []
        for r in rows:
            content = r[2] or ""
            if len(content) > 300:
                content = content[:300] + "..."
            entries.append({
                "type": "transcript",
                "content": content,
                "salience": None,
            })
        return entries
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Transcript fetch failed: {exc}")
        return []


def _format_reflect(
    query: str,
    top_episode: Optional[Dict],
    supporting: List[Dict],
    dg_hits: List[Dict],
) -> str:
    lines = []

    # Main Memory
    lines.append("## Main Memory")
    if top_episode:
        lines.append(f'**The most relevant memory to "{query}"**')
        lines.append(top_episode.get("gist", ""))
    else:
        lines.append(f'No episode memories found for "{query}"')

    # Supporting Memories
    if supporting:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("### Supporting Memories")
        lines.append("__ordered by salience__")
        lines.append("")
        for entry in supporting:
            content = entry.get("content", "")
            salience = entry.get("salience")
            if salience is not None:
                lines.append(f"* {content} [salience: {salience}]")
            else:
                lines.append(f"* {content}")

    # Supporting facts from data_graph
    if dg_hits:
        lines.append("")
        lines.append("### Supporting facts:")
        for hit in dg_hits:
            key = hit.get("id", "")
            value = hit.get("text", "")
            lines.append(f"[{key}] {value}")

    return "\n".join(lines)


def _relevance_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _search_document_artifacts(query: str, limit: int = 3) -> List[Dict]:
    try:
        from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT

        dgs = get_data_graph_service()
        rows = dgs.recall(query=query, kinds=[KIND_DOCUMENT], limit=limit, expand_graph=False)

        if not rows:
            return []

        hits = []
        for row in rows:
            source = row.get('source', '') or ''
            if source.startswith('document:'):
                doc_id = source.split(':', 1)[1]
            else:
                parts = (row.get('key', '') or '').split(':')
                doc_id = parts[1] if len(parts) >= 3 else ''

            hits.append({
                "id": row.get("key", ""),
                "text": f"[document(id:{doc_id},type:fragment)] {row.get('value', '')}",
                "relevance": "high",
                "confidence": row.get("retrieval_weight", 1.0),
            })

        return hits

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Document artifact search failed: {e}")
        return []


def _search_data_graph(query: str, limit: int) -> Tuple[List[Dict], str]:
    try:
        from services.data_graph_service import get_data_graph_service, KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_MOMENT

        dgs = get_data_graph_service()
        rows = dgs.recall(query=query, kinds=[KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_MOMENT], limit=limit)

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            hits.append({
                "id": row.get("key", ""),
                "text": row.get("value", ""),
                "relevance": _relevance_label(row.get("retrieval_weight", 1.0)),
                "confidence": row.get("retrieval_weight", 1.0),
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
    baseline_radius: Optional[float] = None,
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
            return [], status

        if return_raw:
            return episodes[:limit], f"{len(episodes[:limit])} matches"

        hits = []
        for ep in episodes[:limit]:
            gist = ep.get("gist", "")
            conf = min(1.0, ep.get("composite_score", 0) / 100.0)
            hits.append({
                "id": str(ep.get("id", "")),
                "text": gist[:200],
                "relevance": _relevance_label(conf),
                "confidence": conf,
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


def _format_results(results: List[Dict]) -> str:
    lines = []
    for hit in results:
        rid = hit.get("id", "")
        relevance = hit.get("relevance", "low")
        text = hit.get("text", "")
        lines.append(f"[id:{rid},relevance:{relevance}] {text}")
    return "\n".join(lines)


def _store_fok_signal(channel: str, partial_match_count: int) -> None:
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{channel}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")

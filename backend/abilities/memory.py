"""
MemoryAbility — Store, recall, and forget first-party facts about the user.

Covers all actions: store, recall, reflect, forget.
Module-level radius constants are ClassVar for meta-harness patching.
"""

import hashlib
import json
import logging
import math
from typing import ClassVar, Dict, List, Optional, Tuple

from abilities._base import Ability
from services.innate_skills._tag import tag as _tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"
_KIND_BEHAVIORAL_PATTERN = "behavioral_pattern"


# ── Dynamic memory radius — tuning constants ────────────────────────────────
#
# Composition: effective_input = BASELINE × narrow_factor × expand_factor
# `episodic_retrieval_service.retrieve` then applies its own population-aware
# adaptive shrink on top. All eight constants are tuned by the meta-harness
# loop (loop_improve.sh) against the d1-context-recall benchmark suite.
#
# Do NOT read these values from config/env at import time. They are literal
# ClassVar floats so the meta-harness can diff-patch them mechanically.
# ─────────────────────────────────────────────────────────────────────────────


class MemoryAbility(Ability):
    NAME = "memory"
    SEARCH_TOOLTIP = "personal memory store"
    # SYSTEM tool: always allowed in every context and never shown in the Policy
    # Manager. Memory is core to Chalie's operation, so it bypasses policy like
    # skill_manager. (Excluded from policy_visible() → absent from defaults/meta.)
    SYSTEM = True
    SUMMARY = "Store, recall, or forget first-party facts about the user — traits, preferences, relationships, goals, and habits."
    EXAMPLES = [
        "please remember that my wifi password is BlueSky42",
        "what's my wifi password again",
        "I want you to forget that I told you my salary",
        "do you know anything about my dietary preferences",
        "save the fact that I have a dog named Biscuit",
        "reflect on everything you know about my exercise habits",
        "what happened at home last week",
        "recall conversations in Zabbar",
    ]
    INPUT_SCHEMA = {
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
                    "user_specific: first-party facts about the human you "
                    "are talking to (traits, preferences, relationships, "
                    "secrets, goals). system: about how Chalie operates "
                    "(rules, decisions, analysis). misc: short-lived "
                    "scratchpad for Chalie's own working notes — NOT a "
                    "dumping ground for user-supplied bulk content (use the "
                    "`document` tool for that)."
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
                "description": (
                    "For recall/reflect: what to search for. One topic per "
                    "call — to fetch memories about different topics, call "
                    "this tool once for each topic. If results are broad or "
                    "sparse, try searching again with more narrow queries."
                ),
            },
            "location": {
                "type": "string",
                "description": (
                    "A location to filter memories by. Use a city, country, "
                    "or a saved place name like 'home' or 'work'. "
                    "When set, only memories from that location are returned. "
                    "You can use location without query to get all memories "
                    "from a place regardless of topic."
                ),
            },
        },
        "required": ["action"],
    }

    RECALL_RADIUS_BASELINE: ClassVar[float] = 0.5
    SEED_RADIUS_BASELINE: ClassVar[float] = 0.4

    NARROW_MIN_DIST: ClassVar[float] = 0.25
    NARROW_MAX_DIST: ClassVar[float] = 0.05
    NARROW_FACTOR_FLOOR: ClassVar[float] = 0.35

    EXPAND_MIN_DIST: ClassVar[float] = 0.30
    EXPAND_MAX_DIST: ClassVar[float] = 0.55
    EXPAND_FACTOR_CEILING: ClassVar[float] = 2.2

    def run(self, params: dict) -> dict:
        action = params.get("action", "recall")
        mp = self.MessageProcessor
        channel = getattr(getattr(mp, "config", None), "channel", "") or ""

        try:
            if action == "store":
                text = _handle_store(channel, params)
            elif action == "recall":
                text = _handle_recall(mp, channel, params)
            elif action == "reflect":
                text = _handle_reflect(mp, channel, params)
            elif action == "forget":
                text = _handle_forget(params)
            else:
                text = _tag(
                    "memory",
                    error=f"unknown-action:{action}",
                    valid="store,recall,reflect,forget",
                )
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Error in {action}: {e}")
            text = _tag("memory", action=action, error=str(e)[:200])

        return {"text": text}


# ── Store ────────────────────────────────────────────────────────────


def _handle_store(channel: str, params: dict) -> str:
    key = params.get("key")
    value = params.get("value")
    kind = params.get("kind", "user_specific")

    if not key:
        return _tag("memory", action="store", error="key-required")
    if value is None:
        return _tag("memory", action="store", error="value-required")

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.store(kind=kind, key=key, value=str(value), source=f"skill:memory:store:{channel}")

    if result is None:
        return _tag("memory", action="store", key=key, error=f"store-failed-invalid-kind:{kind}")

    body = _format_store_response(result)
    return _tag("memory", body, action="store", key=key)


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


def _handle_forget(params: dict) -> str:
    key = params.get("key")
    value = params.get("value")
    kind = params.get("kind", "user_specific")

    if not key:
        return _tag("memory", action="forget", error="key-required")

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.forget(kind=kind, key=key, value=value)

    if result is None:
        return _tag("memory", action="forget", key=key, error="forget-failed-invalid-kind")

    body = _format_forget_response(result)
    return _tag("memory", body, action="forget", key=key)


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


def _handle_recall(mp, channel: str, params: dict) -> str:
    query = params.get("query", "")
    location = params.get("location", "")
    if not query and not location:
        return _tag("memory", action="recall", error="no-query-or-location")

    limit = 10
    results: List[Dict] = []

    if query and location:
        # AND gate: only episodes that satisfy both location AND semantic query.
        loc_hits, _ = _search_episodes_by_location(channel, location, limit * 3)
        loc_ids = {h["id"] for h in loc_hits}
        loc_by_id = {h["id"]: h for h in loc_hits}

        sem_hits, _ = _search_episodes(mp, channel, query, limit * 3)
        sem_ids = {h["id"] for h in sem_hits}
        sem_by_id = {h["id"]: h for h in sem_hits}

        matched_ids = loc_ids & sem_ids
        for ep_id in matched_ids:
            hit = dict(loc_by_id[ep_id])
            sem_hit = sem_by_id[ep_id]
            hit["confidence"] = sem_hit["confidence"]
            hit["relevance"] = sem_hit["relevance"]
            results.append(hit)

        dg_hits, _ = _search_data_graph(query, limit)
        results.extend(dg_hits)

    elif location:
        hits, _ = _search_episodes_by_location(channel, location, limit)
        results.extend(hits)

    else:
        hits, _ = _search_data_graph(query, limit)
        results.extend(hits)

        hits, _ = _search_episodes(mp, channel, query, limit)
        results.extend(hits)

    if not params.get('_auto'):
        try:
            proc = mp
            if proc is not None and proc._uid is not None:
                if query:
                    from abilities._base import Ability  # noqa: PLC0415
                    Ability.use(proc, 'document', {'action': 'search', 'query': query})
                    Ability.use(proc, 'schedule', {'action': 'search', 'query': query})
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} recall delegation failed: {exc}")

    partial = sum(1 for r in results if r.get("confidence", 0) < 0.5)
    _store_fok_signal(channel, partial)

    if not results:
        return _tag("memory", query=query or location, results=0)

    body = _format_results(results)
    return _tag("memory", body, query=query or location, results=len(results))


# ── Reflect ──────────────────────────────────────────────────────────


def _handle_reflect(mp, channel: str, params: dict) -> str:
    query = params.get("query", "")
    if not query:
        return _tag("memory", action="reflect", error="no-query")

    raw_episodes, _ = recall_episodes(
        mp,
        channel=channel,
        query=query,
        caller="llm_recall",
        limit=1,
        return_raw=True,
    )

    if not raw_episodes:
        dg_hits, _ = _search_data_graph(query, 2)
        if not dg_hits:
            return _tag("memory", action="reflect", query=query, results=0)
        body = _format_reflect(query, None, [], dg_hits)
        return _tag("memory", body, action="reflect", query=query)

    top = raw_episodes[0]

    supporting = _expand_episode_layers(top)

    dg_hits, _ = _search_data_graph(query, 2)

    body = _format_reflect(query, top, supporting, dg_hits)
    return _tag("memory", body, action="reflect", query=query)


def _expand_episode_layers(episode: Dict) -> List[Dict]:
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
    if depth >= max_depth:
        return

    consolidated_from = _parse_json_list(episode.get("consolidated_from"))
    transcript_ids = _parse_json_list(episode.get("transcript_ids"))

    if transcript_ids and not consolidated_from:
        results.extend(_fetch_transcript_entries(db, transcript_ids))
        return

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
    if not episode_ids:
        return []
    try:
        with db_service.connection() as conn:
            placeholders = ",".join("?" for _ in episode_ids)
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
    if not transcript_ids:
        return []
    try:
        with db_service.connection() as conn:
            placeholders = ",".join("?" for _ in transcript_ids)
            cursor = conn.execute(
                f"SELECT id, role, content, tool_name, created_at FROM transcript "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                list(transcript_ids),
            )
            rows = cursor.fetchall()

        entries = []
        for r in rows:
            content = r[2] or ""
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

    lines.append("## Main Memory")
    if top_episode:
        lines.append(f'**The most relevant memory to "{query}"**')
        lines.append(top_episode.get("gist", ""))
    else:
        lines.append(f'No episode memories found for "{query}"')

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


def _render_behavioral_pattern(raw_value: str) -> str:
    """Parse a behavioral_pattern JSON value into a compact human-readable line."""
    try:
        c = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        name = c.get("name", "unknown")
        freq = c.get("frequency", "?")
        anchor = c.get("time_anchor") or ""
        summary = c.get("summary", "")
        confidence = c.get("confidence", 0)
        anchor_part = f" @ {anchor}" if anchor else ""
        return f"{name} ({freq}{anchor_part}): {summary} [confidence={confidence}]"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return str(raw_value) if raw_value is not None else ""


def _search_data_graph(query: str, limit: int) -> Tuple[List[Dict], str]:
    try:
        from services.data_graph_service import (
            get_data_graph_service,
            KIND_BEHAVIORAL_PATTERN,
            KIND_PLACE,
            KIND_USER_SPECIFIC,
            KIND_SYSTEM,
            KIND_MISC,
            KIND_MOMENT,
        )

        dgs = get_data_graph_service()
        rows = dgs.recall(
            query=query,
            kinds=[KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_MOMENT,
                   KIND_BEHAVIORAL_PATTERN, KIND_PLACE],
            limit=limit,
        )

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            cos = row.get("cos_score", 0.0)
            kind = row.get("kind", "")
            text = row.get("value", "")
            if kind == KIND_BEHAVIORAL_PATTERN:
                text = _render_behavioral_pattern(text)
            hits.append({
                "id": row.get("key", ""),
                "kind": kind,
                "text": text,
                "relevance": _relevance_label(cos),
                "confidence": cos,
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Data graph search failed: {e}")
        return [], f"error: {e}"


def _cosine_distance(a: List[float], b: List[float]) -> float:
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


def _scan_history_distances(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float]:
    """Return (min_dist, max_drift) over all history embeddings."""
    min_dist = float("inf")
    max_drift = 0.0
    for entry in history:
        emb = entry.get("embedding")
        if not emb:
            continue
        d = _cosine_distance(q_embedding, emb)
        if d < min_dist:
            min_dist = d
        if d > max_drift:
            max_drift = d
    return min_dist, max_drift


def _compute_narrow_factor(min_dist: float) -> float:
    """Derive the narrow factor from the minimum history distance."""
    if min_dist == float("inf") or min_dist >= MemoryAbility.NARROW_MIN_DIST:
        return 1.0
    if min_dist <= MemoryAbility.NARROW_MAX_DIST:
        return MemoryAbility.NARROW_FACTOR_FLOOR
    span = MemoryAbility.NARROW_MIN_DIST - MemoryAbility.NARROW_MAX_DIST
    if span <= 0:
        return MemoryAbility.NARROW_FACTOR_FLOOR
    t = (MemoryAbility.NARROW_MIN_DIST - min_dist) / span
    f = 1.0 - t * (1.0 - MemoryAbility.NARROW_FACTOR_FLOOR)
    return max(MemoryAbility.NARROW_FACTOR_FLOOR, min(1.0, f))


def _compute_expand_factor(max_drift: float) -> float:
    """Derive the expand factor from the maximum history drift."""
    if max_drift <= MemoryAbility.EXPAND_MIN_DIST:
        return 1.0
    if max_drift >= MemoryAbility.EXPAND_MAX_DIST:
        return MemoryAbility.EXPAND_FACTOR_CEILING
    span = MemoryAbility.EXPAND_MAX_DIST - MemoryAbility.EXPAND_MIN_DIST
    if span <= 0:
        return MemoryAbility.EXPAND_FACTOR_CEILING
    t = (max_drift - MemoryAbility.EXPAND_MIN_DIST) / span
    f = 1.0 + t * (MemoryAbility.EXPAND_FACTOR_CEILING - 1.0)
    return min(MemoryAbility.EXPAND_FACTOR_CEILING, max(1.0, f))


def _compute_radius_factors(
    q_embedding: List[float], history: List[Dict]
) -> Tuple[float, float, float, float]:
    """Single-pass narrow + expand factor computation.

    Returns: (narrow_factor, expand_factor, min_dist, max_drift)
    """
    if not history:
        return 1.0, 1.0, float("inf"), 0.0

    min_dist, max_drift = _scan_history_distances(q_embedding, history)
    narrow_factor = _compute_narrow_factor(min_dist)
    expand_factor = _compute_expand_factor(max_drift)
    return narrow_factor, expand_factor, min_dist, max_drift


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


def _gather_query_history(proc, emb_svc, mp=None) -> tuple[list, str, object]:
    """Build (history, turn_uid, transcript_id) from the bound processor, if any.

    Inserts the seed-query record at the head of history when the processor has a
    pending seed that hasn't been recorded yet. Returns ephemeral defaults if no
    processor is bound.
    """
    if proc is None:
        return [], "ephemeral", None

    history: List[Dict] = list(proc._memory_query_history or [])
    seed_query = getattr(proc, "_memory_seed_query", None) or None
    if seed_query and not any(h.get("caller") == "seed" for h in history):
        try:
            seed_emb = emb_svc.generate_embedding(seed_query, mp=mp)
            history.insert(0, {
                "query": seed_query,
                "embedding": seed_emb,
                "caller": "seed",
                "effective_radius": MemoryAbility.SEED_RADIUS_BASELINE,
            })
        except Exception as _seed_exc:
            logger.debug(f"{LOG_PREFIX} Could not embed seed for drift calc: {_seed_exc}")
    _cfg = getattr(proc, "config", None)
    _channel = getattr(_cfg, "channel", None)
    turn_uid = str(getattr(proc, "_uid", None) or _channel or "unbound")
    transcript_id = getattr(proc, "_uid", None)
    return history, turn_uid, transcript_id


def recall_episodes(
    mp,
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
        from services import episodic_retrieval_service
        from services.database_service import get_shared_db_service
        from services.embedding_service import get_embedding_service
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Episode recall imports failed: {exc}")
        return [], f"error: {exc}"

    if baseline_radius is None:
        baseline_radius = (
            MemoryAbility.SEED_RADIUS_BASELINE if caller == "seed" else MemoryAbility.RECALL_RADIUS_BASELINE
        )

    try:
        db = get_shared_db_service()
        emb_svc = get_embedding_service()

        q_embedding = emb_svc.generate_embedding(query, mp=mp)
        proc = mp
        history, turn_uid, transcript_id = _gather_query_history(proc, emb_svc, mp)

        narrow_factor, expand_factor, _min_dist, _max_drift = _compute_radius_factors(q_embedding, history)
        input_radius = baseline_radius * narrow_factor * expand_factor

        episodes, telemetry = episodic_retrieval_service.retrieve(
            query_text=query,
            query_embedding=q_embedding,
            channel=channel,
            radius=input_radius,
            k=limit,
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
                "location": ep.get("location_name"),
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Episode search failed: {e}", exc_info=True)
        return [], f"error: {e}"


def _search_episodes(
    mp, channel: str, query: str, limit: int
) -> Tuple[List[Dict], str]:
    return recall_episodes(
        mp,
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


def _format_results(results: List[Dict]) -> str:
    """Format recall hits into tagged lines for LLM consumption.

    Labels behavioral_pattern hits explicitly. Includes location when present.
    Omits lat/lon to avoid token bloat.
    """
    lines = []
    for hit in results:
        rid = hit.get("id", "")
        relevance = hit.get("relevance", "low")
        text = hit.get("text", "")
        location = hit.get("location")
        kind = hit.get("kind", "")
        parts = [f"id:{rid}"]
        if kind == _KIND_BEHAVIORAL_PATTERN:
            parts.append("kind:behavioral_pattern")
        parts.append(f"relevance:{relevance}")
        if location:
            parts.append(f"at:{location}")
        prefix = f"[{','.join(parts)}]"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


_LOCATION_SEARCH_CONFIDENCE = 0.9
_LOCATION_SEARCH_RELEVANCE = "high"


def _search_episodes_by_location(
    channel: str, location: str, limit: int
) -> Tuple[List[Dict], str]:
    """Search episodes whose location_name contains the given text.

    Also resolves saved place labels (e.g. 'home') via data_graph kind='place'
    to pick up alternate location_name strings stored at save time.
    """
    try:
        from services.data_graph_service import KIND_PLACE, get_data_graph_service
        from services.database_service import get_shared_db_service

        # Build the list of strings to LIKE-match against location_name.
        # Start with the raw input and add any resolved name from saved places.
        location_names = [location]
        try:
            dgs = get_data_graph_service()
            places = dgs.fetch(kinds=[KIND_PLACE])
            for place in places:
                raw_value = place.get("value") or "{}"
                val = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
                if isinstance(val, dict) and place.get("key", "").lower() == location.lower():
                    place_name = val.get("name")
                    if place_name and place_name.lower() != location.lower():
                        location_names.append(place_name)
        except Exception as _resolve_exc:
            logger.debug("%s Place label resolution failed: %s", LOG_PREFIX, _resolve_exc)

        db = get_shared_db_service()
        like_clauses = " OR ".join(["e.location_name LIKE ?"] * len(location_names))
        like_params = [f"%{name}%" for name in location_names]

        with db.connection() as conn:
            sql = (
                "SELECT e.id, e.gist, e.location_name "
                "FROM episodes e "
                f"WHERE e.deleted_at IS NULL AND e.channel = ? AND ({like_clauses}) "
                "AND e.location_name IS NOT NULL "
                "ORDER BY e.created_at DESC LIMIT ?"
            )
            db_params = [channel] + like_params + [limit]
            rows = conn.execute(sql, db_params).fetchall()

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            ep_id, gist, loc_name = row
            hits.append({
                "id": str(ep_id),
                "text": (gist or "")[:200],
                "relevance": _LOCATION_SEARCH_RELEVANCE,
                "confidence": _LOCATION_SEARCH_CONFIDENCE,
                "location": loc_name,
            })

        return hits, f"{len(hits)} matches"

    except Exception as exc:
        logger.warning("%s Location episode search failed: %s", LOG_PREFIX, exc)
        return [], f"error: {exc}"


def _store_fok_signal(channel: str, partial_match_count: int) -> None:
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{channel}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")

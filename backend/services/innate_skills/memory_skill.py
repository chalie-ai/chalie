"""
Memory Skill — Unified knowledge storage and retrieval.

Replaces memorize_skill (store) and recall_skill (multi-source search)
with a single skill offering four actions: store, recall, update, forget.

Store/update/forget operate on the knowledge table via KnowledgeService.
Recall searches the knowledge store AND episodes AND transcript (preserving
the multi-source search from recall_skill).
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"

TOOL_SCHEMA = {
    "name": "memory",
    "description": (
        "Store, recall, update, or forget knowledge. Use this for any memory "
        "operation — facts, preferences, traits, concepts, procedures, metrics, "
        "or arbitrary key-value data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "recall", "update", "forget"],
                "description": (
                    "store: Save new knowledge. recall: Search memory. "
                    "update: Modify existing entry. forget: Remove knowledge."
                ),
            },
            # store params
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
            # recall params
            "query": {
                "type": "string",
                "description": "For recall action: what to search for.",
            },
            "kinds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by knowledge kinds.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10).",
            },
            "include_transcript": {
                "type": "boolean",
                "description": (
                    "Also search verbatim conversation transcript. "
                    "Use when you need to find what was actually said. Default false."
                ),
            },
            "transcript_topic": {
                "type": "string",
                "description": (
                    "Topic to search transcript in. Defaults to current topic. "
                    "Set to \"global\" to search across all topics."
                ),
            },
            "date_range": {
                "type": "object",
                "properties": {
                    "from": {
                        "type": "string",
                        "description": "ISO datetime lower bound (inclusive).",
                    },
                    "to": {
                        "type": "string",
                        "description": "ISO datetime upper bound (inclusive).",
                    },
                },
                "description": "Optional date range filter for transcript search.",
            },
            # update params
            "key": {
                "type": "string",
                "description": "For update/forget: the key to modify.",
            },
            "value": {
                "type": "string",
                "description": "For update: new value.",
            },
            "confidence": {
                "type": "number",
                "description": "For update: new confidence (0-1).",
            },
            # common
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


def handle_memory(topic: str, params: dict) -> str:
    """
    Unified memory operations: store, recall, update, forget.

    Args:
        topic: Current conversation topic
        params: Action parameters dict with 'action' key

    Returns:
        Formatted result string
    """
    action = params.get("action", "recall")

    try:
        if action == "store":
            return _handle_store(topic, params)
        elif action == "recall":
            return _handle_recall(topic, params)
        elif action == "update":
            return _handle_update(topic, params)
        elif action == "forget":
            return _handle_forget(topic, params)
        else:
            return (
                f"{LOG_PREFIX} Unknown action: {action}. "
                f"Valid: store, recall, update, forget"
            )
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Error in {action}: {e}", exc_info=True)
        return f"{LOG_PREFIX} Error: {e}"


# ── Store ────────────────────────────────────────────────────────────


def _handle_store(topic: str, params: dict) -> str:
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

            ks.store(
                entity=entity,
                key=key,
                value=str(value),
                kind=kind,
                decay_class=decay_class,
                data=data,
                source=f"skill:memory:store:{topic}",
            )
            stored += 1

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


def _handle_recall(topic: str, params: dict) -> str:
    """Search knowledge store, episodes, and optionally transcript."""
    query = params.get("query", "")
    if not query:
        return f"{LOG_PREFIX} Error: no query specified for recall."

    limit = min(params.get("limit", 10), 20)
    entity = params.get("entity")  # None = search all entities
    kinds = params.get("kinds")
    include_transcript = params.get("include_transcript", False)

    results: List[Dict] = []
    layer_status: Dict[str, str] = {}

    # 1. Knowledge store
    hits, status = _search_knowledge(entity, query, kinds, limit)
    layer_status["knowledge"] = status
    results.extend(hits)

    # 2. Episodes
    hits, status = _search_episodes(topic, query, limit)
    layer_status["episodes"] = status
    results.extend(hits)

    # 3. Transcript (opt-in)
    if include_transcript:
        hits, status = _search_transcript(topic, query, limit, params)
        layer_status["transcript"] = status
        results.extend(hits)

    # FOK signal for introspect
    partial = sum(1 for r in results if r.get("confidence", 0) < 0.5)
    _store_fok_signal(topic, partial)

    if not results:
        searched = ["knowledge", "episodes"]
        if include_transcript:
            searched.append("transcript")
        return _format_empty(searched, layer_status, query)

    return _format_results(results, query)


def _search_knowledge(
    entity: str, query: str, kinds: Optional[List[str]], limit: int
) -> Tuple[List[Dict], str]:
    """Search the knowledge store via KnowledgeService."""
    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        rows = ks.recall(entity=entity, query=query, kinds=kinds, limit=limit)

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


def _search_episodes(
    topic: str, query: str, limit: int
) -> Tuple[List[Dict], str]:
    """Search episodic memory via hybrid retrieval."""
    try:
        from services.episodic_service import EpisodicService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        service = EpisodicService(db)

        episodes = service.retrieve_episodes(
            query_text=query,
            topic=topic,
            limit=limit,
        )

        if not episodes:
            candidates = _count_episode_candidates(db, topic)
            return [], f"0 matches ({candidates} candidates evaluated)"

        hits = []
        for ep in episodes:
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
        logger.warning(f"{LOG_PREFIX} Episode search failed: {e}")
        return [], f"error: {e}"


def _search_transcript(
    topic: str, query: str, limit: int, params: dict
) -> Tuple[List[Dict], str]:
    """Search verbatim conversation transcript."""
    try:
        from services import transcript_service

        transcript_topic_param = params.get("transcript_topic", "")
        if transcript_topic_param == "global":
            search_topic = None
        elif transcript_topic_param:
            search_topic = transcript_topic_param
        else:
            search_topic = topic

        date_range = params.get("date_range") or {}
        date_from = date_range.get("from")
        date_to = date_range.get("to")

        results = transcript_service.search(
            search_topic, query, limit=limit,
            date_from=date_from, date_to=date_to,
        )

        if not results:
            scope = "global" if search_topic is None else search_topic
            return [], f"0 matches (scope: {scope})"

        hits = []
        for r in results:
            content = r.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            role = r.get("role", "unknown")
            tool_tag = f" [{r['tool_name']}]" if r.get("tool_name") else ""
            entry_topic = r.get("topic", topic)

            hits.append({
                "layer": "transcript",
                "content": f"[{role}{tool_tag}] {content}",
                "confidence": r.get("similarity", 0.5),
                "freshness": str(r.get("created_at", "")),
                "topic": entry_topic,
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Transcript search failed: {e}")
        return [], f"error: {e}"


def _count_episode_candidates(db_service, topic: str) -> int:
    """Count total episodes to distinguish empty from no-match."""
    try:
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL AND topic = ?",
                (topic,),
            )
            count = cursor.fetchone()[0]
            cursor.close()
        return count
    except Exception:
        return 0


# ── Update ───────────────────────────────────────────────────────────


def _handle_update(topic: str, params: dict) -> str:
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


# ── Forget ───────────────────────────────────────────────────────────


def _handle_forget(topic: str, params: dict) -> str:
    """Soft-delete a knowledge entry."""
    entity = params.get("entity", "user")
    key = params.get("key")
    if not key:
        return f"{LOG_PREFIX} Error: 'key' is required for forget."

    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        forgotten = ks.forget(entity=entity, key=key)

        if forgotten:
            return f"{LOG_PREFIX} Forgotten '{key}' for entity='{entity}'."
        return f"{LOG_PREFIX} No entry found for key='{key}', entity='{entity}'."

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Forget failed: {e}", exc_info=True)
        return f"{LOG_PREFIX} Forget failed: {e}"


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
            if "topic" in hit:
                extra += f", topic={hit['topic']}"
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


def _store_fok_signal(topic: str, partial_match_count: int) -> None:
    """Store partial match count in MemoryStore for introspect's FOK signal."""
    try:
        from services.memory_store import get_shared_store

        store = get_shared_store()
        store.setex(f"fok:{topic}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")

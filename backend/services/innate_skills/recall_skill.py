"""
Recall Skill — Unified memory retrieval across all layers.

Searches working memory, episodes, concepts, and user traits in one call.
Stores partial_match_count in MemoryStore for the introspect skill's FOK signal.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "recall",
    "description": (
        "Unified memory retrieval across all memory layers in one call. "
        "Searches working memory, episodes, concepts, and user traits. "
        "Use when you need to find what is known about something. "
        "Set include_transcript=true to also search verbatim conversation history "
        "(what was actually said). Use transcript_topic=\"global\" to search across "
        "all topics, not just the current one. "
        "For self-knowledge queries ('what do you know about me?'), use "
        "layers=[\"user_traits\"] with query \"user profile\" — returns all "
        "stored traits organized by category with confidence labels. "
        "Broad queries are capped at 15 traits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search text to look up across memory layers.",
            },
            "layers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["working_memory", "episodes", "concepts", "user_traits"],
                },
                "description": "Target specific memory layers. Omit to search all layers.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results per layer (default 3, max 10).",
            },
            "include_transcript": {
                "type": "boolean",
                "description": (
                    "Also search verbatim conversation transcript. "
                    "Use when you need to find what was actually said, "
                    "not just what is known. Default false."
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
        },
        "required": ["query"],
    },
}

ALL_LAYERS = ["working_memory", "episodes", "concepts", "user_traits"]

# Broad self-knowledge queries — return all traits, not keyword-filtered
BROAD_QUERIES = {
    "me", "myself", "user", "about me", "user profile", "everything", "all",
    "what do you know", "what do you remember", "profile",
}

# Soft cap for broad queries to avoid overwhelming responses
BROAD_TRAIT_DISPLAY_CAP = 15


def handle_recall(topic: str, params: dict) -> str:
    """
    Unified memory retrieval across all memory layers.

    Args:
        topic: Current conversation topic
        params: {query, layers (optional), limit (optional),
                 include_transcript (optional), transcript_topic (optional),
                 date_range (optional)}

    Returns:
        Formatted retrieval results with layer labels and confidence metadata
    """
    query = params.get("query", "")
    if not query:
        return "[RECALL] Error: no query specified."

    layers = params.get("layers", ALL_LAYERS)
    limit = min(params.get("limit", 3), 10)
    include_transcript = params.get("include_transcript", False)

    results = []
    layer_status = {}

    for layer in layers:
        if layer == "working_memory":
            hits, status = _search_working_memory(topic, query, limit)
        elif layer == "episodes":
            hits, status = _search_episodes(topic, query, limit)
        elif layer == "concepts":
            hits, status = _search_concepts(topic, query, limit)
        elif layer == "user_traits":
            hits, status = _search_user_traits(topic, query, limit)
        else:
            hits, status = [], f"unknown layer: {layer}"

        layer_status[layer] = status
        results.extend(hits)

    # Transcript search (opt-in)
    if include_transcript:
        hits, status = _search_transcript(topic, query, limit, params)
        layer_status["transcript"] = status
        results.extend(hits)

    # Store partial match count in MemoryStore for FOK signal
    partial_match_count = sum(
        1 for r in results if r.get("confidence", 0) < 0.5
    )
    _store_fok_signal(topic, partial_match_count)

    if not results:
        searched = layers + (["transcript"] if include_transcript else [])
        return _format_empty_results(searched, layer_status, query)

    return _format_results(results, query)


def _search_working_memory(topic: str, query: str, limit: int) -> tuple:
    """Search working memory turns for query keywords."""
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        key = f"working_memory:{topic}"
        turns = store.lrange(key, 0, -1)

        if not turns:
            return [], "empty"

        query_lower = query.lower()
        hits = []
        for turn in turns:
            if query_lower in turn.lower():
                hits.append({
                    "layer": "working_memory",
                    "content": turn[:200],
                    "confidence": 0.8,
                    "freshness": "current",
                })
                if len(hits) >= limit:
                    break

        if hits:
            return hits, f"{len(hits)} matches"
        return [], f"0 matches ({len(turns)} turns searched)"

    except Exception as e:
        logger.warning(f"[RECALL] working_memory search failed: {e}")
        return [], f"error: {e}"


def _search_episodes(topic: str, query: str, limit: int) -> tuple:
    """Search episodic memory via hybrid retrieval."""
    try:
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = EpisodicRetrievalService(db_service)

        episodes = service.retrieve_episodes(
            query_text=query,
            topic=topic,
            limit=limit,
        )

        if not episodes:
            # Count total candidates to distinguish "nothing exists" from "nothing matched"
            candidates = _count_episode_candidates(db_service, topic)
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
        logger.warning(f"[RECALL] episode search failed: {e}")
        return [], f"error: {e}"


def _search_concepts(topic: str, query: str, limit: int) -> tuple:
    """Search semantic concepts via hybrid retrieval."""
    try:
        from services.semantic_retrieval_service import SemanticRetrievalService
        from services.embedding_service import get_embedding_service
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        embedding_service = get_embedding_service()
        service = SemanticRetrievalService(db_service, embedding_service)

        concepts = service.retrieve_concepts(query=query, limit=limit)

        if not concepts:
            candidates = _count_concept_candidates(db_service)
            return [], f"0 matches ({candidates} candidates evaluated)"

        hits = []
        for concept in concepts:
            name = concept.get("concept_name", concept.get("name", "Unknown"))
            definition = concept.get("definition", "")
            hits.append({
                "layer": "concepts",
                "content": f"{name}: {definition[:150]}",
                "confidence": concept.get("confidence", 0.5),
                "freshness": "long-term",
                "strength": concept.get("strength", 0),
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"[RECALL] concept search failed: {e}")
        return [], f"error: {e}"


def _count_episode_candidates(db_service, topic: str) -> int:
    """Count total episodes for topic to distinguish empty from no-match."""
    conn = None
    try:
        conn = db_service.get_connection()
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
    finally:
        if conn:
            db_service.release_connection(conn)


def _count_concept_candidates(db_service) -> int:
    """Count total concepts to distinguish empty from no-match."""
    conn = None
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM semantic_concepts WHERE deleted_at IS NULL"
        )
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception:
        return 0
    finally:
        if conn:
            db_service.release_connection(conn)


def _search_user_traits(topic: str, query: str, limit: int) -> tuple:
    """
    Search user traits by keyword matching on trait key/value pairs.

    For broad self-knowledge queries ("what do you know about me?"), returns all
    traits above confidence threshold with a soft cap. For specific queries,
    keyword-matches on key and value. Includes meta fields so the LLM can
    phrase inferences naturally rather than stating them as facts.
    """
    try:
        from services.user_trait_service import UserTraitService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = UserTraitService(db_service)
        all_traits = service.get_all_traits()

        if not all_traits:
            return [], "0 traits stored"

        query_lower = query.lower()
        is_broad = query_lower in BROAD_QUERIES

        matched = []
        for t in all_traits:
            key = t.get('trait_key', '')
            value = t.get('trait_value', '')
            category = t.get('category', 'general')
            confidence = t.get('confidence', 0.0)

            if is_broad:
                if confidence >= 0.3:
                    matched.append(_format_trait_hit(key, value, category, confidence))
            else:
                if query_lower in key.lower() or query_lower in str(value).lower():
                    matched.append(_format_trait_hit(key, value, category, confidence))

        matched.sort(key=lambda h: h['confidence'], reverse=True)

        total_matched = len(matched)
        if is_broad and total_matched > BROAD_TRAIT_DISPLAY_CAP:
            matched = matched[:BROAD_TRAIT_DISPLAY_CAP]
        elif not is_broad:
            matched = matched[:limit]

        status = f"{len(matched)} matches ({len(all_traits)} traits evaluated)"
        if is_broad and total_matched > BROAD_TRAIT_DISPLAY_CAP:
            status += f" — showing top {BROAD_TRAIT_DISPLAY_CAP}, {total_matched - BROAD_TRAIT_DISPLAY_CAP} more available"

        return matched, status

    except Exception as e:
        logger.warning(f"[RECALL] user_traits search failed: {e}")
        return [], f"error: {e}"


def _search_transcript(topic: str, query: str, limit: int, params: dict) -> tuple:
    """Search verbatim conversation transcript via transcript_service."""
    try:
        from services import transcript_service

        # Determine topic scope
        transcript_topic_param = params.get("transcript_topic", "")
        if transcript_topic_param == "global":
            search_topic = None  # cross-topic
        elif transcript_topic_param:
            search_topic = transcript_topic_param  # specific topic
        else:
            search_topic = topic  # default: current topic

        # Date range
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
        logger.warning(f"[RECALL] transcript search failed: {e}")
        return [], f"error: {e}"


def _format_trait_hit(key: str, value: str, category: str, confidence: float) -> dict:
    """
    Format a user trait as a standard recall hit dict.

    Includes meta fields so the LLM can modulate tone based on confidence:
    - high confidence  → "Your name is Dylan."
    - medium           → "You seem to prefer dark themes."
    - low              → "I think you might enjoy cooking, but I'm not certain."
    """
    conf_label = "well established" if confidence >= 0.7 else "likely" if confidence >= 0.4 else "uncertain"
    return {
        "layer": "user_traits",
        "content": f"{key}: {value}",
        "confidence": confidence,
        "freshness": conf_label,
        "meta": {
            "category": category,
            "confidence_label": conf_label,
        },
    }


def _store_fok_signal(topic: str, partial_match_count: int) -> None:
    """Store partial match count in MemoryStore for introspect's FOK signal."""
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{topic}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"[RECALL] Failed to store FOK signal: {e}")


def _format_results(results: List[Dict], query: str) -> str:
    """Format retrieval results with layer headers."""
    # Group by layer
    by_layer = {}
    for r in results:
        layer = r["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(r)

    lines = [f"[RECALL] {len(results)} results for '{query}':"]
    for layer, hits in by_layer.items():
        lines.append(f"\n  [{layer}]")
        for hit in hits:
            conf = hit.get("confidence", 0)
            content = hit["content"]
            extra = ""
            if "salience" in hit:
                extra = f", salience={hit['salience']}"
            if "strength" in hit:
                extra = f", strength={hit['strength']:.2f}"
            if "meta" in hit:
                m = hit["meta"]
                extra = f", certainty={m.get('confidence_label','')}"
            if "topic" in hit:
                extra += f", topic={hit['topic']}"
            lines.append(f"    - {content} (confidence={conf:.2f}{extra})")

    return "\n".join(lines)


def _format_empty_results(layers: List[str], layer_status: Dict, query: str) -> str:
    """Format structured empty results for ACT to reason about gaps."""
    lines = [f"[RECALL] No matches found for '{query}' across {layers}:"]
    for layer in layers:
        status = layer_status.get(layer, "not searched")
        lines.append(f"  - {layer}: {status}")
    lines.append("Suggestion: Try broader query terms or use associate to explore related concepts.")
    return "\n".join(lines)

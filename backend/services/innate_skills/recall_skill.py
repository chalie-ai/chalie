"""
Recall Skill — Unified memory retrieval across all layers.

Searches working memory, gists, facts, episodes, concepts, and user traits in one call.
Stores partial_match_count in Redis for the introspect skill's FOK signal.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ALL_LAYERS = ["working_memory", "gists", "facts", "episodes", "concepts", "user_traits"]

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
        params: {query, layers (optional), limit (optional)}

    Returns:
        Formatted retrieval results with layer labels and confidence metadata
    """
    query = params.get("query", "")
    if not query:
        return "[RECALL] Error: no query specified."

    layers = params.get("layers", ALL_LAYERS)
    limit = min(params.get("limit", 3), 10)

    results = []
    layer_status = {}

    for layer in layers:
        if layer == "working_memory":
            hits, status = _search_working_memory(topic, query, limit)
        elif layer == "gists":
            hits, status = _search_gists(topic, query, limit)
        elif layer == "facts":
            hits, status = _search_facts(topic, query, limit)
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

    # Store partial match count in Redis for FOK signal
    partial_match_count = sum(
        1 for r in results if r.get("confidence", 0) < 0.5
    )
    _store_fok_signal(topic, partial_match_count)

    if not results:
        return _format_empty_results(layers, layer_status, query)

    return _format_results(results, query)


def _search_working_memory(topic: str, query: str, limit: int) -> tuple:
    """Search working memory turns for query keywords."""
    try:
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()
        key = f"working_memory:{topic}"
        turns = redis.lrange(key, 0, -1)

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


def _search_gists(topic: str, query: str, limit: int) -> tuple:
    """Search gists for query keywords."""
    try:
        from services.gist_storage_service import GistStorageService

        service = GistStorageService()
        gists = service.get_latest_gists(topic)

        if not gists:
            return [], "empty"

        query_lower = query.lower()
        hits = []
        for gist in gists:
            content = gist.get("content", "")
            if query_lower in content.lower():
                hits.append({
                    "layer": "gists",
                    "content": content[:200],
                    "confidence": gist.get("confidence", 5) / 10.0,
                    "freshness": "recent",
                    "type": gist.get("type", "unknown"),
                })
                if len(hits) >= limit:
                    break

        if hits:
            return hits, f"{len(hits)} matches"
        return [], f"0 matches ({len(gists)} gists searched)"

    except Exception as e:
        logger.warning(f"[RECALL] gist search failed: {e}")
        return [], f"error: {e}"


def _search_facts(topic: str, query: str, limit: int) -> tuple:
    """Search facts for query keywords."""
    try:
        from services.fact_store_service import FactStoreService

        service = FactStoreService()
        facts = service.get_all_facts(topic)

        if not facts:
            return [], "empty"

        query_lower = query.lower()
        hits = []
        for fact in facts:
            key = fact.get("key", "")
            value = fact.get("value", "")
            if query_lower in key.lower() or query_lower in str(value).lower():
                hits.append({
                    "layer": "facts",
                    "content": f"{key}: {value}",
                    "confidence": fact.get("confidence", 0.5),
                    "freshness": "medium-term",
                })
                if len(hits) >= limit:
                    break

        if hits:
            return hits, f"{len(hits)} matches"
        return [], f"0 matches ({len(facts)} facts searched)"

    except Exception as e:
        logger.warning(f"[RECALL] fact search failed: {e}")
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
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL AND topic = %s",
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
        all_traits = service.get_all_traits(user_id='primary')

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
            source = t.get('source', 'inferred')

            if is_broad:
                if confidence >= 0.3:
                    matched.append(_format_trait_hit(key, value, category, confidence, source))
            else:
                if query_lower in key.lower() or query_lower in str(value).lower():
                    matched.append(_format_trait_hit(key, value, category, confidence, source))

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


def _format_trait_hit(key: str, value: str, category: str, confidence: float, source: str = 'inferred') -> dict:
    """
    Format a user trait as a standard recall hit dict.

    Includes meta fields so the LLM can modulate tone based on source/confidence:
    - explicit + high confidence  → "Your name is Dylan."
    - inferred + medium           → "You seem to prefer dark themes."
    - inferred + low              → "I think you might enjoy cooking, but I'm not certain."
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
            "source": source,
        },
    }


def _store_fok_signal(topic: str, partial_match_count: int) -> None:
    """Store partial match count in Redis for introspect's FOK signal."""
    try:
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()
        redis.setex(f"fok:{topic}", 300, str(partial_match_count))
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
                extra = f", source={m.get('source','inferred')}, certainty={m.get('confidence_label','')}"
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

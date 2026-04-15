"""
Find Tools Skill — On-demand tool discovery with dynamic injection.

Cognitive primitive: always available in ACT mode. Lets Chalie search for
external capabilities (tools and interface actions) by semantic query.

After a successful search, the ACT orchestrator injects matching tool
schemas into the native tools list so the LLM can call them directly
in subsequent iterations — no second discovery step needed.

Search queries `tool_capability_profiles_vec` (same embeddings used by triage).

Zero-tool-name references in infrastructure — fully tool-agnostic.
"""

import logging
from typing import List, Dict

from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

LOG_PREFIX = "[FIND_TOOLS]"

TOOL_SCHEMA = {
    "name": "find_tools",
    "description": "Find tools for tasks your built-in skills can't handle. Returns usable tools.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Describe what you need.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
}


def handle_find_tools(channel: str, params: dict) -> dict:
    """
    Discover tools by semantic search.

    Returns:
        dict with 'text' (formatted results for the LLM) and
        '_discovered_tools' (tool names for the orchestrator to inject).
    """
    query = params.get("query", "").strip()
    logger.info(f"{LOG_PREFIX} query='{query}' limit={params.get('limit', 5)}")
    if not query:
        return {"text": "ERROR: 'query' is required.", "_discovered_tools": []}

    limit = min(params.get("limit", 5), 10)

    # Generate embedding for the query
    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        query_embedding = emb_service.generate_embedding(query)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Embedding generation failed: {e}")
        return _fallback_keyword_search(query, limit)

    blob = pack_embedding(query_embedding)

    # Query tool_capability_profiles_vec for nearest neighbors
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tcp.tool_name, tcp.tool_type, tcp.short_summary,
                       tcp.full_profile, tcp.effort,
                       v.distance, tcp.keywords
                FROM tool_capability_profiles_vec v
                JOIN tool_capability_profiles tcp ON tcp.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, limit + 5)  # over-fetch to allow filtering
            )
            rows = cursor.fetchall()
            cursor.close()

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Vector search failed: {e}")
        return _fallback_keyword_search(query, limit)

    if not rows:
        return {
            "text": _format_no_tools(query),
            "_discovered_tools": [],
        }

    # Debug: log raw k-NN results with distances
    for row in rows:
        name = row[0] if not isinstance(row, dict) else row['tool_name']
        dist = row[5] if not isinstance(row, dict) else row['distance']
        logger.info(f"{LOG_PREFIX} k-NN: {name} distance={dist:.4f}")

    # Filter to only registered tools (not innate skills) and check availability
    available_tools = _filter_available(rows, query)

    if not available_tools:
        return {
            "text": _format_no_tools(query),
            "_discovered_tools": [],
        }

    # Cap to requested limit
    available_tools = available_tools[:limit]

    text = _format_added_tools(available_tools)

    # Only inject tools that survived the relevance filter
    import json
    added = json.loads(text).get('added_tools', [])
    discovered_names = [t['name'] for t in added]

    if not discovered_names:
        return {
            "text": _format_no_tools(query),
            "_discovered_tools": [],
        }

    return {"text": text, "_discovered_tools": discovered_names}


# -- Search helpers --------------------------------------------------------


def _filter_available(rows: list, query: str = "") -> List[Dict]:
    """Filter vec search results to available, ready tools. Score by distance + keyword match."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
    except Exception:
        registry = None

    query_lower = query.lower()
    results = []
    for row in rows:
        tool_name = row[0] if not isinstance(row, dict) else row['tool_name']
        tool_type = row[1] if not isinstance(row, dict) else row['tool_type']
        short_summary = row[2] if not isinstance(row, dict) else row['short_summary']
        full_profile = row[3] if not isinstance(row, dict) else row['full_profile']
        effort = row[4] if not isinstance(row, dict) else row['effort']
        distance = row[5] if not isinstance(row, dict) else row['distance']
        keywords = row[6] if not isinstance(row, dict) else row.get('keywords', '')

        # Skip innate skills — they're already injected
        if tool_type == 'skill':
            continue

        # Check tool is registered and ready
        if registry:
            tool_data = registry.tools.get(tool_name)
            if not tool_data:
                continue
            if not registry._is_ready(tool_name, tool_data):
                continue
            if not registry._is_interface_online(tool_data):
                continue

        # 2-axis scoring: semantic distance + keyword match bonus
        kw_list = [k.strip().lower() for k in (keywords or '').split(',') if k.strip()]
        query_words = set(query_lower.split())
        kw_match_count = sum(
            1 for kw in kw_list
            if (' ' not in kw and kw in query_words) or (' ' in kw and kw in query_lower)
        )
        score = (distance * 10) - kw_match_count

        results.append({
            'tool_name': tool_name,
            'short_summary': short_summary,
            'full_profile': full_profile,
            'effort': effort,
            'score': score,
            'distance': distance,
        })

    # Lower score = better (closer semantically + more keyword matches)
    results.sort(key=lambda t: t['score'])
    return results


_MIN_RELEVANCE = 0.15


def _format_added_tools(tools: List[Dict]) -> str:
    """Format discovered tools as JSON for tool_calls storage.

    Relevance derived from raw sqlite-vec distance (0–2 range, lower=closer).
    Keyword bonus already baked into sort order via score; relevance is purely
    the semantic signal so the LLM knows how confident the match is.
    """
    import json
    entries = []
    for t in tools:
        dist = t.get('distance', 2.0)
        relevance = round(max(0.0, min(1.0, 1.0 - dist / 2.0)), 2)
        if relevance < _MIN_RELEVANCE:
            continue
        entries.append({"name": t['tool_name'], "relevance": relevance})
    return json.dumps({"added_tools": entries})


def _format_no_tools(query: str) -> str:
    """Format the no-new-tools message."""
    return f'INFO: The best tools for "{query}" are already available.'



# -- Fallback --------------------------------------------------------------


def _fallback_keyword_search(query: str, limit: int) -> dict:
    """Keyword-based fallback when embedding service is unavailable."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        query_pattern = f"%{query}%"
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tool_name, tool_type, short_summary, effort
                FROM tool_capability_profiles
                WHERE tool_type = 'tool'
                  AND (short_summary LIKE ? OR full_profile LIKE ? OR tool_name LIKE ?)
                ORDER BY tool_name
                LIMIT ?
                """,
                (query_pattern, query_pattern, query_pattern, limit)
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return {
                "text": _format_no_tools(query),
                "_discovered_tools": [],
            }

        discovered = []
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row['tool_name']
            discovered.append(name)

        import json
        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        return {
            "text": json.dumps({"added_tools": entries}),
            "_discovered_tools": discovered,
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return {
            "text": f"ERROR: Tool search unavailable — {e}",
            "_discovered_tools": [],
        }

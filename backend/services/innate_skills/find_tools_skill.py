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
    "description": (
        "When a task requires actions you cannot perform with your built-in skills "
        "— such as calling external APIs, fetching web content, or running specialized "
        "processing — use this to discover available tools. Describe what you need; "
        "matched tools become directly callable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Describe the capability you need in natural language.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
}


def handle_find_tools(topic: str, params: dict) -> dict:
    """
    Discover tools by semantic search.

    Returns:
        dict with 'text' (formatted results for the LLM) and
        '_discovered_tools' (tool names for the orchestrator to inject).
    """
    query = params.get("query", "").strip()
    if not query:
        return {"text": f"{LOG_PREFIX} Error: 'query' is required.", "_discovered_tools": []}

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
                       tcp.full_profile, tcp.domain, tcp.effort,
                       v.distance
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
            "text": f"{LOG_PREFIX} No tools found matching '{query}'.",
            "_discovered_tools": [],
        }

    # Filter to only registered tools (not innate skills) and check availability
    available_tools = _filter_available(rows)

    if not available_tools:
        # Collect matching innate skill names so the model knows what to use
        matched_skills = []
        for row in rows:
            tool_type = row[1] if not isinstance(row, dict) else row['tool_type']
            if tool_type == 'skill':
                name = row[0] if not isinstance(row, dict) else row['tool_name']
                summary = row[2] if not isinstance(row, dict) else row['short_summary']
                matched_skills.append((name, summary))
        if matched_skills:
            skill_lines = "\n".join(
                f"  - **{name}**: {summary}" for name, summary in matched_skills[:5]
            )
            hint = (
                f"{LOG_PREFIX} No external tools match '{query}', but these "
                f"built-in skills are already available to you:\n{skill_lines}\n"
                "Call them directly — they are in your current tool list."
            )
        else:
            hint = f"{LOG_PREFIX} No tools found matching '{query}'."
        return {
            "text": hint,
            "_discovered_tools": [],
        }

    # Cap to requested limit
    available_tools = available_tools[:limit]

    discovered_names = [t['tool_name'] for t in available_tools]
    text = _format_search_results(query, available_tools)

    return {"text": text, "_discovered_tools": discovered_names}


# -- Search helpers --------------------------------------------------------


def _filter_available(rows: list) -> List[Dict]:
    """Filter vec search results to available, ready tools."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
    except Exception:
        registry = None

    results = []
    for row in rows:
        tool_name = row[0] if not isinstance(row, dict) else row['tool_name']
        tool_type = row[1] if not isinstance(row, dict) else row['tool_type']
        short_summary = row[2] if not isinstance(row, dict) else row['short_summary']
        full_profile = row[3] if not isinstance(row, dict) else row['full_profile']
        domain = row[4] if not isinstance(row, dict) else row['domain']
        effort = row[5] if not isinstance(row, dict) else row['effort']
        distance = row[6] if not isinstance(row, dict) else row['distance']

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

        # Convert L2 distance to similarity for normalized vectors
        similarity = max(0.0, 1.0 - distance / 2.0)

        results.append({
            'tool_name': tool_name,
            'short_summary': short_summary,
            'full_profile': full_profile,
            'domain': domain,
            'effort': effort,
            'similarity': similarity,
        })

    return results


def _format_search_results(query: str, tools: List[Dict]) -> str:
    """Format search results for the LLM."""
    lines = [f"{LOG_PREFIX} Found {len(tools)} tool(s) matching '{query}':\n"]

    for t in tools:
        param_str = _get_param_summary(t['tool_name'])
        sim_pct = int(t['similarity'] * 100)

        lines.append(f"- **{t['tool_name']}** ({sim_pct}% match, {t['effort']}): {t['short_summary']}")
        if param_str:
            lines.append(f"  Parameters: {param_str}")

    lines.append("\nThese tools are now available for you to call directly.")
    return "\n".join(lines)


def _get_param_summary(tool_name: str) -> str:
    """Get compact parameter summary from tool manifest."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        tool_data = registry.tools.get(tool_name)
        if not tool_data:
            return ""

        manifest = tool_data['manifest']

        if 'input_schema' in manifest:
            schema_props = manifest['input_schema'].get('properties', {})
            schema_required = manifest['input_schema'].get('required', [])
            if not schema_props:
                return "(no parameters)"
            parts = []
            for pname in schema_props:
                parts.append(pname if pname in schema_required else f"{pname}?")
            return f"({', '.join(parts)})"

        params = manifest.get('parameters', {})
        if not params:
            return "(no parameters)"

        parts = []
        for pname, pdef in params.items():
            required = pdef.get('required', False)
            parts.append(pname if required else f"{pname}?")
        return f"({', '.join(parts)})"
    except Exception:
        return ""


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
                SELECT tool_name, tool_type, short_summary, domain, effort
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
                "text": f"{LOG_PREFIX} No tools found matching '{query}' (keyword fallback).",
                "_discovered_tools": [],
            }

        discovered = []
        lines = [f"{LOG_PREFIX} Found {len(rows)} tool(s) matching '{query}' (keyword search):\n"]
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row['tool_name']
            summary = row[2] if not isinstance(row, dict) else row['short_summary']
            effort = row[4] if not isinstance(row, dict) else row['effort']
            param_str = _get_param_summary(name)
            lines.append(f"- **{name}** ({effort}): {summary}")
            if param_str:
                lines.append(f"  Parameters: {param_str}")
            discovered.append(name)

        lines.append("\nThese tools are now available for you to call directly.")
        return {"text": "\n".join(lines), "_discovered_tools": discovered}

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return {
            "text": f"{LOG_PREFIX} Tool search unavailable. Error: {e}",
            "_discovered_tools": [],
        }

"""
Find Tools Skill — On-demand external tool discovery.

Cognitive primitive: always available in ACT mode. Lets Chalie search for
external capabilities (tools and interface actions) by semantic query instead
of having all tool descriptions pre-injected into the prompt.

Search queries `tool_capability_profiles_vec` (same embeddings used by triage).
Details returns full invocation guide including parameters.

Zero-tool-name references in infrastructure — fully tool-agnostic.
"""

import json
import logging
import struct
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[FIND_TOOLS]"


def handle_find_tools(topic: str, params: dict) -> str:
    """
    Discover external tools by semantic search or get invocation details.

    Args:
        topic: Current conversation topic (unused, present for handler contract)
        params: {
            action: "search" | "details"
            query: str (for search — what capability is needed)
            tool_name: str (for details — which tool to inspect)
            limit: int (optional, default 5, max 10)
        }

    Returns:
        Formatted tool discovery results or invocation guide.
    """
    action = params.get("action", "search")

    if action == "search":
        return _search_tools(params)
    elif action == "details":
        return _get_tool_details(params)
    else:
        return f"{LOG_PREFIX} Unknown action: {action}. Use 'search' or 'details'."


# ── Search ────────────────────────────────────────────────────────────────


def _search_tools(params: dict) -> str:
    """Semantic search across tool capability profiles."""
    query = params.get("query", "").strip()
    if not query:
        return f"{LOG_PREFIX} Error: 'query' parameter is required for search."

    limit = min(params.get("limit", 5), 10)

    # Generate embedding for the query
    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        query_embedding = emb_service.generate_embedding(query)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Embedding generation failed: {e}")
        return _fallback_keyword_search(query, limit)

    blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

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
        return f"{LOG_PREFIX} No tools found matching '{query}'. No external capabilities are currently registered."

    # Filter to only external tools (not innate skills) and check availability
    available_tools = _filter_available(rows)

    if not available_tools:
        return f"{LOG_PREFIX} No available external tools match '{query}'. Only innate skills matched — those are already at your disposal."

    # Cap to requested limit
    available_tools = available_tools[:limit]

    return _format_search_results(query, available_tools)


def _filter_available(rows: list) -> List[Dict]:
    """Filter vec search results to only available, ready external tools."""
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
    """Format search results with enough detail for the ACT loop to decide."""
    lines = [f"{LOG_PREFIX} Found {len(tools)} tool(s) matching '{query}':\n"]

    for t in tools:
        # Get parameter info from registry
        param_str = _get_param_summary(t['tool_name'])
        sim_pct = int(t['similarity'] * 100)

        lines.append(f"### {t['tool_name']} (relevance: {sim_pct}%, effort: {t['effort']})")
        lines.append(f"{t['short_summary']}")
        if param_str:
            lines.append(f"Parameters: {param_str}")
        lines.append("")

    lines.append("Use find_tools with action='details' and tool_name='<name>' for full invocation guide.")
    lines.append("To invoke a tool, use it as an action type: {\"type\": \"<tool_name>\", ...params}")
    return "\n".join(lines)


def _get_param_summary(tool_name: str) -> str:
    """Get compact parameter summary from tool manifest."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        tool_data = registry.tools.get(tool_name)
        if not tool_data:
            return ""

        params = tool_data['manifest'].get('parameters', {})
        if not params:
            return "(no parameters)"

        parts = []
        for pname, pdef in params.items():
            required = pdef.get('required', False)
            parts.append(pname if required else f"{pname}?")
        return f"({', '.join(parts)})"
    except Exception:
        return ""


# ── Details ───────────────────────────────────────────────────────────────


def _get_tool_details(params: dict) -> str:
    """Get full invocation guide for a specific tool."""
    tool_name = params.get("tool_name", "").strip()
    if not tool_name:
        return f"{LOG_PREFIX} Error: 'tool_name' parameter is required for details."

    # Get profile from DB
    profile = _fetch_profile(tool_name)

    # Get manifest from registry
    manifest = _fetch_manifest(tool_name)

    if not profile and not manifest:
        return f"{LOG_PREFIX} Tool '{tool_name}' not found. Use search to discover available tools."

    lines = [f"{LOG_PREFIX} Tool details for '{tool_name}':\n"]

    # Profile info
    if profile:
        lines.append(f"## Description")
        lines.append(profile.get('full_profile', profile.get('short_summary', '')))
        lines.append("")

        scenarios = profile.get('usage_scenarios', [])
        if isinstance(scenarios, str):
            try:
                scenarios = json.loads(scenarios)
            except (json.JSONDecodeError, TypeError):
                scenarios = []
        if scenarios:
            lines.append(f"## When to use")
            for s in scenarios[:5]:
                lines.append(f"- {s}")
            lines.append("")

        anti_scenarios = profile.get('anti_scenarios', [])
        if isinstance(anti_scenarios, str):
            try:
                anti_scenarios = json.loads(anti_scenarios)
            except (json.JSONDecodeError, TypeError):
                anti_scenarios = []
        if anti_scenarios:
            lines.append(f"## When NOT to use")
            for s in anti_scenarios[:3]:
                lines.append(f"- {s}")
            lines.append("")

    # Manifest parameter details
    if manifest:
        params_def = manifest.get('parameters', {})
        if params_def:
            lines.append(f"## Parameters")
            for pname, pdef in params_def.items():
                required = "required" if pdef.get('required', False) else "optional"
                ptype = pdef.get('type', 'string')
                desc = pdef.get('description', '')
                default = pdef.get('default')
                default_str = f", default: {default}" if default is not None else ""
                lines.append(f"- `{pname}` ({ptype}, {required}{default_str}): {desc}")
            lines.append("")

        lines.append(f"## Invocation")
        lines.append(f'```json\n{{"type": "{tool_name}", {_build_example_params(params_def)}}}\n```')

    # Performance hint
    perf = _get_performance(tool_name)
    if perf:
        lines.append(f"\n## Performance")
        lines.append(perf)

    return "\n".join(lines)


def _fetch_profile(tool_name: str) -> Optional[Dict]:
    """Fetch tool profile from database."""
    try:
        from services.tool_profile_service import ToolProfileService
        return ToolProfileService().get_full_profile(tool_name)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Profile fetch failed for {tool_name}: {e}")
        return None


def _fetch_manifest(tool_name: str) -> Optional[Dict]:
    """Fetch tool manifest from registry."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        tool_data = registry.tools.get(tool_name)
        return tool_data['manifest'] if tool_data else None
    except Exception:
        return None


def _build_example_params(params_def: dict) -> str:
    """Build example parameter JSON for invocation guide."""
    if not params_def:
        return ""
    parts = []
    for pname, pdef in params_def.items():
        if pdef.get('required', False):
            parts.append(f'"{pname}": "..."')
    return ", ".join(parts) if parts else '"param": "..."'


def _get_performance(tool_name: str) -> str:
    """Get performance summary string."""
    try:
        from services.tool_performance_service import ToolPerformanceService
        service = ToolPerformanceService()
        stats = service.get_tool_stats(tool_name)
        if not stats or stats.get('total', 0) < 3:
            return ""
        success_pct = int(stats['success_rate'] * 100)
        avg_ms = int(stats['avg_latency'])
        label = 'reliable' if success_pct >= 80 else 'moderate' if success_pct >= 50 else 'unreliable'
        return f"{label} — {success_pct}% success rate, ~{avg_ms}ms avg latency, {stats['total']} invocations"
    except Exception:
        return ""


# ── Fallback ──────────────────────────────────────────────────────────────


def _fallback_keyword_search(query: str, limit: int) -> str:
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
            return f"{LOG_PREFIX} No tools found matching '{query}' (keyword fallback)."

        lines = [f"{LOG_PREFIX} Found {len(rows)} tool(s) matching '{query}' (keyword search):\n"]
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row['tool_name']
            summary = row[2] if not isinstance(row, dict) else row['short_summary']
            effort = row[4] if not isinstance(row, dict) else row['effort']
            param_str = _get_param_summary(name)
            lines.append(f"- **{name}** ({effort}): {summary}")
            if param_str:
                lines.append(f"  Parameters: {param_str}")

        lines.append("\nUse find_tools with action='details' for full invocation guide.")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return f"{LOG_PREFIX} Tool search unavailable. Error: {e}"

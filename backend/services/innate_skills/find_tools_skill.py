"""
Find Tools Skill — On-demand tool discovery with dynamic injection.

Cognitive primitive: always available in ACT mode. Lets Chalie search for
external capabilities (tools and interface actions) by semantic query.

After a successful search, the ACT orchestrator injects matching tool
schemas into the native tools list so the LLM can call them directly
in subsequent iterations — no second discovery step needed.

Search queries `tool_capability_profiles_vec` (same embeddings used by triage).
Phase 1: also queries abilities.sqlite (new DB) with priority merge — new DB
wins when an entry exists.

Zero-tool-name references in infrastructure — fully tool-agnostic.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

from services.embedding_utils import pack_embedding
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)

LOG_PREFIX = "[FIND_TOOLS]"

TOOL_SCHEMA = {
    "name": "find_tools",
    "description": (
        "Find tools for tasks your built-in skills can't handle. "
        "Returns a list of tools that are now available to you. "
        "IMPORTANT: after this returns, if any listed tool matches the user's request, "
        "INVOKE it in the same turn to complete the task. Do not respond to the user "
        "without invoking the matching tool first."
    ),
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
        return {"text": _skill_tag("find_tools", error="query-required"), "_discovered_tools": []}

    limit = min(params.get("limit", 5), 10)

    # Generate embedding for the query — falls back to keyword search on failure
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

    # Phase 1 dual-read: query abilities.sqlite for nearest neighbors and
    # merge with priority — new DB wins when an entry exists (§4.1).
    new_db_rows = _query_abilities_db(blob, limit + 5)

    if new_db_rows:
        # Build set of names already covered by the new DB
        new_db_names = {r['tool_name'] for r in new_db_rows}
        # Drop old-DB rows whose name is already in new DB
        old_rows_filtered = [r for r in rows if r[0] not in new_db_names]
        # Concat: new DB first (lower/better distances from a fresh model)
        rows = new_db_rows + old_rows_filtered

    if not rows:
        return {
            "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
            "_discovered_tools": [],
        }

    # Debug: log raw k-NN results with distances
    for row in rows:
        name = row[0] if not isinstance(row, dict) else row['tool_name']
        dist = row[6] if not isinstance(row, dict) else row['distance']
        logger.info(f"{LOG_PREFIX} k-NN: {name} distance={dist:.4f}")

    # Filter to only registered tools (not innate skills) and check availability
    available_tools = _filter_available(rows, query)

    if not available_tools:
        return {
            "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
            "_discovered_tools": [],
        }

    # Cap to requested limit
    available_tools = available_tools[:limit]

    raw_text = _format_added_tools(available_tools)

    # Only inject tools that survived the relevance filter
    import json
    added = json.loads(raw_text).get('added_tools', [])
    discovered_names = [t['name'] for t in added]

    if not discovered_names:
        return {
            "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
            "_discovered_tools": [],
        }

    return {
        "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered_names)),
        "_discovered_tools": discovered_names,
    }


# -- New-DB (abilities.sqlite) query ---------------------------------------

_ABILITIES_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "abilities" / "assets" / "abilities.sqlite"
)


def _query_abilities_db(blob: bytes, k: int) -> List[Dict]:
    """Query abilities.sqlite for nearest neighbors.

    Returns a list of result dicts in the same shape _filter_available produces,
    with score = distance * 10 (no keyword bonus — Phase 3 enrichment).
    Returns [] if the DB is missing, empty, or fails to open.
    """
    if not _ABILITIES_DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_ABILITIES_DB_PATH))
        try:
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                conn.load_extension("vec0")

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.name, a.summary, v.distance
                FROM ability_search_vec v
                JOIN ability_search_entries ase ON ase.id = v.rowid
                JOIN abilities a ON a.id = ase.ability_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, k),
            )
            db_rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} abilities.sqlite query failed: {e}")
        return []

    if not db_rows:
        return []

    # Deduplicate by ability name — multiple entries (summary + examples) may match.
    seen: set = set()
    results: List[Dict] = []
    for name, summary, distance in db_rows:
        if name in seen:
            continue
        seen.add(name)
        score = distance * 10
        results.append({
            'tool_name': name,
            'short_summary': summary,
            'full_profile': summary,
            'domain': '',
            'effort': 'low',
            'score': score,
            'distance': distance,
        })

    return results


# -- Search helpers --------------------------------------------------------


def _filter_available(rows: list, query: str = "") -> List[Dict]:
    """Filter vec search results to available, ready tools. Score by distance + keyword match.

    Rows may be either:
    - 8-tuples from tool_capability_profiles_vec (old DB)
    - result dicts with 'score' already set (new abilities.sqlite DB — bypass registry checks)
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
    except Exception:
        registry = None

    query_lower = query.lower()
    results = []
    for row in rows:
        # New-DB rows arrive as complete result dicts — pass through directly.
        if isinstance(row, dict) and 'score' in row:
            results.append(row)
            continue

        tool_name = row[0] if not isinstance(row, dict) else row['tool_name']
        tool_type = row[1] if not isinstance(row, dict) else row['tool_type']
        short_summary = row[2] if not isinstance(row, dict) else row['short_summary']
        full_profile = row[3] if not isinstance(row, dict) else row['full_profile']
        domain = row[4] if not isinstance(row, dict) else row['domain']
        effort = row[5] if not isinstance(row, dict) else row['effort']
        distance = row[6] if not isinstance(row, dict) else row['distance']
        keywords = row[7] if not isinstance(row, dict) else row.get('keywords', '')

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
            'domain': domain,
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
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        discovered = []
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row['tool_name']
            discovered.append(name)

        import json
        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        raw_text = json.dumps({"added_tools": entries})
        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered)),
            "_discovered_tools": discovered,
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return {
            "text": _skill_tag("find_tools", error=f"tool-search-unavailable:{str(e)[:100]}"),
            "_discovered_tools": [],
        }

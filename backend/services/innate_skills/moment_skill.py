"""
Moment Skill — Search and list pinned moments via the ACT loop.

Actions: search, list
"""

import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "moment",
    "description": (
        "Capture and read ambient context snapshots (time, place, energy, device). "
        "Search for pinned moments by semantic query or list all active moments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list"],
                "description": "Operation: 'search' finds moments by query; 'list' shows all active moments.",
            },
            "query": {
                "type": "string",
                "description": "Text to search for matching moments. Required for search action.",
            },
        },
        "required": ["action"],
    },
}


def handle_moment(topic: str, params: dict) -> str:
    """
    Manage pinned moments.

    Actions:
    - search:  Semantic search for a moment by query
    - list:    Show all active moments

    Args:
        topic: Current conversation topic
        params: Action parameters dict

    Returns:
        Formatted result string
    """
    action = params.get("action", "search")

    try:
        from services.moment_service import MomentService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        service = MomentService(db)

        if action == "search":
            return _handle_search(service, params, topic)
        elif action == "list":
            return _handle_list(service, topic)
        else:
            return f"[MOMENT] Unknown action '{action}'. Use: search, list"

    except Exception as e:
        logger.error(f"[MOMENT SKILL] Error: {e}", exc_info=True)
        return f"[MOMENT] Error: {e}"


def _handle_search(service, params: dict, topic: str) -> str:
    query = params.get("query", "").strip()
    if not query:
        return "[MOMENT] 'query' is required for search."

    results = service.search_moments(query, limit=3)

    if not results:
        return "[MOMENT] No matching moments found."

    lines = ["[MOMENT] Found:"]
    for m in results:
        title = m.get("title") or "Untitled"
        summary = m.get("summary") or m.get("message_text", "")[:80]
        lines.append(f"  - {title}: {summary}")

    return "\n".join(lines)


def _handle_list(service, topic: str) -> str:
    moments = service.get_all_moments()

    if not moments:
        return "[MOMENT] No moments found."

    lines = ["[MOMENT] All moments:"]
    for m in moments:
        title = m.get("title") or "Untitled"
        status = m.get("status", "")
        lines.append(f"  - {title} ({status})")

    return "\n".join(lines)

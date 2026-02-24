"""
Moment Skill — Search and list pinned moments via the ACT loop.

Actions: search, list
"""

import logging

logger = logging.getLogger(__name__)


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

    if results:
        # Emit best match as card
        try:
            from services.moment_card_service import MomentCardService
            MomentCardService().emit_moment_card(topic, results[0])
        except Exception as card_err:
            logger.warning(f"[MOMENT SKILL] Card emit failed (non-fatal): {card_err}")

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

    if moments:
        try:
            from services.moment_card_service import MomentCardService
            MomentCardService().emit_moment_list_card(topic, moments)
        except Exception as card_err:
            logger.warning(f"[MOMENT SKILL] Card emit failed (non-fatal): {card_err}")

    if not moments:
        return "[MOMENT] No moments found."

    lines = ["[MOMENT] All moments:"]
    for m in moments:
        title = m.get("title") or "Untitled"
        status = m.get("status", "")
        lines.append(f"  - {title} ({status})")

    return "\n".join(lines)

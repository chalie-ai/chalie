"""
Focus Skill — Manage focus sessions via the ACT loop.

Actions: set, check, clear
"""

import logging

logger = logging.getLogger(__name__)


def handle_focus(topic: str, params: dict) -> str:
    """
    Manage focus sessions for the current thread.

    Actions:
    - set: Declare a focus session with a description
    - check: Return current focus status + distraction signal for latest message
    - clear: End the current focus session

    Args:
        topic: Current conversation topic
        params: Action parameters

    Returns:
        Formatted result string
    """
    action = params.get('action', 'check')

    # Retrieve thread_id from params or use topic as fallback
    thread_id = params.get('thread_id', topic)

    try:
        from services.focus_session_service import FocusSessionService
        service = FocusSessionService()

        if action == 'set':
            return _handle_set(service, thread_id, topic, params)
        elif action == 'check':
            return _handle_check(service, thread_id)
        elif action == 'clear':
            return _handle_clear(service, thread_id)
        else:
            return f"[FOCUS] Unknown action '{action}'. Use: set, check, clear"

    except Exception as e:
        logger.error(f"[FOCUS SKILL] Error: {e}", exc_info=True)
        return f"[FOCUS] Error: {e}"


def _handle_set(service, thread_id: str, topic: str, params: dict) -> str:
    """Set a focus session."""
    description = params.get('description', '').strip()
    if not description:
        return "[FOCUS] 'description' is required to set focus."

    success = service.set_focus(
        thread_id=thread_id,
        description=description,
        topic=topic,
        source='explicit',
    )

    if success:
        return f"[FOCUS] Focus set: '{description}'"
    else:
        return "[FOCUS] Failed to set focus session."


def _handle_check(service, thread_id: str) -> str:
    """Report current focus status."""
    focus = service.get_focus(thread_id)
    if not focus:
        return "[FOCUS] No active focus session."

    source_label = "declared" if focus.get('source') == 'explicit' else "inferred"
    modifier = service.get_boundary_modifier(thread_id)
    return (
        f"[FOCUS] Active focus ({source_label}): '{focus['description']}'\n"
        f"  Topic: {focus.get('topic', '(none)')}\n"
        f"  Boundary modifier: +{modifier:.1f}"
    )


def _handle_clear(service, thread_id: str) -> str:
    """Clear the current focus session."""
    focus = service.get_focus(thread_id)
    if not focus:
        return "[FOCUS] No active focus session to clear."

    service.clear_focus(thread_id)
    return f"[FOCUS] Focus session ended: '{focus['description']}'"

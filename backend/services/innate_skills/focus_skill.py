"""
Focus Skill — Manage focus sessions via the ACT loop.

Actions: set, check, clear
"""

import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "focus",
    "description": (
        "Manage per-thread focus sessions that gate distraction and raise topic boundaries. "
        "Use when the user declares a deep work session, or to check distraction status "
        "when a focus session is active. Never use focus to block the user — it is a signal "
        "to anchor conversation, not restrict it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "check", "clear"],
                "description": "The focus action to perform.",
            },
            "description": {
                "type": "string",
                "description": "Required for 'set': what the user is focused on (e.g. 'deep architecture review').",
            },
            "thread_id": {
                "type": "string",
                "description": "Optional: thread ID. Defaults to current topic.",
            },
        },
        "required": ["action"],
    },
}


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
    return (
        f"[FOCUS] Active focus ({source_label}): '{focus['description']}'\n"
        f"  Topic: {focus.get('topic', '(none)')}"
    )


def _handle_clear(service, thread_id: str) -> str:
    """Clear the current focus session."""
    focus = service.get_focus(thread_id)
    if not focus:
        return "[FOCUS] No active focus session to clear."

    service.clear_focus(thread_id)
    return f"[FOCUS] Focus session ended: '{focus['description']}'"

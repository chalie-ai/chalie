"""
Goal Skill — Manage persistent directional goals via the ACT loop.

Actions: create, list, update, progress, check_in
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def handle_goal(topic: str, params: dict) -> str:
    """
    Manage user goals.

    Actions:
    - create: Create a new goal
    - list: List active goals (optional status filter)
    - update: Change goal status
    - progress: Add a progress note
    - check_in: Report all active goals with days since last mentioned

    Args:
        topic: Current conversation topic
        params: Action parameters dict

    Returns:
        Formatted result string
    """
    action = params.get('action', 'list')

    try:
        from services.goal_service import GoalService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        service = GoalService(db)

        if action == 'create':
            return _handle_create(service, topic, params)
        elif action == 'list':
            return _handle_list(service, topic, params)
        elif action == 'update':
            return _handle_update(service, params)
        elif action == 'progress':
            return _handle_progress(service, params)
        elif action == 'check_in':
            return _handle_check_in(service)
        else:
            return f"[GOALS] Unknown action '{action}'. Use: create, list, update, progress, check_in"

    except Exception as e:
        logger.error(f"[GOAL SKILL] Error: {e}", exc_info=True)
        return f"[GOALS] Error: {e}"


def _handle_create(service, topic: str, params: dict) -> str:
    """Create a new goal."""
    title = params.get('title', '').strip()
    if not title:
        return "[GOALS] 'title' is required to create a goal."

    description = params.get('description', '')
    priority = int(params.get('priority', 5))
    source = params.get('source', 'explicit')  # Explicit when user requests

    goal_id = service.create_goal(
        title=title,
        description=description,
        priority=priority,
        source=source,
        related_topics=[topic] if topic else [],
    )

    # Touch the goal so last_mentioned is set
    service.touch_goal(goal_id, topic=topic)

    return f"[GOALS] Goal created (id={goal_id}): '{title}' (priority={priority})"


def _handle_list(service, topic: str, params: dict) -> str:
    """List active goals."""
    goals = service.get_active_goals(limit=10)
    if not goals:
        return "[GOALS] No active goals."

    lines = ["[GOALS] Active goals:"]
    for g in goals:
        last_mentioned = g.get('last_mentioned')
        if last_mentioned:
            if hasattr(last_mentioned, 'replace'):
                days_ago = (datetime.now(timezone.utc) - last_mentioned.replace(tzinfo=timezone.utc)).days
                last_str = f", last mentioned {days_ago}d ago"
            else:
                last_str = ""
        else:
            last_str = ", never mentioned"

        lines.append(
            f"  [{g['priority']}] {g['id']}: {g['title']} "
            f"({g['status']}{last_str})"
        )

    return "\n".join(lines)


def _handle_update(service, params: dict) -> str:
    """Update goal status."""
    goal_id = params.get('goal_id', '').strip()
    new_status = params.get('status', '').strip()
    note = params.get('note', '')

    if not goal_id or not new_status:
        return "[GOALS] 'goal_id' and 'status' are required to update a goal."

    valid_statuses = ('active', 'progressing', 'achieved', 'abandoned', 'dormant')
    if new_status not in valid_statuses:
        return f"[GOALS] Invalid status '{new_status}'. Use: {', '.join(valid_statuses)}"

    success = service.update_status(goal_id, new_status, note=note)
    if success:
        return f"[GOALS] Goal {goal_id} status updated to '{new_status}'."
    else:
        return f"[GOALS] Failed to update goal {goal_id} — invalid transition or goal not found."


def _handle_progress(service, params: dict) -> str:
    """Add a progress note to a goal."""
    goal_id = params.get('goal_id', '').strip()
    note = params.get('note', '').strip()

    if not goal_id or not note:
        return "[GOALS] 'goal_id' and 'note' are required to add progress."

    success = service.add_progress_note(goal_id, note)
    if success:
        return f"[GOALS] Progress note added to goal {goal_id}."
    else:
        return f"[GOALS] Failed to add progress note to goal {goal_id}."


def _handle_check_in(service) -> str:
    """Report all active goals with days since last mentioned."""
    goals = service.get_active_goals(limit=20)
    if not goals:
        return "[GOALS] No active goals to check in on."

    now = datetime.now(timezone.utc)
    lines = ["[GOALS] Check-in:"]

    for g in goals:
        last_mentioned = g.get('last_mentioned')
        if last_mentioned:
            try:
                if not last_mentioned.tzinfo:
                    last_mentioned = last_mentioned.replace(tzinfo=timezone.utc)
                days = (now - last_mentioned).days
                recency = f"{days}d since mentioned"
            except Exception:
                recency = "unknown"
        else:
            recency = "never mentioned"

        progress = g.get('progress_notes') or []
        last_progress = progress[-1]['note'][:60] if progress else "(no progress notes)"

        lines.append(
            f"  [{g['priority']}] {g['id']}: {g['title']} "
            f"({g['status']}, {recency})"
        )
        lines.append(f"    Last progress: {last_progress}")

    return "\n".join(lines)

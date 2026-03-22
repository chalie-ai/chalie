"""
Goals Skill — Natural language goal management via the ACT loop.

Actions: list, view, confirm, complete, dismiss, adjust, mute, unmute
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOALS SKILL]"


def handle_goals(topic: str, params: dict) -> str:
    """
    Manage user goals.

    Args:
        topic: Current conversation topic
        params: Action parameters dict with 'action' key

    Returns:
        Formatted result string
    """
    action = params.get('action', 'list')

    try:
        from services.goal_ecology_service import GoalEcologyService
        ecology = GoalEcologyService()

        if action == 'list':
            return _handle_list(ecology)
        elif action == 'view':
            return _handle_view(ecology, params.get('goal_id'))
        elif action == 'confirm':
            return _handle_confirm(ecology, params.get('goal_id'))
        elif action == 'complete':
            return _handle_complete(ecology, params.get('goal_id'))
        elif action == 'dismiss':
            return _handle_dismiss(ecology, params.get('goal_id'))
        elif action == 'adjust':
            return _handle_adjust(ecology, params.get('goal_id'), params)
        elif action == 'mute':
            return _handle_mute(ecology, params.get('goal_id'))
        elif action == 'unmute':
            return _handle_unmute(ecology, params.get('goal_id'))
        else:
            return f"[GOALS] Unknown action: {action}. Valid: list, view, confirm, complete, dismiss, adjust, mute, unmute"

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
        return f"[GOALS] Error: {e}"


def _handle_list(ecology) -> str:
    """List active goals sorted by salience."""
    goals = ecology.get_active_stack(limit=10)
    if not goals:
        return "[GOALS] No active goals. Goals emerge from conversation patterns or explicit statements."

    lines = [f"[GOALS] {len(goals)} active goal(s):\n"]
    for i, g in enumerate(goals, 1):
        sal = f"{g['salience']:.0%}"
        conf = f"{g['confidence']:.0%}"
        ev = g.get('evidence_count', 0)
        lines.append(
            f"{i}. [{g['type']}] {g['description'][:120]}\n"
            f"   salience={sal} | confidence={conf} | evidence={ev} | "
            f"timescale={g.get('timescale', 'unknown')} | status={g.get('status', 'unknown')}"
        )
    return '\n'.join(lines)


def _handle_view(ecology, goal_id: str) -> str:
    """View detailed goal information including evidence timeline."""
    if not goal_id:
        return "[GOALS] goal_id required for view action"

    goal = ecology.get_goal(goal_id)
    if not goal:
        return f"[GOALS] Goal {goal_id} not found"

    lines = [f"[GOALS] Goal Detail: {goal['description'][:200]}"]
    lines.append(f"  Type: {goal['type']} | Status: {goal['status']}")
    lines.append(f"  Salience: {goal['salience']:.0%} | Confidence: {goal['confidence']:.0%}")
    lines.append(f"  Urgency: {goal.get('urgency', 0):.0%} | Timescale: {goal.get('timescale', 'unknown')}")
    lines.append(f"  Evidence count: {goal.get('evidence_count', 0)}")

    if goal.get('strategy'):
        lines.append(f"  Strategy: {goal['strategy'][:200]}")

    if goal.get('created_at'):
        lines.append(f"  Created: {goal['created_at']}")

    # Evidence timeline
    try:
        from services.database_service import get_lightweight_db_service
        db = get_lightweight_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, content, source, created_at
                FROM goal_evidence
                WHERE goal_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            """, (goal_id,))
            evidence = cursor.fetchall()
            cursor.close()

        if evidence:
            lines.append(f"\n  Recent evidence ({len(evidence)} entries):")
            for e in evidence:
                lines.append(f"    - [{e[0]}] {e[1][:100]} ({e[3]})")
    except Exception:
        pass

    # Outcome feedback
    feedback = goal.get('outcome_feedback', [])
    if isinstance(feedback, str):
        try:
            feedback = json.loads(feedback)
        except Exception:
            feedback = []
    if feedback:
        lines.append(f"\n  Feedback history ({len(feedback)} entries):")
        for f in feedback[-5:]:  # Last 5
            lines.append(f"    - {f.get('response', 'unknown')} ({f.get('timestamp', '')})")

    return '\n'.join(lines)


def _handle_confirm(ecology, goal_id: str) -> str:
    """Confirm an inferred/emergent goal, elevating it to stated type."""
    if not goal_id:
        return "[GOALS] goal_id required for confirm action"
    result = ecology.confirm_goal(goal_id)
    if result:
        return f"[GOALS] Goal confirmed as stated: {result['description'][:120]} (confidence now {result['confidence']:.0%})"
    return f"[GOALS] Failed to confirm goal {goal_id}"


def _handle_complete(ecology, goal_id: str) -> str:
    """Mark a goal as completed."""
    if not goal_id:
        return "[GOALS] goal_id required for complete action"
    if ecology.complete_goal(goal_id):
        return f"[GOALS] Goal {goal_id} marked as completed."
    return f"[GOALS] Goal {goal_id} not found or already completed/decayed."


def _handle_dismiss(ecology, goal_id: str) -> str:
    """User-initiated dismissal of a goal."""
    if not goal_id:
        return "[GOALS] goal_id required for dismiss action"
    if ecology.dismiss_goal(goal_id):
        return f"[GOALS] Goal {goal_id} dismissed."
    return f"[GOALS] Goal {goal_id} not found or already completed/decayed."


def _handle_adjust(ecology, goal_id: str, params: dict) -> str:
    """Adjust urgency and/or timescale of a goal."""
    if not goal_id:
        return "[GOALS] goal_id required for adjust action"

    updates = {}
    if 'urgency' in params:
        updates['urgency'] = max(0.0, min(1.0, float(params['urgency'])))
    if 'timescale' in params:
        valid = ('immediate', 'short_term', 'medium_term', 'long_term')
        if params['timescale'] in valid:
            updates['timescale'] = params['timescale']
        else:
            return f"[GOALS] Invalid timescale. Valid: {', '.join(valid)}"

    if not updates:
        return "[GOALS] No valid adjustments provided. Use urgency (0.0-1.0) or timescale."

    if ecology.update_goal(goal_id, updates):
        return f"[GOALS] Goal {goal_id} adjusted: {updates}"
    return f"[GOALS] Goal {goal_id} not found."


def _handle_mute(ecology, goal_id: str) -> str:
    """Mute a goal to silence proactive actions while keeping it for context."""
    if not goal_id:
        return "[GOALS] goal_id required for mute action"
    if ecology.mute_goal(goal_id):
        return (
            f"[GOALS] Goal {goal_id} muted. It will remain active for context "
            f"but will not trigger proactive actions until unmuted."
        )
    return f"[GOALS] Goal {goal_id} not found."


def _handle_unmute(ecology, goal_id: str) -> str:
    """Unmute a previously muted goal, re-enabling proactive triggering."""
    if not goal_id:
        return "[GOALS] goal_id required for unmute action"
    if ecology.unmute_goal(goal_id):
        return f"[GOALS] Goal {goal_id} unmuted. Proactive actions re-enabled."
    return f"[GOALS] Goal {goal_id} not found."

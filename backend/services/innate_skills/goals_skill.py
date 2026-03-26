"""
Goals Skill — Natural language goal management via the ACT loop.

Actions: list, view, confirm, complete, dismiss, adjust, mute, unmute, narrate
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
        elif action == 'narrate':
            return _handle_narrate(ecology, params)
        elif action == 'cluster_confirm':
            return _handle_cluster_confirm(ecology, params)
        elif action == 'cluster_dismiss':
            return _handle_cluster_dismiss(ecology, params)
        else:
            return (
                f"[GOALS] Unknown action: {action}. Valid: list, view, confirm, "
                f"complete, dismiss, adjust, mute, unmute, narrate, "
                f"cluster_confirm, cluster_dismiss"
            )

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
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
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

    # Show parent goal
    if goal.get('lineage_parent_id'):
        try:
            parent = ecology.get_goal(goal['lineage_parent_id'])
            if parent:
                lines.append(f"\nParent Goal: [{parent['type']}] {parent['description']}")
                lines.append(f"  Salience: {parent['salience']:.0%}, Confidence: {parent['confidence']:.0%}")
        except Exception:
            pass

    # Show child goals
    try:
        children = ecology.get_children(goal_id)
        if children:
            lines.append(f"\nChild Goals ({len(children)}):")
            for child in children:
                lines.append(f"  - [{child['type']}] {child['description'][:80]} ({child['status']})")
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


def _handle_narrate(ecology, params: dict) -> str:
    """Generate a narrative synthesis of goal evolution via LLM."""
    goals = ecology.get_active_stack(limit=5)
    if not goals:
        return "[GOALS] No active goals to narrate."

    # Gather evidence timelines and lineage for each goal
    narratives_data = []
    for g in goals:
        evidence = _get_evidence_for_goal(g['id'], limit=10)
        parent_desc = None
        if g.get('lineage_parent_id'):
            parent = ecology.get_goal(g['lineage_parent_id'])
            if parent:
                parent_desc = parent.get('description', '')

        children = []
        try:
            children = ecology.get_children(g['id'])
        except Exception:
            pass

        narratives_data.append({
            'description': g['description'],
            'type': g['type'],
            'status': g['status'],
            'salience': g['salience'],
            'confidence': g['confidence'],
            'evidence_count': g.get('evidence_count', 0),
            'timescale': g.get('timescale', 'medium_term'),
            'created_at': g.get('created_at', ''),
            'strategy': g.get('strategy', ''),
            'parent_goal': parent_desc,
            'child_count': len(children),
            'evidence_timeline': evidence,
            'outcome_feedback': g.get('outcome_feedback', []),
        })

    return _synthesize_narrative(narratives_data)


def _get_evidence_for_goal(goal_id: str, limit: int = 10) -> list:
    """Fetch recent evidence for a goal."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, content, source, created_at
                FROM goal_evidence
                WHERE goal_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (goal_id, limit))
            rows = cursor.fetchall()
            cursor.close()
        return [{'signal_type': r[0], 'content': r[1], 'source': r[2], 'timestamp': r[3]} for r in rows]
    except Exception:
        return []


def _synthesize_narrative(goals_data: list) -> str:
    """Generate a narrative synthesis via lightweight LLM call."""
    try:
        from services.background_llm_queue import create_background_llm_proxy
        proxy = create_background_llm_proxy("goal-strategy")

        system_prompt = (
            "You are synthesizing the story of this user's goals. Write in second person "
            '("you\'ve been..."). Be warm but precise. Highlight:\n'
            "- How goals formed (from explicit statements vs emerged from patterns)\n"
            "- How they've evolved over time (evidence accumulation, confidence changes)\n"
            "- Connections between goals (parent-child hierarchies, shared themes)\n"
            "- What the system has done or plans to do about them (strategies, proactive actions)\n"
            "Keep it to 2-4 paragraphs. This is a narrative, not a list.\n"
            "Do not use corporate language. Speak as someone who has genuinely been thinking about this.\n"
            "Anchor all statements in evidence — reference specific counts and patterns, not speculation."
        )

        # Build context
        lines = []
        for i, g in enumerate(goals_data, 1):
            lines.append(f"Goal {i}: {g['description']}")
            lines.append(f"  Type: {g['type']} | Status: {g['status']} | Timescale: {g['timescale']}")
            lines.append(f"  Salience: {g['salience']:.0%} | Confidence: {g['confidence']:.0%}")
            lines.append(f"  Evidence count: {g['evidence_count']} | Created: {g['created_at']}")
            if g.get('strategy'):
                lines.append(f"  Strategy: {g['strategy'][:150]}")
            if g.get('parent_goal'):
                lines.append(f"  Serves larger goal: {g['parent_goal']}")
            if g['child_count'] > 0:
                lines.append(f"  Has {g['child_count']} sub-goal(s)")

            if g['evidence_timeline']:
                lines.append(f"  Evidence timeline ({len(g['evidence_timeline'])} recent):")
                for e in g['evidence_timeline'][:5]:
                    lines.append(f"    - [{e['signal_type']}] {e['content'][:80]} ({e['timestamp']})")

            feedback = g.get('outcome_feedback', [])
            if isinstance(feedback, str):
                try:
                    import json as _json
                    feedback = _json.loads(feedback)
                except Exception:
                    feedback = []
            if feedback:
                engaged = sum(1 for f in feedback if f.get('response') == 'engaged')
                rejected = sum(1 for f in feedback if f.get('response') == 'rejected')
                lines.append(f"  Feedback: {engaged} engaged, {rejected} rejected")
            lines.append("")

        user_message = "\n".join(lines)

        response = proxy.send_message(system_prompt, user_message)
        if not response or not response.text.strip():
            return "[GOALS] Unable to generate narrative at this time."

        narrative = response.text.strip()
        # Cap at ~2000 chars to keep response reasonable
        if len(narrative) > 2000:
            narrative = narrative[:2000] + "..."

        return f"[GOALS NARRATIVE]\n{narrative}"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Narrative synthesis failed: {e}")
        return "[GOALS] Unable to generate narrative at this time. Try the 'list' action for a structured view."


def _handle_cluster_confirm(ecology, params: dict) -> str:
    """Confirm a detected goal cluster -- create parent goal and link children."""
    goal_ids = params.get('goal_ids', [])
    description = params.get('description', '')

    if not goal_ids or len(goal_ids) < 2:
        return "[GOALS] cluster_confirm requires goal_ids (list of 2+ IDs) and description"
    if not description:
        return "[GOALS] cluster_confirm requires a description for the parent goal"

    try:
        # Determine parent timescale: one level above the highest child
        child_timescales = []
        for gid in goal_ids:
            g = ecology.get_goal(gid)
            if g:
                child_timescales.append(g.get('timescale', 'medium_term'))

        highest = max(
            (ecology.TIMESCALE_ORDER.get(ts, 1) for ts in child_timescales),
            default=1
        )
        reverse_order = {v: k for k, v in ecology.TIMESCALE_ORDER.items()}
        parent_timescale = reverse_order.get(min(highest + 1, 3), 'long_term')

        # Create parent goal
        parent = ecology.create_goal(
            description=description,
            type='stated',
            timescale=parent_timescale,
        )

        if not parent:
            return "[GOALS] Failed to create parent goal"

        # Link children via lineage_parent_id
        from services.time_utils import utc_now
        now = utc_now().isoformat()
        linked = 0
        with ecology.db.connection() as conn:
            cursor = conn.cursor()
            for gid in goal_ids:
                cursor.execute("""
                    UPDATE goals SET lineage_parent_id = ?, updated_at = ?
                    WHERE id = ? AND status NOT IN ('completed', 'decayed')
                """, (parent['id'], now, gid))
                linked += cursor.rowcount
            cursor.close()

        return (
            f"[GOALS] Cluster confirmed! Created parent goal: {description[:100]}\n"
            f"  Linked {linked} child goal(s). Parent timescale: {parent_timescale}."
        )

    except Exception as e:
        logger.error(f"{LOG_PREFIX} cluster_confirm failed: {e}")
        return f"[GOALS] Failed to confirm cluster: {e}"


def _handle_cluster_dismiss(ecology, params: dict) -> str:
    """Dismiss a detected cluster -- set cooldown to prevent re-detection."""
    goal_ids = params.get('goal_ids', [])

    if not goal_ids or len(goal_ids) < 2:
        return "[GOALS] cluster_dismiss requires goal_ids (list of 2+ IDs)"

    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()

        cluster_ids = sorted(goal_ids[:5])
        cluster_key = f"goal_cluster:{'|'.join(cluster_ids)}"

        # Set 30-day cooldown
        store.setex(f"{cluster_key}:cooldown", 86400 * 30, "dismissed")

        return "[GOALS] Cluster dismissed. Will not suggest this grouping again for 30 days."

    except Exception as e:
        logger.error(f"{LOG_PREFIX} cluster_dismiss failed: {e}")
        return f"[GOALS] Failed to dismiss cluster: {e}"

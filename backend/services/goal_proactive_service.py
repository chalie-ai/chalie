"""
Goal Proactive Service — Trigger engine for actionable goals.

When a goal is actionable (salience >= threshold, strategy exists), this service
decides whether and how to act on it. Replaces the autonomous action system.

Proactive output styles (based on confidence):
  confidence < 0.7  → Ask: proactive push asking if the user wants help
  confidence 0.7-0.85 → Suggest: proactive push with a concrete suggestion
  confidence > 0.85 → Act: create a persistent task that runs through the ACT loop
    with full tool access (search, read, find_tools, etc.), then surfaces results

The 'act' style is the key differentiator from the old system. Instead of sending a
canned message, it creates a real task that uses tools to research, analyze, and prepare
actionable output — then delivers results proactively.

Social cost checks prevent interrupting deep focus, quiet hours, or recently ignored attempts.
"""

import logging
from typing import Dict, Any, Optional

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL PROACTIVE]"

# Social cost thresholds
MAX_SOCIAL_COST = 0.4
MIN_CONFIDENCE_FOR_ACTION = 0.6

# Output style thresholds
ASK_THRESHOLD = 0.7
SUGGEST_THRESHOLD = 0.85


def check_and_execute(goal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check initiation conditions and execute proactive action for a goal.

    Args:
        goal: Goal dict with id, description, confidence, salience, strategy, etc.

    Returns:
        Result dict or None if conditions not met.
    """
    goal_id = goal.get('id', 'unknown')
    confidence = goal.get('confidence', 0.0)
    description = goal.get('description', '')

    # 1. Confidence check
    if confidence < MIN_CONFIDENCE_FOR_ACTION:
        logger.debug(
            f"{LOG_PREFIX} Goal {goal_id[:8]} confidence too low "
            f"({confidence:.2f} < {MIN_CONFIDENCE_FOR_ACTION})"
        )
        return None

    # 2. Social cost check
    social_cost = _calculate_social_cost()
    if social_cost > MAX_SOCIAL_COST:
        logger.debug(
            f"{LOG_PREFIX} Social cost too high ({social_cost:.2f}) for "
            f"goal {goal_id[:8]}"
        )
        return None

    # 3. Choose output style
    style = _choose_output_style(confidence)

    # 4. Execute
    try:
        result = _execute_proactive(goal, style)

        # 5. Record outcome (will be updated when user responds)
        _mark_acted(goal_id)

        logger.info(
            f"{LOG_PREFIX} Executed proactive action for goal {goal_id[:8]}: "
            f"style={style}, description='{description[:60]}'"
        )

        return result

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Proactive execution failed for {goal_id[:8]}: {e}")
        return None


def _calculate_social_cost() -> float:
    """
    Calculate social cost of proactive action.

    Factors: deep focus, recent ignored attempts, quiet hours.
    Returns 0.0 (no cost) to 1.0 (high cost).
    """
    cost = 0.0

    # Deep focus check
    try:
        from services.ambient_inference_service import AmbientInferenceService
        inference = AmbientInferenceService()
        if inference.is_user_deep_focus():
            cost += 0.5
    except Exception:
        pass

    # Recent ignored attempts
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        recent_ignored = store.get('goal:recent_ignored_count')
        if recent_ignored and int(recent_ignored) >= 2:
            cost += 0.3
    except Exception:
        pass

    # Message momentum: active conversation = higher cost
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        last_msg_ts = store.get('last_user_message_ts')
        if last_msg_ts:
            from services.time_utils import parse_utc, utc_now
            last = parse_utc(last_msg_ts)
            gap_seconds = (utc_now() - last).total_seconds()

            # Active conversation (message within last 2 minutes) — don't interrupt
            if gap_seconds < 120:
                cost += 0.3

            # Re-entry after gap (2-8 hours) — good time to suggest
            elif 7200 < gap_seconds < 28800:
                cost -= 0.15
    except Exception:
        pass

    return min(1.0, cost)


def _choose_output_style(confidence: float) -> str:
    """Choose proactive output style based on goal confidence."""
    if confidence >= SUGGEST_THRESHOLD:
        return 'act'
    elif confidence >= ASK_THRESHOLD:
        return 'suggest'
    else:
        return 'ask'


def _execute_proactive(goal: Dict[str, Any], style: str) -> Dict[str, Any]:
    """
    Execute proactive action for a goal.

    - 'ask' / 'suggest': Push a proactive message to the user via prompt queue.
    - 'act': Create a persistent task that runs through the full ACT loop with
      tool access (search, read, find_tools, etc.). The persistent task worker
      picks it up, decomposes it into steps, executes with tools, and surfaces
      the results when done.
    """
    if style == 'act':
        return _execute_via_persistent_task(goal)
    else:
        return _execute_via_proactive_push(goal, style)


def _execute_via_persistent_task(goal: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-confidence path: create a persistent task backed by the ACT loop.

    The persistent task worker runs the goal through the full ACT orchestrator
    with access to all innate skills (recall, search, read, find_tools, etc.).
    This is how the system autonomously uses tools — e.g., searching for tech news,
    preparing competitive reports, researching neighborhoods.

    The task is created with scope derived from the goal's strategy and evidence,
    giving the ACT loop concrete direction rather than a vague objective.
    """
    goal_id = goal.get('id', 'unknown')
    description = goal.get('description', '')
    strategy = goal.get('strategy', '')

    # Build a scoped task goal that gives the ACT loop direction
    task_goal = description
    if strategy:
        task_goal = f"{description}. Approach: {strategy}"

    try:
        from services.database_service import get_shared_db_service
        from services.persistent_task_service import PersistentTaskService

        db = get_shared_db_service()
        task_service = PersistentTaskService(db)

        # Resolve account ID
        account_id = 1
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM master_account LIMIT 1")
                row = cursor.fetchone()
                cursor.close()
                if row:
                    account_id = row[0]
        except Exception:
            pass

        # Check for duplicate tasks (don't re-create for the same goal)
        existing = task_service.find_duplicate(task_goal)
        if existing:
            logger.info(
                f"{LOG_PREFIX} Task already exists for goal {goal_id[:8]} "
                f"(task {existing['id']})"
            )
            return {
                'action': 'task_exists',
                'style': 'act',
                'goal_id': goal_id,
                'task_id': existing['id'],
            }

        task = task_service.create_task(
            account_id=account_id,
            goal=task_goal,
            scope=f"Goal-driven proactive task from goal ecology (goal_id={goal_id})",
            priority=4,  # Slightly above default — goal-driven tasks matter
        )

        # Transition to accepted so the worker picks it up
        task_service.transition(task['id'], 'accepted')

        logger.info(
            f"{LOG_PREFIX} Created persistent task {task['id']} for goal "
            f"{goal_id[:8]}: '{task_goal[:80]}'"
        )

        return {
            'action': 'persistent_task_created',
            'style': 'act',
            'goal_id': goal_id,
            'task_id': task['id'],
            'task_goal': task_goal[:200],
        }

    except Exception as e:
        logger.warning(
            f"{LOG_PREFIX} Persistent task creation failed for goal "
            f"{goal_id[:8]}, falling back to proactive push: {e}"
        )
        # Fallback: if task creation fails, at least push a message
        return _execute_via_proactive_push(goal, 'suggest')


def _execute_via_proactive_push(goal: Dict[str, Any], style: str) -> Dict[str, Any]:
    """
    Low/medium confidence path: push a proactive message to the user.

    'ask' style checks if the user wants help. 'suggest' style offers a
    concrete suggestion. Both route through the prompt queue so the digest
    worker generates a natural response.
    """
    goal_id = goal.get('id', 'unknown')
    description = goal.get('description', '')

    if style == 'ask':
        prompt = (
            f"[PROACTIVE GOAL CHECK-IN]\n"
            f"Goal: {description}\n"
            f"Evidence count: {goal.get('evidence_count', 0)}\n"
            f"Goal type: {goal.get('type', 'emergent')}\n"
            f"Confidence: {goal.get('confidence', 0):.0%}\n\n"
            f"You've accumulated evidence that the user cares about this. "
            f"Check in naturally — ask if they'd like help, referencing "
            f"specific evidence if available. Be warm, not corporate."
        )
    else:  # suggest
        strategy = goal.get('strategy', '')
        prompt = (
            f"[PROACTIVE GOAL SUGGESTION]\n"
            f"Goal: {description}\n"
            f"Strategy: {strategy}\n"
            f"Evidence count: {goal.get('evidence_count', 0)}\n"
            f"Goal type: {goal.get('type', 'emergent')}\n"
            f"Confidence: {goal.get('confidence', 0):.0%}\n\n"
            f"You're confident enough to suggest concrete help. "
            f"Offer a specific next step based on the strategy. "
            f"Be direct and personal, not formulaic."
        )

    try:
        from services.memory_client import MemoryClientService
        import json

        store = MemoryClientService.create_connection()
        store.rpush('prompt-queue', json.dumps({
            'prompt': prompt,
            'metadata': {
                'type': 'proactive_drift',
                'source': 'goal_ecology',
                'goal_id': goal_id,
                'style': style,
                'topic': 'proactive',
            },
        }))

        return {
            'action': 'proactive_push',
            'style': style,
            'goal_id': goal_id,
            'prompt': prompt[:200],
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Proactive push failed: {e}")
        raise


def _mark_acted(goal_id: str) -> None:
    """Mark a goal as having been acted upon."""
    try:
        now = utc_now().isoformat()

        from services.database_service import get_lightweight_db_service
        db = get_lightweight_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE goals SET last_acted_at = ?, status = 'active', updated_at = ?
                WHERE id = ?
            """, (now, now, goal_id))
            cursor.close()

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Failed to mark goal as acted: {e}")

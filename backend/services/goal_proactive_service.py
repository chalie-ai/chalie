"""
Goal Proactive Service — Trigger engine for actionable goals.

When a goal is actionable (salience >= threshold, strategy exists), this service
decides whether and how to act on it. Replaces the autonomous action system.

Proactive output styles (based on confidence):
  confidence < 0.7  → Ask: proactive push asking if the user wants help
  confidence 0.7-0.85 → Suggest: proactive push with a concrete suggestion
  confidence > 0.85 → Act: launch a goal pursuit that runs through the ACT loop
    with full tool access (search, read, find_tools, etc.), then surfaces results

The 'act' style is the key differentiator from the old system. Instead of sending a
canned message, it creates a real task that uses tools to research, analyze, and prepare
actionable output — then delivers results proactively.

Social cost checks prevent interrupting deep focus, quiet hours, or recently ignored attempts.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL PROACTIVE]"

# Social cost thresholds
MAX_SOCIAL_COST = 0.6
MIN_CONFIDENCE_FOR_ACTION = 0.6

# Output style thresholds
ASK_THRESHOLD = 0.7
SUGGEST_THRESHOLD = 0.85


def _get_user_local_hour() -> int:
    """Return the current hour (0-23) in the user's local timezone.

    Reads the IANA timezone from ClientContextService and derives the local
    hour via get_timezone_offset(). Falls back to the UTC hour if no timezone
    is available or any error occurs.
    """
    try:
        from services.client_context_service import ClientContextService
        ctx_service = ClientContextService()
        offset_minutes = ctx_service.get_timezone_offset()
        if offset_minutes is not None:
            from datetime import timedelta
            local = utc_now() + timedelta(minutes=offset_minutes)
            return local.hour
    except Exception:
        pass
    return utc_now().hour


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

        # 6. Emit telemetry for the proactive trigger decision
        try:
            from services.telemetry_service import get_telemetry_collector, PROACTIVE_TRIGGER
            get_telemetry_collector().record(PROACTIVE_TRIGGER, {
                "confidence": confidence,
                "social_cost": social_cost,
                "action_taken": style,
                "goal_id": goal_id,
            })
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Telemetry emit for proactive trigger failed: {e}")

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
            cost += 0.3
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Deep focus check failed: {e}")

    # Recent ignored attempts
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        recent_ignored = store.get('goal:recent_ignored_count')
        if recent_ignored and int(recent_ignored) >= 2:
            cost += 0.3
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Recent ignored attempts check failed: {e}")

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
                cost += 0.2

            # Re-entry after gap (2-8 hours) — good time to suggest
            elif 7200 < gap_seconds < 28800:
                cost -= 0.15
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Message momentum check failed: {e}")

    # Quiet hours: learned from interaction log
    try:
        cost += _get_quiet_hours_cost()
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Quiet hours cost check failed: {e}")

    # Rapid conversation pace: multiple messages in quick succession
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        recent_count = store.get('recent_message_count_5min')
        if recent_count and int(recent_count) >= 3:
            cost += 0.25  # User is in active conversation, don't interrupt
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Rapid conversation pace check failed: {e}")

    return min(1.0, cost)


def _get_quiet_hours_cost() -> float:
    """
    Check if current hour is a quiet hour based on learned interaction patterns.

    Queries interaction_log to find hours with < 5% of total interactions.
    Caches result in MemoryStore for 24 hours.

    Returns:
        0.4 if current hour is a quiet hour, 0.0 otherwise.
    """
    from services.memory_client import MemoryClientService

    store = MemoryClientService.create_connection()

    # Check cache first
    cached = store.get('goal:quiet_hours')
    if cached:
        quiet_hours = json.loads(cached)
    else:
        quiet_hours = _learn_quiet_hours()
        if quiet_hours is not None:
            store.setex('goal:quiet_hours', 86400, json.dumps(quiet_hours))
        else:
            return 0.0

    current_hour = _get_user_local_hour()
    if current_hour in quiet_hours:
        return 0.4

    return 0.0


def _learn_quiet_hours() -> Optional[List[int]]:
    """
    Learn quiet hours from interaction_log timestamps.

    Bucket interaction timestamps by user-local hour (via stored IANA timezone).
    Falls back to UTC if no timezone is available. Hours with < 5% of total
    interactions are considered quiet hours.

    Returns:
        List of quiet hour integers (0-23) or None if insufficient data.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        # Determine the user's UTC offset so we can shift SQLite timestamps.
        offset_minutes = 0
        try:
            from services.client_context_service import ClientContextService
            offset = ClientContextService().get_timezone_offset()
            if offset is not None:
                offset_minutes = offset
        except Exception:
            pass

        # Build a SQLite datetime modifier that shifts UTC timestamps to local.
        # E.g. UTC+2 → '+120 minutes'; UTC-5 → '-300 minutes'.
        sign = '+' if offset_minutes >= 0 else '-'
        tz_modifier = f"'{sign}{abs(offset_minutes)} minutes'"

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT CAST(strftime('%H', datetime(created_at, {tz_modifier})) AS INTEGER) as hour,
                       COUNT(*) as count
                FROM interaction_log
                WHERE created_at > datetime('now', '-30 days')
                GROUP BY hour
            """)
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return None

        total = sum(r[1] for r in rows)
        if total < 50:  # Not enough data to learn from
            return None

        threshold = total * 0.05  # 5% of total
        quiet = [r[0] for r in rows if r[1] < threshold]

        # Also mark hours with zero interactions as quiet
        active_hours = {r[0] for r in rows}
        for h in range(24):
            if h not in active_hours:
                quiet.append(h)

        return sorted(set(quiet))

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Quiet hours learning failed: {e}")
        return None


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
    - 'act': Launch a goal pursuit that runs through the full ACT loop with
      tool access (search, read, find_tools, etc.). GoalPursuitProcessor
      runs in a daemon thread, executes with tools, and surfaces the results
      when done.
    """
    if style == 'act':
        return _execute_via_goal_pursuit(goal)
    else:
        return _execute_via_proactive_push(goal, style)


def _execute_via_goal_pursuit(goal: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-confidence path: spawn a GoalPursuitProcessor in a daemon thread.

    Runs the goal through GoalPursuitProcessor with access to all innate skills.
    When complete, OutputService.enqueue_proactive() surfaces the result.
    """
    import threading
    import uuid

    goal_id = goal.get('id', 'unknown')
    description = goal.get('description', '')
    strategy = goal.get('strategy', '')

    task_goal = description
    if strategy:
        task_goal = f"{description}. Approach: {strategy}"

    pursuit_id = uuid.uuid4().hex[:12]

    def _run():
        try:
            from services.goal_pursuit_processor import GoalPursuitProcessor
            from services.output_service import OutputService
            response_text = GoalPursuitProcessor(
                raw_input=task_goal,
                metadata={'pursuit_id': pursuit_id},
            ).send()
            response_text = (response_text or '').strip()
            if not response_text:
                response_text = "Goal pursuit completed but produced no output."
            OutputService().enqueue_proactive(
                topic='user',
                response=response_text,
                source='goal_pursuit',
            )
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Goal pursuit failed for goal {goal_id[:8]}: {e}", exc_info=True)
            try:
                from services.output_service import OutputService
                OutputService().enqueue_proactive(
                    topic='user',
                    response=f"Goal pursuit failed: {e}",
                    source='goal_pursuit',
                )
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name=f"goal-pursuit-{pursuit_id}")
    t.start()

    logger.info(
        f"{LOG_PREFIX} Spawned goal pursuit {pursuit_id} for goal "
        f"{goal_id[:8]}: '{task_goal[:80]}'"
    )

    return {
        'action': 'goal_pursuit_spawned',
        'style': 'act',
        'goal_id': goal_id,
        'pursuit_id': pursuit_id,
        'task_goal': task_goal[:200],
    }


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
                'type': 'proactive',
                'source': 'goal_ecology',
                'goal_id': goal_id,
                'style': style,
                'topic': 'proactive',
            },
        }))
        store.expire('prompt-queue', 3600)

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

        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE goals SET last_acted_at = ?, status = 'active', updated_at = ?
                WHERE id = ?
            """, (now, now, goal_id))
            cursor.close()

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Failed to mark goal as acted: {e}")

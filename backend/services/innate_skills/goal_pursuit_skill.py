"""
Goal Pursuit Skill — Launch a background GoalPursuitProcessor for long-running tasks.

Spawns a daemon thread running GoalPursuitProcessor(raw_input=..., metadata=...).send()
and returns immediately. When the loop completes, OutputService.enqueue_proactive()
pushes the result to the user's WebSocket.
"""

import json
import logging
import threading
import uuid

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL PURSUIT SKILL]"

TOOL_SCHEMA = {
    "name": "goal_pursuit",
    "description": (
        "Launch a background copy of yourself for deep, long-running research or multi-source investigation. "
        "Fire this tool FIRST (before answering inline) when the user asks for: deep research, comprehensive analysis, "
        "multi-source comparison, deep dive, extended investigation, or anything requiring 2+ minutes of work across many sources. "
        "Signal phrases: 'deep dive', 'research', 'analyze multiple', 'compare X and Y', 'comprehensive study', "
        "'competitive analysis', 'find out about', 'investigate thoroughly'. "
        "When unsure if a request is heavy enough, prefer this tool — surface-level answers to deep-research asks waste the user's time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Describe the goal you want to achieve. Be detailed and specific. "
                    "You can use existing tools to prepare data you think will benefit you "
                    "when running the long-running task. If the task is big enough, you should "
                    "supply a high-level plan as the goal parameter."
                ),
            },
        },
        "required": ["goal"],
    },
}


def handle_goal_pursuit(channel: str, params: dict) -> str:
    """Spawn a background GoalPursuitProcessor and return immediately.

    Before spawning the thread, a goal row is persisted to the goals table
    via GoalEcologyService so downstream scenario queries can confirm the
    intent was recorded.  Persistence failure is logged but never blocks the
    pursuit — the background worker spawns regardless.

    Args:
        channel: Current conversation channel (unused — pursuit runs in its own channel).
        params: Dict with 'goal' key.

    Returns:
        JSON string confirming the pursuit was launched.
    """
    goal = params.get("goal", "").strip()
    if not goal:
        return json.dumps({"success": False, "error": "No goal specified."})

    pursuit_id = uuid.uuid4().hex

    # Persist the goal to the goals table before spawning the worker so the
    # intent is recorded even if the background thread is slow or fails.
    goal_id = None
    try:
        from services.goal_ecology_service import GoalEcologyService
        ecology = GoalEcologyService()
        goal_row = ecology.create_goal(description=goal, type='stated')
        if goal_row:
            goal_id = goal_row.get('id')
            logger.info(f"{LOG_PREFIX} Persisted goal {goal_id[:8] if goal_id else '?'} for pursuit {pursuit_id[:8]}")
        else:
            logger.warning(f"{LOG_PREFIX} create_goal returned falsy for pursuit {pursuit_id[:8]}; thread will still spawn")
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to persist goal for pursuit {pursuit_id[:8]}: {e}", exc_info=True)

    def _run():
        try:
            from services.goal_pursuit_processor import GoalPursuitProcessor
            from services.output_service import OutputService

            response_text = GoalPursuitProcessor(
                raw_input=goal,
                metadata={'pursuit_id': pursuit_id, 'goal_id': goal_id},
            ).send()
            response_text = (response_text or '').strip()
            if not response_text:
                response_text = "Goal pursuit completed but produced no output."

            OutputService().enqueue_proactive(
                topic='user',
                response=response_text,
                source='goal_pursuit',
            )
            logger.info(f"{LOG_PREFIX} Pursuit {pursuit_id[:8]} complete")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Pursuit {pursuit_id[:8]} failed: {e}", exc_info=True)
            try:
                from services.output_service import OutputService
                OutputService().enqueue_proactive(
                    topic='user',
                    response=f"Goal pursuit failed: {e}",
                    source='goal_pursuit',
                )
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name=f"goal-pursuit-{pursuit_id[:8]}")
    t.start()

    return json.dumps({
        "success": True,
        "pursuit_id": pursuit_id,
        "response": "Working on it. I'll notify you when done.",
    })

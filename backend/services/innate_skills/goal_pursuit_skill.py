"""
Goal Pursuit Skill — Launch a background GoalPursuitProcessor for long-running tasks.

Spawns a daemon thread running GoalPursuitProcessor().process() and returns
immediately. When the loop completes, OutputService.enqueue_proactive() pushes
the result to the user's WebSocket.
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
        "Use this tool to launch a separate instance of yourself with unconstrained limits "
        "to research or perform long running tasks. Only use this if you expect a request to "
        "take longer than 2 minutes to complete. If not, use other tools to resolve the request."
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

    def _run():
        try:
            from services.goal_pursuit_processor import GoalPursuitProcessor
            from services.output_service import OutputService

            response_text = GoalPursuitProcessor(
                raw_input=goal,
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

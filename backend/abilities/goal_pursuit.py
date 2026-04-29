"""
GoalPursuitAbility — Launch a background GoalPursuitProcessor for long-running tasks.

Spawns a daemon thread running GoalPursuitProcessor(raw_input=..., metadata=...).send()
and returns immediately. When the loop completes, OutputService.enqueue_proactive()
pushes the result to the user's WebSocket.
"""

import json
import logging
import threading
import uuid

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL PURSUIT SKILL]"


class GoalPursuitAbility(Ability):
    NAME = "goal_pursuit"
    SUMMARY = "Spawn a background research agent for deep, multi-source investigation that would take more than two minutes."
    EXAMPLES = [
        "research the top 3 health benefits of cold water swimming as a background task",
        "do a deep dive on the competitive landscape for electric vehicles",
        "investigate thoroughly how LLM context windows affect reasoning quality",
        "comprehensive analysis of the housing market in Malta over the last 5 years",
        "research multiple sources on the best diet for endurance athletes",
        "background task: compare React vs Vue for a large enterprise project",
        "deep research on the history and current state of quantum computing",
        "find out everything about carbon capture technology and write me a brief",
    ]
    INPUT_SCHEMA = {
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
    }
    TIMEOUT = 10

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        goal = params.get("goal", "").strip()
        if not goal:
            return {"text": _skill_tag("goal_pursuit", error="no-goal-specified")}

        pursuit_id = uuid.uuid4().hex

        def _run():
            try:
                from services.goal_pursuit_processor import GoalPursuitProcessor
                from services.output_service import OutputService

                response_text = GoalPursuitProcessor(
                    raw_input=goal,
                    metadata={"pursuit_id": pursuit_id},
                ).send()
                response_text = (response_text or "").strip()
                if not response_text:
                    response_text = "Goal pursuit completed but produced no output."

                OutputService().enqueue_proactive(
                    topic="user",
                    response=response_text,
                    source="goal_pursuit",
                )
                logger.info(f"{LOG_PREFIX} Pursuit {pursuit_id[:8]} complete")
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Pursuit {pursuit_id[:8]} failed: {e}", exc_info=True)
                try:
                    from services.output_service import OutputService
                    OutputService().enqueue_proactive(
                        topic="user",
                        response=f"Goal pursuit failed: {e}",
                        source="goal_pursuit",
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True, name=f"goal-pursuit-{pursuit_id[:8]}")
        t.start()

        body = json.dumps({
            "success": True,
            "pursuit_id": pursuit_id,
            "response": "Working on it. I'll notify you when done.",
        })
        return {"text": _skill_tag("goal_pursuit", body, pursuit_id=pursuit_id[:8])}

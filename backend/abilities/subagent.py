"""
SubagentAbility — Spawn a focused subagent to execute a delegated task.

Default mode is fire-and-forget (async). The result is pushed back via
OutputService.enqueue_proactive(source='subagent') as a 'task' SSE event.
When wait=true the parent ACT iteration blocks until the subagent finishes
(capped at 300 s when wait=true, up to 7200 s when wait=false).
"""

import json
import logging
import threading
import time
import uuid

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SUBAGENT SKILL]"


class SubagentAbility(Ability):
    NAME = "subagent"
    SUMMARY = (
        "A subagent is initialised to perform long running tasks or tasks that "
        "require large data traversal. Use this tool to be able to parallelise "
        "work or perform actions in the background or to minimise context "
        "bloat. If a task requires you to read a large document, use this tool "
        "to have an agent summarise it for you. If you need to search multiple "
        "pages, use this tool to keep your context window clean, etc... You "
        "can launch as many subagents as you need. Be descriptive and launch "
        "bounded tasks to answer to the user more efficiently."
    )
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
            "prompt": {
                "type": "string",
                "description": "Detailed task. Be specific. Include context, success criteria, and any pre-fetched data the subagent needs.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max runtime in seconds. Default 1800 (30m). Min 180 (3m). Max 7200 (2h) when wait=false; capped at 300 (5m) when wait=true.",
                "default": 1800,
                "minimum": 180,
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "Set this to true to make the request synchronous. "
                    "Request will be hard capped to 5 minutes, else leave it "
                    "False and you will be notified automatically once the "
                    "request is completed."
                ),
                "default": False,
            },
        },
        "required": ["prompt"],
    }
    TIMEOUT = 10  # parent dispatch timeout; per-call timeout flows to SubagentProcessor

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"text": _skill_tag("subagent", error="prompt-required")}

        timeout = params.get("timeout", 1800)
        wait = params.get("wait", False)

        # Validate timeout bounds
        if timeout < 180:
            return {"text": _skill_tag("subagent", error="timeout-below-min-180s")}
        if wait and timeout > 300:
            return {"text": _skill_tag("subagent", error="wait-true-requires-timeout-le-300s")}
        if timeout > 7200:
            logger.debug("%s timeout %ds exceeds ceiling — clamped to 7200s", LOG_PREFIX, timeout)
            timeout = 7200

        sub_id = uuid.uuid4().hex

        if wait:
            return self._run_sync(prompt, timeout, sub_id)
        return self._run_async(prompt, timeout, sub_id)

    # ── Execution paths ───────────────────────────────────────────────────────

    def _run_async(self, prompt: str, timeout: int, sub_id: str) -> dict:
        """Fire-and-forget: spawn daemon thread, return immediately."""

        def _run():
            try:
                from services.subagent_processor import SubagentProcessor
                from services.output_service import OutputService

                response_text = SubagentProcessor(
                    raw_input=prompt,
                    metadata={"sub_id": sub_id},
                    max_timeout_override=timeout,
                ).send()
                response_text = (response_text or "").strip()
                if not response_text:
                    response_text = "Subagent completed but produced no output."

                OutputService().enqueue_proactive(
                    topic="user",
                    response=response_text,
                    source="subagent",
                )
                logger.info("%s Subagent %s complete", LOG_PREFIX, sub_id[:8])
            except Exception as e:
                logger.error(
                    "%s Subagent %s failed: %s", LOG_PREFIX, sub_id[:8], e, exc_info=True
                )
                try:
                    from services.output_service import OutputService
                    OutputService().enqueue_proactive(
                        topic="user",
                        response=f"Subagent failed: {e}",
                        source="subagent",
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True, name=f"subagent-{sub_id[:8]}")
        t.start()

        body = json.dumps({
            "success": True,
            "sub_id": sub_id,
            "response": "Working on it. I'll notify you when done.",
        })
        return {"text": _skill_tag("subagent", body, sub_id=sub_id[:8], wait=False)}

    def _run_sync(self, prompt: str, timeout: int, sub_id: str) -> dict:
        """Blocking: run SubagentProcessor inline and return its result."""
        from services.subagent_processor import SubagentProcessor
        from services.message_processor import current_processor

        t_start = time.monotonic()
        try:
            response_text = SubagentProcessor(
                raw_input=prompt,
                metadata={"sub_id": sub_id},
                max_timeout_override=timeout,
            ).send()
            response_text = (response_text or "").strip()
        except Exception as exc:
            return {
                "text": _skill_tag(
                    "subagent", sub_id=sub_id[:8], wait=True, error=str(exc)
                )
            }

        elapsed = time.monotonic() - t_start

        # Advance the parent processor's loop_start so the elapsed wait time
        # does not count against the parent's MAX_TIMEOUT budget.
        parent = current_processor()
        if parent is not None and hasattr(parent, "_loop_start") and parent._loop_start is not None:
            parent._loop_start += elapsed

        return {
            "text": _skill_tag(
                "subagent",
                response_text,
                sub_id=sub_id[:8],
                wait=True,
            )
        }

"""
Sub-Agent Skill — Spawn a focused worker for a specific task.

Blocks until the sub-agent completes, then returns the result.
Sub-agents run with clean context (no parent history) and a limited
skill set to prevent recursive nesting.

Used primarily by the persistent task execute loop to run parallel
or sequential steps via the ACT loop.
"""

import logging
import uuid

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SUB AGENT SKILL]"

TOOL_SCHEMA = {
    "name": "sub_agent",
    "description": "Spawn a focused worker for a specific task. Blocks until done, returns result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Clear, specific goal for the sub-agent.",
            }
        },
        "required": ["goal"],
    },
}

# Skills available to sub-agents — no sub_agent, no persistent_task (prevent recursion)
_SUB_AGENT_SKILLS = ['memory', 'find_tools', 'document', 'read']

_SUB_AGENT_PROMPT = (
    "Execute the following task. When done, provide your findings as a clear text response.\n\n"
    "## Task\n{goal}"
)


def handle_sub_agent(channel: str, params: dict) -> str:
    """
    Spawn a focused sub-agent to execute a specific task.

    Args:
        channel: Parent conversation channel (used for namespacing)
        params: {
            goal (str, required): Clear, specific goal for the sub-agent
        }

    Returns:
        Sub-agent result text, or error string on failure.
    """
    goal = (params.get('goal') or '').strip()
    if not goal:
        return "[SUB AGENT] Error: 'goal' parameter is required."

    logger.info(f"{LOG_PREFIX} Spawning sub-agent for channel={channel!r}: {goal[:80]}")

    try:
        from services.config_service import ConfigService
        from services import FrontalCortexService
        from services.act_orchestrator_service import ACTOrchestrator

        try:
            config = ConfigService.resolve_agent_config("frontal-cortex-unified")
        except Exception:
            config = ConfigService.resolve_agent_config("frontal-cortex")

        cortex_service = FrontalCortexService(config)

        act_prompt = _SUB_AGENT_PROMPT.format(goal=goal)

        orchestrator = ACTOrchestrator(
            config=config,
            max_iterations=50,
            cumulative_timeout=None,
            per_action_timeout=None,
            execution_gate=False,
            proactive=False,
        )

        result = orchestrator.run(
            channel=f"sub_agent_{channel}",
            text=goal,
            cortex_service=cortex_service,
            act_prompt=act_prompt,
            classification={'topic': 'sub_agent', 'confidence': 10},
            chat_history=[],
            selected_skills=_SUB_AGENT_SKILLS,
            session_id='sub_agent',
            exchange_id=f"sub_{channel}_{uuid.uuid4().hex[:8]}",
        )

        if result.final_response:
            logger.info(f"{LOG_PREFIX} Sub-agent completed: {result.iterations_used} iterations")
            return result.final_response

        # No final response — summarise termination
        return (
            f"[SUB AGENT] Task completed after {result.iterations_used} iterations "
            f"(termination: {result.termination_reason}). No final response produced."
        )

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Sub-agent failed for goal={goal[:80]!r}: {e}", exc_info=True)
        return f"[SUB AGENT] Error executing task: {str(e)[:200]}"

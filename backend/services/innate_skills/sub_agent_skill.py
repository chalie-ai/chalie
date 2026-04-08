"""
Sub-Agent Skill — Spawn a focused worker for a specific task.

Blocks until the sub-agent completes, then returns the result.
Sub-agents run with clean context (no parent history) and a limited
skill set to prevent recursive nesting.

Used primarily by the persistent task execute loop to run parallel
or sequential steps via the ACT loop.
"""

import logging

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

    # TODO: migrate to MessageProcessor tool loop
    # ACTOrchestrator has been deleted. sub_agent_skill must be rewired to use
    # MessageProcessor.send() with the new tool loop. Until then, return an error
    # so the calling LLM knows the skill is temporarily unavailable.
    logger.warning(f"{LOG_PREFIX} sub_agent is temporarily unavailable — ACTOrchestrator removed, migration pending")
    return "[SUB AGENT] Temporarily unavailable: sub_agent is being migrated to the new tool loop. Use individual tools directly for now."

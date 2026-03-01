"""
Shared Skill Outcome Recorder — records innate skill outcomes to procedural memory.

Extracted from tool_worker.py so both digest_worker (sync ACT path) and
tool_worker (background ACT path) share the same recording logic.

Dynamic tools are already recorded by
ToolRegistryService._log_outcome — skip them here to avoid double recording.
"""

import logging

logger = logging.getLogger(__name__)

from services.innate_skills.registry import PROCEDURAL_MEMORY_SKILLS

# Only the 4 core primitives are tracked in procedural memory.
# Dynamic tools are already recorded by ToolRegistryService._log_outcome.
INNATE_SKILLS = PROCEDURAL_MEMORY_SKILLS

UTILITY_MAP = {
    'recall': lambda r: 0.3 if r.get('status') == 'success' and 'no matches' not in str(r.get('result', '')).lower() else -0.1,
    'associate': lambda r: 0.2 if r.get('status') == 'success' else 0.0,
    'memorize': lambda r: 0.1,
    'introspect': lambda r: 0.1,
}


def record_skill_outcomes(actions_executed: list, topic: str):
    """Record innate skill outcomes to procedural memory. Dynamic tools skip (handled by ToolRegistryService)."""
    try:
        from services.procedural_memory_service import ProceduralMemoryService
        from services.database_service import get_shared_db_service
        from services.config_service import ConfigService

        db_service = get_shared_db_service()
        proc_config = ConfigService.get_agent_config("procedural-memory")
        proc_memory = ProceduralMemoryService(db_service, proc_config)

        for result in actions_executed:
            action_type = result.get('action_type', 'unknown')
            if action_type not in INNATE_SKILLS:
                continue
            success = result.get('status') == 'success'
            utility_fn = UTILITY_MAP.get(action_type, lambda r: 0.0)
            utility = utility_fn(result)

            proc_memory.record_action_outcome(
                action_name=action_type,
                success=success,
                reward=utility,
                topic=topic,
            )
    except Exception as e:
        logger.debug(f"[SKILL OUTCOMES] Procedural memory recording failed: {e}")

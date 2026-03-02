"""
ACT Reflection Service — Enqueues tool outputs for background experience assimilation.

Extracted from tool_worker.py to break the circular import between
tool_worker and digest_worker (P8).

Applies novelty gate layers 1 (ephemeral tool) and 2 (output size).
Layer 3 (content hash dedup) runs in the assimilation service.
"""

import json
import time
import logging

from services.innate_skills.registry import REFLECTION_FILTER_SKILLS
from services import act_redis_keys

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ACT REFLECTION]"


def _is_ephemeral_tool(tool_name: str) -> bool:
    """Return True if the tool declares output.ephemeral=true in its manifest."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if tool:
            return tool.get('manifest', {}).get('output', {}).get('ephemeral', False)
    except Exception:
        pass
    return False


def enqueue_tool_reflection(act_history: list, topic: str, user_prompt: str):
    """Push tool outputs to Redis for background experience assimilation.

    Applies novelty gate layers 1 (ephemeral tool type) and 2 (output size).
    Layer 3 (content hash dedup) runs in the assimilation service.
    """
    try:
        tool_outputs = []
        for action in act_history:
            if action.get('status') != 'success':
                continue
            action_type = action.get('action_type', '')
            if action_type in REFLECTION_FILTER_SKILLS:
                continue
            if _is_ephemeral_tool(action_type):
                continue
            result_str = str(action.get('result', ''))
            if len(result_str) < 50:
                continue
            tool_outputs.append({
                'tool': action_type,
                'result': result_str[:2000],
            })

        if not tool_outputs:
            return

        from services.redis_client import RedisClientService
        redis_conn = RedisClientService.create_connection()
        payload = json.dumps({
            'topic': topic,
            'user_prompt': user_prompt,
            'tool_outputs': tool_outputs,
            'timestamp': time.time(),
        })
        redis_conn.rpush(act_redis_keys.TOOL_REFLECTION_QUEUE, payload)
        redis_conn.expire(act_redis_keys.TOOL_REFLECTION_QUEUE, act_redis_keys.TOOL_REFLECTION_TTL)
        logger.debug(
            f"{LOG_PREFIX} Enqueued reflection for topic '{topic}' "
            f"({len(tool_outputs)} tool output(s))"
        )
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Reflection enqueue failed: {e}")

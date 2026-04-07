"""
Unified invocation tracker — single chokepoint for ALL execution tracking.

Every tool, innate skill, and interface capability invocation flows through
track(). Writes to both tool_performance_metrics (observability) and
procedural_memory (learning). No flags, no filters, no opt-out.
"""

import logging

logger = logging.getLogger(__name__)

# ── Reward map ────────────────────────────────────────────────────────────
# Skills with known reward profiles. Everything else gets the default.
_REWARD_MAP = {
    # Memory primitives
    'recall': lambda ok, res: 0.3 if ok and 'no matches' not in str(res).lower() else -0.1,
    'memorize': lambda ok, res: 0.1 if ok else -0.1,
    'associate': lambda ok, res: 0.2 if ok else 0.0,
    'introspect': lambda ok, res: 0.1 if ok else 0.0,
    # Productivity / knowledge
    'list': lambda ok, res: 0.2 if ok else -0.1,
    'schedule': lambda ok, res: 0.2 if ok else -0.1,
    'document': lambda ok, res: 0.2 if ok else -0.1,
    # Identity
    'autobiography': lambda ok, res: 0.1 if ok else 0.0,
    # Task management
    'persistent_task': lambda ok, res: 0.2 if ok else -0.1,
    # Research / discovery
    'read': lambda ok, res: 0.2 if ok else -0.1,
    'find_tools': lambda ok, res: 0.1 if ok else 0.0,
}

# Default reward for tools / interface capabilities not in the map.
_DEFAULT_REWARD_SUCCESS = 0.3
_DEFAULT_REWARD_FAILURE = -0.2
_DEFAULT_REWARD_EXTERNAL = -0.05  # upstream/network failures


def _compute_reward(name: str, success: bool, failure_class: str | None, result: str = '') -> float:
    """Compute reward for procedural memory."""
    if name in _REWARD_MAP:
        return _REWARD_MAP[name](success, result)
    if success:
        return _DEFAULT_REWARD_SUCCESS
    if failure_class == 'external':
        return _DEFAULT_REWARD_EXTERNAL
    return _DEFAULT_REWARD_FAILURE


def track(
    name: str,
    success: bool,
    latency_ms: float,
    topic: str = '',
    exchange_id: str = '',
    failure_class: str | None = None,
    result: str = '',
    channel: str = '',
) -> None:
    """Record a single invocation to both observability and learning stores.

    Args:
        name:          Tool, skill, or interface capability name.
        success:       Whether the invocation succeeded.
        latency_ms:    Wall-clock execution time in milliseconds.
        topic:         Deprecated alias for channel. Use channel instead.
        exchange_id:   Optional exchange ID for cross-referencing.
        failure_class: 'external' for upstream errors, 'internal' for bugs, None for success.
        result:        Raw result string (used by reward map for richer signal).
        channel:       Conversation channel (for context-specific stats).
    """
    # Accept topic as backward-compat alias for channel
    channel = channel or topic
    # ── tool_performance_metrics (observability) ──────────────────────────
    try:
        from services.tool_performance_service import ToolPerformanceService
        ToolPerformanceService().record_invocation(
            tool_name=name,
            exchange_id=exchange_id,
            success=success,
            latency_ms=float(latency_ms),
            cost=0.0,
        )
    except Exception as e:
        logger.debug(f"[TRACKER] Performance metric failed for {name}: {e}")

    # ── procedural_memory (learning) ──────────────────────────────────────
    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        reward = _compute_reward(name, success, failure_class, result)
        KnowledgeService(db_service).record_procedure_outcome(
            action_name=name,
            success=success,
            reward=reward,
            channel=channel,
            failure_class=failure_class,
        )
    except Exception as e:
        logger.debug(f"[TRACKER] Procedural memory failed for {name}: {e}")

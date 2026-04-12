"""
Unified invocation tracker — single chokepoint for ALL execution tracking.

Every tool, innate skill, and interface capability invocation flows through
track(). Writes to tool_performance_metrics for observability.
"""

import logging

logger = logging.getLogger(__name__)


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
    """Record a single invocation to the observability store.

    Args:
        name:          Tool, skill, or interface capability name.
        success:       Whether the invocation succeeded.
        latency_ms:    Wall-clock execution time in milliseconds.
        topic:         Deprecated alias for channel. Use channel instead.
        exchange_id:   Optional exchange ID for cross-referencing.
        failure_class: 'external' for upstream errors, 'internal' for bugs, None for success.
        result:        Raw result string (unused; kept for call-site compatibility).
        channel:       Conversation channel (for context-specific stats).
    """
    channel = channel or topic
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

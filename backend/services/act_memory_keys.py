"""
Centralized MemoryStore key patterns for the ACT loop system.

All ACT-related keys MUST be defined here to prevent scattered
string construction across 5+ files.
"""


def tool_raw_cache(topic: str, exchange_id: str = '') -> str:
    """Return the MemoryStore key for the raw tool output cache for a topic+exchange.

    Args:
        topic: Conversation topic identifier.
        exchange_id: Exchange correlation ID to prevent cross-exchange bleed.

    Returns:
        MemoryStore key string.
    """
    if exchange_id:
        return f"tool_raw_cache:{topic}:{exchange_id}"
    return f"tool_raw_cache:{topic}"


def cancel_flag(cycle_id: str) -> str:
    """Return the MemoryStore key for the cancellation flag of a cycle.

    Args:
        cycle_id: Unique cycle identifier.

    Returns:
        MemoryStore key string.
    """
    return f"cancel:{cycle_id}"


def heartbeat(job_id: str) -> str:
    """Return the MemoryStore key for the heartbeat signal of a job.

    Args:
        job_id: Unique job identifier.

    Returns:
        MemoryStore key string.
    """
    return f"heartbeat:{job_id}"


def sse_channel(uuid: str) -> str:
    """Return the MemoryStore key for the SSE channel of a session.

    Args:
        uuid: Session UUID.

    Returns:
        MemoryStore key string.
    """
    return f"sse:{uuid}"


TOOL_REFLECTION_QUEUE = "tool_reflection:pending"
TOOL_REFLECTION_TTL = 86400  # 24 hours

"""
Centralized MemoryStore key patterns for the ACT loop system.

All ACT-related keys MUST be defined here to prevent scattered
string construction across 5+ files.
"""


def deferred_cards(topic: str) -> str:
    """Return the MemoryStore key for deferred card offers for a topic.

    Args:
        topic: Conversation topic identifier.

    Returns:
        MemoryStore key string.
    """
    return f"deferred_cards:{topic}"


def tool_raw_cache(topic: str) -> str:
    """Return the MemoryStore key for the raw tool output cache for a topic.

    Args:
        topic: Conversation topic identifier.

    Returns:
        MemoryStore key string.
    """
    return f"tool_raw_cache:{topic}"


def tool_card_cache(topic: str, invocation_id: str) -> str:
    """Return the MemoryStore key for a rendered tool card.

    Args:
        topic: Conversation topic identifier.
        invocation_id: Unique invocation identifier for the tool call.

    Returns:
        MemoryStore key string.
    """
    return f"tool_card_cache:{topic}:{invocation_id}"


def rendered_cards(topic: str) -> str:
    """Short-lived set tracking which tools have had cards rendered for this topic."""
    return f"rendered_cards:{topic}"


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


def sse_pending(uuid: str) -> str:
    """Return the MemoryStore key for pending SSE events of a session.

    Args:
        uuid: Session UUID.

    Returns:
        MemoryStore key string.
    """
    return f"sse_pending:{uuid}"


TOOL_REFLECTION_QUEUE = "tool_reflection:pending"
TOOL_REFLECTION_TTL = 86400  # 24 hours

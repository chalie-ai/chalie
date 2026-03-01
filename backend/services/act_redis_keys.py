"""
Centralized Redis key patterns for the ACT loop system.

All ACT-related Redis keys MUST be defined here to prevent scattered
string construction across 5+ files.
"""


def deferred_cards(topic: str) -> str:
    return f"deferred_cards:{topic}"


def tool_raw_cache(topic: str) -> str:
    return f"tool_raw_cache:{topic}"


def tool_card_cache(topic: str, invocation_id: str) -> str:
    return f"tool_card_cache:{topic}:{invocation_id}"


def rendered_cards(topic: str) -> str:
    """Short-lived set tracking which tools have had cards rendered for this topic."""
    return f"rendered_cards:{topic}"


def cancel_flag(cycle_id: str) -> str:
    return f"cancel:{cycle_id}"


def heartbeat(job_id: str) -> str:
    return f"heartbeat:{job_id}"


def sse_channel(uuid: str) -> str:
    return f"sse:{uuid}"


def sse_pending(uuid: str) -> str:
    return f"sse_pending:{uuid}"


TOOL_REFLECTION_QUEUE = "tool_reflection:pending"
TOOL_REFLECTION_TTL = 86400  # 24 hours

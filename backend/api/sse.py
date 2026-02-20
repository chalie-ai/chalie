"""
Server-Sent Events (SSE) utilities.

Provides event formatting, keepalive, and response headers for SSE streams.
"""

import json


SSE_RETRY_MS = 3000  # EventSource auto-reconnect interval


def sse_event(event: str, data: dict) -> str:
    """Format a named SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_keepalive() -> str:
    """SSE comment line used as keepalive ping."""
    return ": keepalive\n\n"


def sse_retry() -> str:
    """Initial retry directive for EventSource auto-reconnect."""
    return f"retry: {SSE_RETRY_MS}\n\n"


def sse_headers() -> dict:
    """HTTP headers for an SSE response."""
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    }

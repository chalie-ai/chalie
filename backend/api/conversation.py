"""
Conversation blueprint — /conversation/recent.

The /chat endpoint has been replaced by the WebSocket handler in api/websocket.py.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session
from services.blocks_render_service import BlocksRenderService

_blocks_svc = BlocksRenderService()

logger = logging.getLogger(__name__)

conversation_bp = Blueprint('conversation', __name__)



@conversation_bp.route('/conversation/recent', methods=['GET'])
@require_session
def conversation_recent():
    """Return paginated conversation from the current (or most recently expired) thread."""
    try:
        from services.thread_service import get_thread_service
        from services.thread_conversation_service import ThreadConversationService

        WORKING_MEMORY_SIZE = 12

        # Parse and clamp query params
        try:
            limit = max(1, min(120, int(request.args.get("limit", 12))))
        except (ValueError, TypeError):
            limit = 12
        try:
            offset = max(0, int(request.args.get("offset", 0)))
        except (ValueError, TypeError):
            offset = 0

        ts = get_thread_service()
        thread_id = ts.get_active_thread_id("default")

        tcs = ThreadConversationService()
        from_expired = False

        # Resolve thread: MemoryStore pointer first, then SQLite
        if not thread_id:
            thread_id, from_expired = tcs.get_most_recent_thread_id()
            if not thread_id:
                return jsonify({
                    "thread_id": None,
                    "exchanges": [],
                    "total": 0,
                    "has_more": False,
                    "working_memory_count": WORKING_MEMORY_SIZE,
                    "from_expired": False,
                }), 200

        # Always read from SQLite — survives restarts, no MemoryStore dependency
        page = tcs.get_paginated_history_durable(thread_id, limit=limit, offset=offset)
        total = page["total"]
        exchanges_raw = page["exchanges"]
        has_more = page["has_more"]

        formatted = []
        for i, ex in enumerate(exchanges_raw):
            prompt = ex.get("prompt", {}) or {}
            response = ex.get("response", {}) or {}

            # Distance from the end of the full history for this exchange in the slice
            distance_from_end = total - (offset + i + 1)
            in_working_memory = (not from_expired) and (distance_from_end < WORKING_MEMORY_SIZE)

            response_text = response.get("message", "") if isinstance(response, dict) else ""
            formatted.append({
                "id": ex.get("id", ""),
                "prompt": prompt.get("message", "") if isinstance(prompt, dict) else "",
                "blocks": _blocks_svc.from_markdown(response_text) if response_text else [],
                "topic": ex.get("topic", ""),
                "timestamp": prompt.get("time", "") if isinstance(prompt, dict) else "",
                "in_working_memory": in_working_memory,
            })

        return jsonify({
            "thread_id": thread_id,
            "exchanges": formatted,
            "total": total,
            "has_more": has_more,
            "working_memory_count": WORKING_MEMORY_SIZE,
            "from_expired": from_expired,
        }), 200

    except Exception as e:
        logger.error(f"[REST API] conversation/recent error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve conversation"}), 500



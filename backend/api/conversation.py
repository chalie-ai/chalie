"""
Conversation blueprint — /conversation/recent, /conversation/summary.

The /chat endpoint has been replaced by the WebSocket handler in api/websocket.py.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session

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

        # If no active thread, or active thread is empty, fall back to most recent expired thread
        if not thread_id:
            expired_id = tcs.get_most_recent_expired_thread_id()
            if expired_id:
                thread_id = expired_id
                from_expired = True
            else:
                return jsonify({
                    "thread_id": None,
                    "exchanges": [],
                    "total": 0,
                    "has_more": False,
                    "working_memory_count": WORKING_MEMORY_SIZE,
                    "from_expired": False,
                }), 200
        else:
            # Check if the active thread has any exchanges (MemoryStore or SQLite)
            active_total = tcs.store.llen(tcs._conv_key(thread_id))
            if active_total == 0:
                # Try SQLite fallback for active thread first
                page = tcs.get_paginated_history(thread_id, limit=1, offset=0)
                if page["total"] == 0:
                    expired_id = tcs.get_most_recent_expired_thread_id()
                    if expired_id:
                        thread_id = expired_id
                        from_expired = True

        page = tcs.get_paginated_history(thread_id, limit=limit, offset=offset)
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

            formatted.append({
                "id": ex.get("id", ""),
                "prompt": prompt.get("message", "") if isinstance(prompt, dict) else "",
                "response": response.get("message", "") if isinstance(response, dict) else "",
                "topic": ex.get("topic", ""),
                "timestamp": ex.get("timestamp", ""),
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


@conversation_bp.route('/conversation/summary', methods=['GET'])
@require_session
def conversation_summary():
    """Return compressed conversation summaries across time ranges."""
    try:
        from datetime import datetime, timedelta, timezone
        from services.time_utils import parse_utc
        from services.database_service import get_shared_db_service
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.config_service import ConfigService

        result = {"today": [], "this_week": [], "older_highlights": []}

        # Episodes for this week and older
        try:
            db = get_shared_db_service()
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            retrieval = EpisodicRetrievalService(db, episodic_config)
            episodes = retrieval.retrieve_episodes("conversation summary", limit=10)

            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)

            for ep in episodes:
                created = ep.get("created_at")
                entry = {
                    "gist": ep.get("gist", ""),
                    "topic": ep.get("topic", ""),
                    "salience": ep.get("salience", 0),
                    "created_at": str(created) if created else "",
                }
                if created:
                    created_aware = parse_utc(created)
                    if created_aware >= week_ago:
                        result["this_week"].append(entry)
                    else:
                        result["older_highlights"].append(entry)
                else:
                    result["this_week"].append(entry)
        except Exception as e:
            logger.warning(f"[REST API] Episode retrieval failed: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] conversation/summary error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve summary"}), 500

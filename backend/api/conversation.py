"""
Conversation blueprint — /conversation/recent, /conversation/summary, /conversation/spark-status.

The /chat endpoint has been replaced by the WebSocket handler in api/websocket.py.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

conversation_bp = Blueprint('conversation', __name__)


@conversation_bp.route('/conversation/spark-status', methods=['GET'])
@require_session
def spark_status():
    """Return whether a welcome message is still needed for this user."""
    try:
        from services.spark_state_service import SparkStateService
        svc = SparkStateService()
        return jsonify({"needs_welcome": svc.needs_welcome()}), 200
    except Exception as e:
        logger.error(f"[REST API] spark-status error: {e}", exc_info=True)
        return jsonify({"needs_welcome": False}), 200


@conversation_bp.route('/conversation/recent', methods=['GET'])
@require_session
def conversation_recent():
    """Return recent conversation from the current thread."""
    try:
        from services.thread_service import get_thread_service
        from services.thread_conversation_service import ThreadConversationService

        ts = get_thread_service()
        thread_id = ts.get_active_thread_id("default")

        if not thread_id:
            return jsonify({"thread_id": None, "exchanges": []}), 200

        tcs = ThreadConversationService()
        exchanges = tcs.get_conversation_history(thread_id)

        formatted = []
        for ex in exchanges:
            prompt = ex.get("prompt", {}) or {}
            response = ex.get("response", {}) or {}
            formatted.append({
                "id": ex.get("id", ""),
                "prompt": prompt.get("message", "") if isinstance(prompt, dict) else "",
                "response": response.get("message", "") if isinstance(response, dict) else "",
                "topic": ex.get("topic", ""),
                "timestamp": ex.get("timestamp", ""),
            })

        return jsonify({"thread_id": thread_id, "exchanges": formatted}), 200

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

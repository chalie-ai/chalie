"""
Memory blueprint — /memory/search.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

memory_bp = Blueprint('memory', __name__)


@memory_bp.route('/memory/search', methods=['GET'])
@require_session
def memory_search():
    """Semantic search across all memory layers."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400

    try:
        from services import episodic_retrieval_service
        from services.data_graph_service import get_data_graph_service

        results = []

        # Episodic search
        try:
            episodes = episodic_retrieval_service.retrieve(
                query_text=query,
                channel=None,
                radius=0.5,
            )
            for ep in episodes:
                results.append({
                    "type": "episode",
                    "content": ep.get("gist", ""),
                    "score": ep.get("composite_score", ep.get("score", 0)),
                    "created_at": str(ep.get("created_at", "")),
                })
        except Exception as e:
            logger.warning(f"[Memory] Episode search failed: {e}")

        # Data graph concept search
        try:
            dgs = get_data_graph_service()
            items = dgs.recall(query, kinds=['user_specific', 'system'], limit=5)
            for c in items:
                results.append({
                    "type": "concept",
                    "content": c.get("key", "") + ": " + c.get("value", ""),
                    "score": c.get("composite_score", c.get("retrieval_weight", 0)),
                    "confidence": c.get("retrieval_weight", 0),
                })
        except Exception as e:
            logger.warning(f"[Memory] Data graph search failed: {e}")

        # Sort by score descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"[REST API] memory/search error: {e}", exc_info=True)
        return jsonify({"error": "Failed to search memory"}), 500

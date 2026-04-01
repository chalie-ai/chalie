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
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        from services.knowledge_service import KnowledgeService
        from services.config_service import ConfigService

        db = get_shared_db_service()
        results = []

        # Episodic search
        try:
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            retrieval = EpisodicService(db, episodic_config)
            episodes = retrieval.retrieve_episodes(query_text=query, limit=5)
            for ep in episodes:
                results.append({
                    "type": "episode",
                    "content": ep.get("gist", ""),
                    "score": ep.get("composite_score", ep.get("score", 0)),
                    "created_at": str(ep.get("created_at", "")),
                })
        except Exception as e:
            logger.warning(f"[Memory] Episode search failed: {e}")

        # Knowledge concept search
        try:
            ks = KnowledgeService(db)
            concepts = ks.recall(query, kinds=['concept'], limit=5)
            for c in concepts:
                results.append({
                    "type": "concept",
                    "content": c.get("key", "") + ": " + c.get("value", ""),
                    "score": c.get("rrf_score", c.get("confidence", 0)),
                    "confidence": c.get("confidence", 0),
                })
        except Exception as e:
            logger.warning(f"[Memory] Concept search failed: {e}")

        # Sort by score descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"[REST API] memory/search error: {e}", exc_info=True)
        return jsonify({"error": "Failed to search memory"}), 500

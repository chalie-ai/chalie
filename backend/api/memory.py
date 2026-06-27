import logging
from typing import TYPE_CHECKING, cast

from flask import request
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

memory_bp = Namespace('memory', description='Memory search')


@memory_bp.route('/memory/search')
class MemorySearchResource(Resource):
    @require_session
    @memory_bp.response(200, "Search results")
    @memory_bp.response(400, "Missing query")
    @memory_bp.response(500, "Failed to search memory")
    def get(self):
        query = request.args.get("q", "").strip()
        if not query:
            return {"error": "Missing 'q' query parameter"}, 400

        try:
            from services import episodic_retrieval_service
            from services.data_graph_service import get_data_graph_service

            results = []

            # Episodic search
            try:
                episodes = cast("list[dict[str, object]]", episodic_retrieval_service.retrieve(
                    query_text=query,
                    channel=None,
                ))
                for ep in episodes:
                    results.append({
                        "type": "episode",
                        "content": cast(str, ep.get("gist", "")),
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
                        "content": cast(str, c.get("key", "")) + ": " + cast(str, c.get("value", "")),
                        "score": c.get("composite_score", c.get("retrieval_weight", 0)),
                        "confidence": c.get("retrieval_weight", 0),
                    })
            except Exception as e:
                logger.warning(f"[Memory] Data graph search failed: {e}")

            # Sort by score descending
            results.sort(key=lambda r: cast(float, r.get("score", 0)), reverse=True)

            return {"results": results}, 200

        except Exception as e:
            logger.exception(f"[REST API] memory/search error: {e}")
            return {"error": "Failed to search memory"}, 500
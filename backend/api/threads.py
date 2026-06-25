"""Threads search blueprint — GET /threads/search?q=

Reuses the ``thread_gist`` index (KNN + FTS) built in workstream D.
"""

import logging

from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

from .auth import require_session

logger = logging.getLogger(__name__)
threads_bp = Blueprint("threads", __name__)


@threads_bp.route("/threads/search", methods=["GET"])
@require_session
def threads_search() -> ResponseReturnValue:
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400
    try:
        limit = min(request.args.get("limit", 10, type=int), 20)
    except (ValueError, TypeError):
        limit = 10

    try:
        from services.thread_gist_service import get_thread_gist_service  # noqa: PLC0415
        results = get_thread_gist_service().search(query, limit=limit)
        return jsonify({"results": results, "query": query})
    except Exception as exc:
        logger.exception("[THREADS API] search error: %s", exc)
        return jsonify({"error": "Search failed"}), 500
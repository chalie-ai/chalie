"""Probe endpoints — test harness data injection.

POST /api/probe/knowledge  — insert into data_graph
POST /api/probe/episode    — insert an episode
"""

import logging
from flask import Blueprint, jsonify, request
from .auth import require_auth

logger = logging.getLogger(__name__)

probe_bp = Blueprint("probe", __name__, url_prefix="/api/probe")


@probe_bp.route("/knowledge", methods=["POST"])
@require_auth
def insert_knowledge():
    body = request.get_json(silent=True) or {}

    key = body.get("key")
    value = body.get("value", "")
    if not key:
        return jsonify({"error": "key is required"}), 400

    from services.data_graph_service import get_data_graph_service
    svc = get_data_graph_service()

    result = svc.store(
        kind=body.get("kind", "user_specific"),
        key=key,
        value=value,
        source=body.get("source", "probe"),
    )

    if result is None:
        return jsonify({"error": "store rejected"}), 422

    return jsonify(result), 201


@probe_bp.route("/episode", methods=["POST"])
@require_auth
def insert_episode():
    body = request.get_json(silent=True) or {}

    required = ["intent", "context", "action", "emotion", "outcome",
                "gist", "salience", "channel"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    from services.database_service import get_shared_db_service
    from services.episodic_service import EpisodicService

    svc = EpisodicService(get_shared_db_service())
    episode_id = svc.store_episode(body)

    return jsonify({"id": episode_id}), 201

"""
Moments API — pin, list, search, and forget moments.

Routes (all require session auth):
  POST   /moments              — pin a moment
  GET    /moments              — list all active moments
  POST   /moments/<id>/forget  — forget a moment (reversible)
  GET    /moments/search       — semantic search (?q=query)
"""

import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

moments_bp = Blueprint("moments", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_moment_service():
    from services.database_service import get_shared_db_service
    from services.moment_service import MomentService
    return MomentService(get_shared_db_service())


def _serialize_dt(val):
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize_moment(moment: dict) -> dict:
    """Convert datetime fields to ISO strings for JSON serialisation."""
    out = dict(moment)
    for field in ("pinned_at", "sealed_at", "last_enriched_at", "created_at", "updated_at"):
        if field in out:
            out[field] = _serialize_dt(out[field])
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@moments_bp.route("/moments", methods=["POST"])
@require_session
def create_moment():
    """Pin a moment."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    message_text = (data.get("message_text") or "").strip()
    if not message_text:
        return jsonify({"error": "message_text is required"}), 400

    if len(message_text) > 10000:
        return jsonify({"error": "message_text must be 10000 characters or fewer"}), 400

    exchange_id = (data.get("exchange_id") or "").strip() or None
    topic = (data.get("topic") or "").strip() or None
    thread_id = (data.get("thread_id") or "").strip() or None
    title = (data.get("title") or "").strip() or None

    try:
        svc = _get_moment_service()
        result = svc.create_moment(
            message_text=message_text,
            exchange_id=exchange_id,
            topic=topic,
            thread_id=thread_id,
            title=title,
        )

        if not result:
            return jsonify({"error": "Failed to create moment"}), 500

        # Near-duplicate detected
        if result.get("duplicate"):
            return jsonify({
                "item": _serialize_moment(result),
                "duplicate": True,
                "existing_id": result.get("existing_id"),
            }), 200

        return jsonify({"item": _serialize_moment(result)}), 201

    except Exception as e:
        logger.error(f"[MOMENTS API] create_moment error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@moments_bp.route("/moments", methods=["GET"])
@require_session
def list_moments():
    """List all active moments."""
    try:
        svc = _get_moment_service()
        moments = svc.get_all_moments()
        return jsonify({"items": [_serialize_moment(m) for m in moments]})

    except Exception as e:
        logger.error(f"[MOMENTS API] list_moments error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@moments_bp.route("/moments/<moment_id>/forget", methods=["POST"])
@require_session
def forget_moment(moment_id):
    """Forget a moment — sets status to 'forgotten', reverses salience boosts."""
    try:
        svc = _get_moment_service()
        success = svc.forget_moment(moment_id)
        if success:
            return jsonify({"ok": True})
        return jsonify({"error": "Moment not found"}), 404

    except Exception as e:
        logger.error(f"[MOMENTS API] forget_moment error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@moments_bp.route("/moments/search", methods=["GET"])
@require_session
def search_moments():
    """Semantic search over moments."""
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    try:
        svc = _get_moment_service()
        results = svc.search_moments(query, limit=3)
        return jsonify({"items": [_serialize_moment(m) for m in results]})

    except Exception as e:
        logger.error(f"[MOMENTS API] search_moments error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

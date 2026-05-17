"""
Moments API — pin, list, search, and forget moments.

Routes (all require session auth):
  POST   /moments                         — pin a moment (body: {transcript_id})
  GET    /moments                         — list all active moments
  POST   /moments/<transcript_id>/forget  — soft-delete
  GET    /moments/search                  — semantic search (?q=query)
"""

import logging

from flask import Blueprint, g, jsonify, request

from .auth import require_session
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"

moments_bp = Blueprint("moments", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dg():
    if 'moments_dg' not in g:
        from services.data_graph_service import get_data_graph_service
        g.moments_dg = get_data_graph_service()
    return g.moments_dg


def _get_db():
    if 'moments_db' not in g:
        from services.database_service import get_shared_db_service
        g.moments_db = get_shared_db_service()
    return g.moments_db


def _serialize_moment(row: dict) -> dict:
    created_at = row.get('first_seen_at') or row.get('created_at')
    if created_at:
        try:
            created_at = parse_utc(created_at).isoformat()
        except Exception:
            pass

    key = row.get('key', '')
    transcript_id = None
    if key.startswith('moment_'):
        try:
            transcript_id = int(key[len('moment_'):])
        except (ValueError, TypeError):
            pass

    return {
        'transcript_id': transcript_id,
        'key': key,
        'value': row.get('value') or '',
        'created_at': created_at,
    }


def _fetch_transcript_row(db, transcript_id: int) -> dict | None:
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role, content FROM transcript WHERE id = ?",
            (transcript_id,),
        )
        row = cursor.fetchone()
        cursor.close()

    if row is None:
        return None
    return {'id': row[0], 'role': row[1], 'content': row[2]}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@moments_bp.route("/moments", methods=["POST"])
@require_session
def create_moment():
    """Pin an assistant transcript turn as a moment."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    transcript_id = data.get("transcript_id")
    if not transcript_id:
        return jsonify({"error": "transcript_id is required"}), 400

    try:
        transcript_id = int(transcript_id)
    except (TypeError, ValueError):
        return jsonify({"error": "transcript_id must be an integer"}), 400

    db = _get_db()
    dg = _get_dg()

    turn = _fetch_transcript_row(db, transcript_id)
    if turn is None:
        return jsonify({"error": "Transcript row not found"}), 404

    if turn['role'] != 'assistant':
        return jsonify({"error": "Only assistant turns can be pinned as moments"}), 400

    from services.data_graph_service import KIND_MOMENT
    key = f"moment_{transcript_id}"

    with db.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM data_graph WHERE kind=? AND key=? AND active=1 LIMIT 1",
            (KIND_MOMENT, key)
        ).fetchone()
    already_exists = row is not None

    result = dg.store(
        kind=KIND_MOMENT,
        key=key,
        value=turn['content'],
        source='pin',
    )

    if result is None:
        return jsonify({"error": "Failed to store moment"}), 500

    status = 200 if already_exists else 201
    return jsonify({"item": _serialize_moment(result)}), status


@moments_bp.route("/moments", methods=["GET"])
@require_session
def list_moments():
    """List all active kind='moment' rows ordered by first_seen_at DESC."""
    try:
        from services.data_graph_service import KIND_MOMENT
        dg = _get_dg()
        rows = dg.fetch(kinds=[KIND_MOMENT], order_by='first_seen_at DESC')
        return jsonify({"items": [_serialize_moment(r) for r in rows]})
    except Exception as e:
        logger.exception(f"[MOMENTS API] list_moments error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@moments_bp.route("/moments/<int:transcript_id>/forget", methods=["POST"])
@require_session
def forget_moment(transcript_id):
    """Soft-delete a moment row."""
    try:
        from services.data_graph_service import KIND_MOMENT
        dg = _get_dg()
        key = f"moment_{transcript_id}"
        rows = dg.fetch(kinds=[KIND_MOMENT])
        match = next((r for r in rows if r.get('key') == key), None)
        if match is None:
            return jsonify({"error": "Moment not found"}), 404
        dg.soft_delete_by_id(match['id'])
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception(f"[MOMENTS API] forget_moment error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@moments_bp.route("/moments/search", methods=["GET"])
@require_session
def search_moments():
    """Semantic search over moments."""
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    try:
        from services.data_graph_service import KIND_MOMENT
        dg = _get_dg()
        results = dg.recall(query, kinds=[KIND_MOMENT])
        return jsonify({"items": [_serialize_moment(r) for r in results]})
    except Exception as e:
        logger.exception(f"[MOMENTS API] search_moments error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500

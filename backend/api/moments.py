"""
Moments API — pin, list, search, and forget moments.

Routes (all require session auth):
  POST   /moments                    — pin a moment (body: {transcript_id})
  GET    /moments                    — list all active moments
  POST   /moments/<transcript_id>/forget — cascade soft-delete
  GET    /moments/search             — semantic search (?q=query)
"""

import json
import logging

from flask import Blueprint, g, jsonify, request

from .auth import require_session
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

moments_bp = Blueprint("moments", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ks():
    """
    Return a request-scoped KnowledgeService.

    One KnowledgeService per request, cached on Flask's ``g`` object. The
    underlying DatabaseService is a process singleton so this is purely about
    avoiding repeated wrapper construction within a single route call.
    """
    if 'moments_ks' not in g:
        from services.database_service import get_shared_db_service
        from services.knowledge_service import KnowledgeService
        g.moments_db = get_shared_db_service()
        g.moments_ks = KnowledgeService(g.moments_db)
    return g.moments_ks


def _get_db():
    """Return the same DatabaseService bound to the request-scoped KS."""
    if 'moments_db' not in g:
        _get_ks()  # side-effect: populates g.moments_db
    return g.moments_db


def _serialize_moment(row: dict) -> dict:
    """Build the wire-format moment dict from a knowledge row."""
    data = row.get('data')
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            data = {}
    elif not isinstance(data, dict):
        data = {}

    created_at = row.get('created_at')
    if created_at:
        try:
            created_at = parse_utc(created_at).isoformat()
        except Exception:
            pass  # leave the raw value; parse_utc should not hard-fail reads

    return {
        'transcript_id': data.get('transcript_id'),
        'key': row.get('key'),
        'value': row.get('value') or '',
        'created_at': created_at,
    }


def _fetch_transcript_row(db, transcript_id: int) -> dict | None:
    """
    Return transcript row dict or ``None`` when the row genuinely does not
    exist.

    Transcript is append-only with no soft-delete column — callers see every
    row that was ever written. DB failures are re-raised so Flask's error
    handler produces a 500 instead of this helper pretending the row just
    isn't there.
    """
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


def _cascade_forget(ks, transcript_id: int) -> bool:
    """
    Soft-delete parent moment and all context children.

    Returns True when at least the parent was deleted, False when the parent
    did not exist (caller responds 404).

    All mutation runs through ``KnowledgeService`` so the FTS shadow table
    stays in sync — no raw SQL from this layer.
    """
    moment_key = f"moment_{transcript_id}"

    # Parent via KnowledgeService.forget — this path already clears FTS and
    # returns False when the row was already absent (→ 404 upstream).
    parent_existed = ks.forget('user', moment_key)
    if not parent_existed:
        return False

    # Children: bulk soft-delete via the service method that also clears FTS.
    try:
        ks.forget_all_by_entity(moment_key, kind='moment_context')
    except Exception as e:
        # Parent already forgotten — surface the inconsistency but don't
        # lie to the UI (it already sees the parent as gone).
        logger.error(
            f"[MOMENTS API] cascade child cleanup failed for {moment_key}: {e}"
        )

    return True


def _list_active_moments(ks) -> list:
    """
    Return all active kind='moment' rows for entity='user' ordered by created_at DESC.

    ``KnowledgeService.get_by_kind`` sorts by confidence, which is useless for
    moments (they're all 1.0). This small wrapper is the one place in the
    codebase that needs a chronological list of moments — keeping it inline
    is cheaper than bloating ``KnowledgeService`` with a moment-specific sort
    option.
    """
    with ks.db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, kind, entity, key, value, data, decay_class,
                   confidence, created_at, updated_at
            FROM knowledge
            WHERE kind = 'moment'
              AND entity = 'user'
              AND deleted_at IS NULL
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()

    cols = ['id', 'kind', 'entity', 'key', 'value', 'data', 'decay_class',
            'confidence', 'created_at', 'updated_at']
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@moments_bp.route("/moments", methods=["POST"])
@require_session
def create_moment():
    """Pin an assistant transcript turn as a moment.

    Returns 201 the first time a moment is created, 200 when re-pinning an
    existing moment (idempotent UPSERT surface).
    """
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

    ks = _get_ks()
    db = _get_db()

    turn = _fetch_transcript_row(db, transcript_id)
    if turn is None:
        return jsonify({"error": "Transcript row not found"}), 404

    if turn['role'] != 'assistant':
        return jsonify({"error": "Only assistant turns can be pinned as moments"}), 400

    moment_key = f"moment_{transcript_id}"

    # Distinguish create vs idempotent re-pin for the HTTP status code.
    already_exists = ks.get('user', moment_key) is not None

    result = ks.store(
        kind='moment',
        entity='user',
        key=moment_key,
        value=turn['content'],
        data={'transcript_id': transcript_id, 'channel': 'user', 'created_at': utc_now().isoformat()},
        decay_class='permanent',
        confidence=1.0,
        source='user_pin',
    )

    if result is None:
        return jsonify({"error": "Failed to store moment"}), 500

    status = 200 if already_exists else 201
    return jsonify({"item": _serialize_moment(result)}), status


@moments_bp.route("/moments", methods=["GET"])
@require_session
def list_moments():
    """List all active kind='moment' rows ordered by created_at DESC."""
    try:
        ks = _get_ks()
        rows = _list_active_moments(ks)
        return jsonify({"items": [_serialize_moment(r) for r in rows]})

    except Exception as e:
        logger.error(f"[MOMENTS API] list_moments error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@moments_bp.route("/moments/<int:transcript_id>/forget", methods=["POST"])
@require_session
def forget_moment(transcript_id):
    """Cascade soft-delete moment + all context children."""
    try:
        ks = _get_ks()
        if not _cascade_forget(ks, transcript_id):
            return jsonify({"error": "Moment not found"}), 404
        return jsonify({"ok": True})

    except Exception as e:
        logger.error(f"[MOMENTS API] forget_moment error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@moments_bp.route("/moments/search", methods=["GET"])
@require_session
def search_moments():
    """Semantic search over moments and their context. Rollup to parent."""
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    try:
        ks = _get_ks()
        results = ks.recall(query, kinds=['moment', 'moment_context'])

        seen = {}  # parent_key -> serialized moment dict
        for row in results:
            kind = row.get('kind')
            if kind == 'moment':
                parent_key = row.get('key')
            else:
                row_data = row.get('data') or {}
                if isinstance(row_data, str):
                    try:
                        row_data = json.loads(row_data)
                    except (json.JSONDecodeError, TypeError):
                        row_data = {}
                parent_key = row_data.get('parent_key')

            if not parent_key or parent_key in seen:
                continue

            if kind == 'moment':
                moment_row = row
            else:
                moment_row = ks.get('user', parent_key)

            if moment_row is not None:
                seen[parent_key] = _serialize_moment(moment_row)

        return jsonify({"items": list(seen.values())})

    except Exception as e:
        logger.error(f"[MOMENTS API] search_moments error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

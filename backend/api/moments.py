"""
Moments API — pin, list, search, and forget moments.

Routes (all require session auth):
  POST   /moments                         — pin a moment
                                            (body: {transcript_id} OR {message_text})
  GET    /moments                         — list all active moments
  POST   /moments/<transcript_id>/forget  — soft-delete
  GET    /moments/search                  — semantic search (?q=query)

Resolving the pinned turn
-------------------------
The chat UI does NOT hold the assistant transcript row id: the WS message event
carries only ``exchange_id`` (a per-request UUID, never stored in the transcript
table — see api/chat.py) and the rendered message text. The ``transcript`` table
has no ``exchange_id`` column either (see schema.sql), so the row cannot be
resolved from ``exchange_id``.

``create_moment`` therefore accepts EITHER an explicit integer ``transcript_id``
(programmatic callers / existing contract) OR a ``message_text`` string (what the
remember button actually sends). When only ``message_text`` is given, the server
finds the matching assistant turn by comparing the plaintext of recent assistant
turns — via ``services.markup.extract_plaintext`` — against the supplied text.
The remember button derives its text from the same sanitised content with the
frontend ``extractPlaintext``, so the two plaintext projections line up.
"""

import logging

from flask import Blueprint, g, jsonify, request

from .auth import require_session
from services.markup import extract_plaintext
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"

# Channel the user chat turns are written under (api/chat.py UMP path).
_USER_CHANNEL = "user"

# How many recent assistant turns to scan when resolving a pin by message text.
_RESOLVE_SCAN_LIMIT = 200

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

    value = row.get('value') or ''
    return {
        # ``id`` is the data_graph row id; ``transcript_id`` is the pinned
        # assistant turn. The forget route is keyed by ``transcript_id`` (the
        # moment key encodes it), so the client uses that for the round-trip.
        'id': row.get('id'),
        'transcript_id': transcript_id,
        'key': key,
        'value': value,
        # Alias the Recall overlay reads directly (moment_search.js renders
        # ``item.message_text``); kept beside ``value`` so both shapes resolve.
        'message_text': value,
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


def _normalise(text: str) -> str:
    """Collapse whitespace so two plaintext projections compare cleanly."""
    return " ".join((text or "").split())


def _resolve_assistant_turn_by_text(db, message_text: str) -> dict | None:
    """Find the assistant transcript turn whose rendered text matches.

    Scans recent assistant turns on the user channel (newest first) and returns
    the first whose ``extract_plaintext(content)`` equals the supplied text. The
    transcript stores the raw LLM response; ``extract_plaintext`` strips tags to
    the same plaintext the chat UI rendered, so an exact match is reliable for
    the message the user is looking at. Returns ``None`` when nothing matches.
    """
    target = _normalise(message_text)
    if not target:
        return None

    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role, content FROM transcript "
            "WHERE channel = ? AND role = 'assistant' "
            "ORDER BY id DESC LIMIT ?",
            (_USER_CHANNEL, _RESOLVE_SCAN_LIMIT),
        )
        rows = cursor.fetchall()
        cursor.close()

    for row in rows:
        if _normalise(extract_plaintext(row[2] or "")) == target:
            return {'id': row[0], 'role': row[1], 'content': row[2]}
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@moments_bp.route("/moments", methods=["POST"])
@require_session
def create_moment():
    """Pin an assistant transcript turn as a moment.

    Accepts either an explicit integer ``transcript_id`` or a ``message_text``
    string. Only assistant turns can be pinned. Same dedupe key + ``source='pin'``
    regardless of which identifier resolved the turn.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    raw_transcript_id = data.get("transcript_id")
    message_text = (data.get("message_text") or "").strip()

    db = _get_db()
    dg = _get_dg()

    if raw_transcript_id:
        try:
            transcript_id = int(raw_transcript_id)
        except (TypeError, ValueError):
            return jsonify({"error": "transcript_id must be an integer"}), 400

        turn = _fetch_transcript_row(db, transcript_id)
        if turn is None:
            return jsonify({"error": "Transcript row not found"}), 404
        if turn['role'] != 'assistant':
            return jsonify({"error": "Only assistant turns can be pinned as moments"}), 400
    elif message_text:
        turn = _resolve_assistant_turn_by_text(db, message_text)
        if turn is None:
            return jsonify({"error": "No matching assistant message found"}), 404
        transcript_id = turn['id']
    else:
        return jsonify({"error": "transcript_id or message_text is required"}), 400

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
    return jsonify({"item": _serialize_moment(result), "duplicate": already_exists}), status


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

"""
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
from typing import TYPE_CHECKING, cast

from flask import Blueprint, g, jsonify, request

from .auth import require_session
from services.markup import extract_plaintext
from services.time_utils import parse_utc

if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue
    from services.moments_service import MomentsService
    from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"

# Prefix for the synthesized client-facing key (``moment_<transcript_id>``) and
# the forget round-trip — the key is derived, not stored, now that transcript_id
# is a real column.
_KEY_PREFIX = "moment_"

# Channel the user chat turns are written under (api/chat.py UMP path).
_USER_CHANNEL = "user"

# How many recent assistant turns to scan when resolving a pin by message text.
_RESOLVE_SCAN_LIMIT = 200

moments_bp = Blueprint("moments", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_moments() -> "MomentsService":
    if 'moments_svc' not in g:
        from services.moments_service import get_moments_service
        g.moments_svc = get_moments_service()
    return cast("MomentsService", g.moments_svc)


def _get_db() -> "DatabaseService":
    if 'moments_db' not in g:
        from services.database_service import get_shared_db_service
        g.moments_db = get_shared_db_service()
    return cast("DatabaseService", g.moments_db)


def _serialize_moment(row: "dict[str, object]") -> "dict[str, object]":
    """``key`` is synthesized from ``transcript_id`` (the client keys the forget round-trip on it)."""
    created_at = row.get('created_at')
    if created_at:
        try:
            created_at = parse_utc(cast(str, created_at)).isoformat()
        except Exception:
            pass

    transcript_id = row.get('transcript_id')
    value = row.get('content') or ''
    return {
        'id': row.get('id'),
        'transcript_id': transcript_id,
        'key': f"{_KEY_PREFIX}{transcript_id}",
        'value': value,
        'message_text': value,
        'created_at': created_at,
    }


def _fetch_transcript_row(db: "DatabaseService", transcript_id: int) -> "dict[str, object] | None":
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
    return " ".join((text or "").split())


def _resolve_assistant_turn_by_text(db: "DatabaseService", message_text: str) -> "dict[str, object] | None":
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
def create_moment() -> "ResponseReturnValue":
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    raw_transcript_id = data.get("transcript_id")
    message_text = (data.get("message_text") or "").strip()
    note = data.get("note")

    db = _get_db()
    moments = _get_moments()

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
        transcript_id = cast(int, turn['id'])
    else:
        return jsonify({"error": "transcript_id or message_text is required"}), 400

    already_exists = moments.find_by_transcript(transcript_id) is not None
    row = moments.store(transcript_id, cast(str, turn['content']), note=cast("str | None", note))

    status = 200 if already_exists else 201
    return jsonify({"item": _serialize_moment(cast("dict[str, object]", row)), "duplicate": already_exists}), status


@moments_bp.route("/moments", methods=["GET"])
@require_session
def list_moments() -> "ResponseReturnValue":
    try:
        rows = _get_moments().list_all()
        return jsonify({"items": [_serialize_moment(cast("dict[str, object]", r)) for r in rows]})
    except Exception as e:
        logger.exception(f"[MOMENTS API] list_moments error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@moments_bp.route("/moments/<int:transcript_id>/forget", methods=["POST"])
@require_session
def forget_moment(transcript_id: int) -> "ResponseReturnValue":
    try:
        if not _get_moments().delete_by_transcript(transcript_id):
            return jsonify({"error": "Moment not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception(f"[MOMENTS API] forget_moment error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@moments_bp.route("/moments/search", methods=["GET"])
@require_session
def search_moments() -> "ResponseReturnValue":
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    try:
        rows = _get_moments().search(query)
        return jsonify({"items": [_serialize_moment(cast("dict[str, object]", r)) for r in rows]})
    except Exception as e:
        logger.exception(f"[MOMENTS API] search_moments error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500

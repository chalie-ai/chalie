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

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask import g
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.markup import extract_plaintext
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.moment import Moment, MomentCreate, MomentSearch

if TYPE_CHECKING:
    from services.moments_service import MomentsService

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_NOT_FOUND = "Not found"

# Prefix for the synthesized client-facing key (``moment_<transcript_id>``) and
# the forget round-trip — the key is derived, not stored, now that transcript_id
# is a real column.
_KEY_PREFIX = "moment_"

# Channel the user chat turns are written under (api/chat.py UMP path).
_USER_CHANNEL = "user"

# How many recent assistant turns to scan when resolving a pin by message text.
_RESOLVE_SCAN_LIMIT = 200

moments_ns = Namespace("moments", description="Moment (pinned turn) operations", path="/moments")

register_dto(moments_ns, Moment, MomentCreate, MomentSearch, Error)

_M = moments_ns.models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_moments() -> "MomentsService":
    if 'moments_svc' not in g:
        from services.moments_service import get_moments_service
        g.moments_svc = get_moments_service()
    return cast("MomentsService", g.moments_svc)


def _error(message: str, status: int) -> ResponseReturnValue:
    """Build a uniform non-2xx ``Error`` body carrying its own status code."""
    return Error(error=message).model_dump(mode="json"), status


def _moment_dto(row: dict[str, object]) -> Moment:
    """Build the read DTO from a moments-service row dict.

    ``key`` is synthesized from ``transcript_id`` (the client keys the forget
    round-trip on it); ``message_text`` mirrors ``value`` — the frontend reads both.
    """
    transcript_id = cast(int, row['transcript_id'])
    value = cast(str, row.get('content') or '')
    return Moment(
        id=cast(int, row['id']),
        transcript_id=transcript_id,
        key=f"{_KEY_PREFIX}{transcript_id}",
        value=value,
        message_text=value,
        created_at=cast(datetime, row['created_at']),
    )


def _fetch_transcript_row(transcript_id: int) -> dict[str, object] | None:
    from services.transcript_service import Transcript
    rows = Transcript.by_ids([transcript_id])
    if not rows:
        return None
    return {'id': rows[0]['id'], 'role': rows[0]['role'], 'content': rows[0]['content']}


def _normalise(text: str) -> str:
    return " ".join((text or "").split())


def _resolve_assistant_turn_by_text(message_text: str) -> dict[str, object] | None:
    """Find the assistant transcript turn whose rendered text matches.

    Scans recent assistant turns on the user channel (newest first) and returns
    the first whose ``extract_plaintext(content)`` equals the supplied text. The
    transcript stores the raw LLM response; ``extract_plaintext`` strips tags to
    the same plaintext the chat UI rendered, so an exact match is reliable for
    the message the user is looking at. Returns ``None`` when nothing matches.
    """
    from services.transcript_service import Transcript
    target = _normalise(message_text)
    if not target:
        return None

    for row in reversed(Transcript.get_recent(_USER_CHANNEL, limit=_RESOLVE_SCAN_LIMIT, role='assistant')):
        if _normalise(extract_plaintext(cast(str, row['content']) or "")) == target:
            return {'id': row['id'], 'role': row['role'], 'content': row['content']}
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@moments_ns.route("")
class MomentsResource(Resource):
    @require_session
    @moments_ns.expect(_M["MomentCreate"])
    @moments_ns.response(201, "Pinned moment", model=_M["Moment"])
    @moments_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @moments_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(Moment, code=201)
    @expects(MomentCreate)
    def post(self, dto: MomentCreate) -> Moment | ResponseReturnValue:
        """Pin an assistant turn by transcript_id or message_text."""
        moments = _get_moments()
        if dto.transcript_id is not None:
            turn = _fetch_transcript_row(dto.transcript_id)
            if turn is None:
                return _error("Transcript row not found", 404)
            if turn['role'] != 'assistant':
                return _error("Only assistant turns can be pinned as moments", 422)
            transcript_id = dto.transcript_id
        else:
            turn = _resolve_assistant_turn_by_text(dto.message_text or "")
            if turn is None:
                return _error("No matching assistant message found", 404)
            transcript_id = cast(int, turn['id'])

        row = moments.store(transcript_id, cast(str, turn['content']), note=dto.note)
        return _moment_dto(cast("dict[str, object]", row))

    @require_session
    @moments_ns.response(200, "All moments", model=_M["Moment"])
    @moments_ns.response(500, _ERR_INTERNAL, model=_M["Error"])
    @responds(Moment, code=200)
    def get(self) -> list[Moment] | ResponseReturnValue:
        """List every pinned moment, newest first."""
        try:
            return [_moment_dto(cast("dict[str, object]", r)) for r in _get_moments().list_all()]
        except Exception as exc:
            logger.exception("[MOMENTS API] list error: %s", exc)
            return _error(_ERR_INTERNAL, 500)


@moments_ns.route("/<int:transcript_id>/forget")
class MomentForgetResource(Resource):
    @require_session
    @moments_ns.param("transcript_id", "Transcript row id of the moment to forget")
    @moments_ns.response(204, "Forgotten")
    @moments_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(code=204)
    def post(self, transcript_id: int) -> None | ResponseReturnValue:
        """Forget (un-pin) a moment keyed by its transcript row id."""
        if not _get_moments().delete_by_transcript(transcript_id):
            return _error("Moment not found", 404)
        return None


@moments_ns.route("/search")
class MomentsSearchResource(Resource):
    @require_session
    @moments_ns.response(200, "Matching moments", model=_M["Moment"])
    @moments_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(Moment, code=200)
    @expects(MomentSearch, source="args")
    def get(self, dto: MomentSearch) -> list[Moment] | ResponseReturnValue:
        """Lexical + semantic search across pinned moments."""
        return [_moment_dto(cast("dict[str, object]", r)) for r in _get_moments().search(dto.q)]
"""Threads namespace — the ``thread`` REST resource, addressed by turn_id.

Routes (mounted under ``/api``):
  GET  /api/threads            — collapsed thread feed (id + gist/preview/last activity).
  GET  /api/threads/batch      — many turn blocks in one round-trip (repeated ``id[]``).
  GET    /api/thread/<turn_id> — one turn's full block (the WS-refetch + expand read).
  POST   /api/thread/<turn_id> — send: ``-1`` creates a new thread (allocates a fresh
                                 turn_id); a real id replies into that thread.
  DELETE /api/thread/<turn_id> — interrupt the running turn (the FE stop button).

Reads are GET, sends are POST, interrupt is DELETE; turn_id is path-only, never in
the body. The path speaks the MessageProcessor constructor's own language — ``-1``
is its unset sentinel, so the send needs no mapping layer, and a supplied id that
names no existing turn is rejected by the constructor (``Invalid turn_id
specified``). Both POST and DELETE return the turn's ``turn_execution`` row (its
DB-backed lifecycle record) — the turn_id is allocated synchronously (the request
thread opens the row before responding) so the FE holds it inline, no WS round-trip.
Live output then surfaces via the ``updated`` signal → REST pull, plus a
``turn_execution`` WS frame on every lifecycle flip (working → completed / cancelled
/ crashed). The stop button calls DELETE with that same turn_id — it flips
``cancel_requested`` on the turn's open execution row, which the running turn polls
cooperatively; an idle/finished turn_id is a harmless ``no_active_turn`` ack. ``type``
(the ProcessorConfig identity — the only surface the FE speaks) defaults to ``user``
and rides every read/write, resolving to its transcript channel server-side so the
surface can address other configs (e.g. the scheduler) without knowing channels.
"""

import logging
import sqlite3
from typing import cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from configs.enums.config_type import ConfigTypeEnum
from services.database import Database
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.rich_media_parser import parse as _parse_rich_media
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.attachment import Attachment
from .dto.boundary import error
from .dto.chip import Chip
from .dto.message import Message
from .dto.segment import Segment
from .dto.subagent import Interrupted
from .dto.thread import (
    ThreadBatch, ThreadFeed, ThreadFeedQuery, ThreadSendRequest, ThreadSummary, TurnBlock,
)
from .dto.turn_execution import TurnExecutionDTO

logger = logging.getLogger(__name__)

threads_ns = Namespace("threads", description="Thread feed and per-turn blocks", path="/api")

register_dto(
    threads_ns,
    ThreadFeedQuery, ThreadSummary, ThreadFeed, TurnBlock, ThreadBatch, ThreadSendRequest, TurnExecutionDTO,
    Message, Attachment, Chip, Segment, Interrupted, Error,
)
_T = threads_ns.models

_TYPE = "user"
_THREAD_EXCLUDE = ("subagent_return",)
_SEARCH_LIMIT = 5  # search collapses the feed to a single capped page (no pagination)
_MAX_FILES = 10
_PREVIEW_CHARS = 200
_COMPACTOR_TOOL = "chat_history_compactor"
_MSG_REQUIRED = "message required"
_FILE_PLACEHOLDER = "[File attached]"


# ── Row → message projection ──────────────────────────────────────────────────


def _fetch_tool_calls_for_transcripts(transcript_ids: list[int]) -> list[dict[str, object]]:
    """Tool-call rows anchored to any of the given transcript ids, ordered by id.

    Rows within the 7-day retention window carry the rich-media payloads the
    parser uses to pair span tags with cards, and the act_summary the frontend
    prints as the blue box under each tool chip. Ordered by ``id`` (not
    ``created_at``, which is 1-second granular and ambiguous within a batch) so
    multiple calls on one row keep their emission order.
    """
    if not transcript_ids:
        return []
    tc_rows = ToolCall.by_transcripts(transcript_ids)
    return [
        {
            "transcript_id": tc.transcript_id,
            "tool_name": tc.tool_name,
            "params": tc.params,
            "result": tc.result or "",
            "summary": tc.summary or "",
            "created_at": tc.created_at,
            "ended_at": tc.ended_at,
            "state": tc.state,
        }
        for tc in tc_rows
    ]



def _fetch_attachments_for_transcripts(conn: sqlite3.Connection, transcript_ids: list[int]) -> dict[int, list[dict[str, object]]]:
    """Soft-deleted docs are filtered out, so a removed file silently does not
    render. Each attachment carries the inline-serving
    ``/api/documents/<id>/preview`` URL.
    """
    if not transcript_ids:
        return {}
    placeholders = ",".join("?" * len(transcript_ids))
    rows = conn.execute(
        f"SELECT td.transcript_id, d.id, d.original_name, d.mime_type "
        f"FROM transcript_docs td JOIN documents d ON d.id = td.doc_id "
        f"WHERE td.transcript_id IN ({placeholders}) AND d.deleted_at IS NULL "
        f"ORDER BY td.rowid",
        tuple(transcript_ids),
    ).fetchall()
    by_id: dict[int, list[dict[str, object]]] = {}
    for tid, doc_id, name, mime in rows:
        mime = mime or ""
        by_id.setdefault(tid, []).append({
            "doc_id": doc_id,
            "filename": name,
            "mime_type": mime,
            "is_image": mime.startswith("image/"),
            "url": f"/api/documents/{doc_id}/preview",
        })
    return by_id


def _group_calls_by_transcript(calls: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    """Bucket flat tool-call rows by the transcript row that owns each call."""
    by_transcript: dict[int, list[dict[str, object]]] = {}
    for c in calls:
        by_transcript.setdefault(cast("int", c["transcript_id"]), []).append(c)
    return by_transcript


def _base_message(r: dict[str, object]) -> dict[str, object]:
    """The role/content/timestamp shape common to every projected message."""
    return {
        "id": str(cast("int", r['id'])),
        "role": r['role'],
        "content": r['content'] or "",
        "timestamp": format_date(cast("str", r['created_at']), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
        "turn_id": r['turn_id'],
    }


def _apply_user_fields(msg: dict[str, object], r: dict[str, object], attachments_by_id: dict[int, list[dict[str, object]]]) -> None:
    attachments = attachments_by_id.get(cast("int", r['id']))
    if attachments:
        msg["attachments"] = attachments


def _tool_call_chips(own_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    """Chips for whatever transcript row anchors these calls — the user input row
    or an assistant text row alike. A step-1 tool-only call (the commonest shape)
    anchors to the user row, so chips are role-agnostic (the ratified feed vision:
    tool calls render under WHATEVER row anchors them). Each chip carries the
    persisted ``state`` + ``ended_at`` so a refetched error stays an error pill,
    not a downgraded neutral chip."""
    return [
        {
            "tool_name": c["tool_name"],
            "summary": c["summary"],
            "state": c["state"],
            "ended_at": c["ended_at"],
        }
        for c in own_calls
        if c["tool_name"] != _COMPACTOR_TOOL
    ]


def _apply_assistant_fields(
    msg: dict[str, object],
    r: dict[str, object],
    turn_calls: list[dict[str, object]],
) -> None:
    content = msg["content"]
    segments = _parse_rich_media(str(content), turn_calls)
    if not segments and content:
        segments = [{"type": "text", "content": content}]
    msg["segments"] = segments


def _rows_to_messages(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project raw transcript rows (oldest-first) into the conversation message
    shape — attachments for user rows, rich-media segments for assistant rows,
    and tool-call chips on WHATEVER row anchors them (user input row included).
    Shared by thread-list and thread-expand."""
    messages: list[dict[str, object]] = []
    conn = Database.conn()
    attachments_by_id = _fetch_attachments_for_transcripts(
        conn, [cast("int", r['id']) for r in rows if r['role'] == 'user']
    )

    # Resolve turn scope from EVERY row, not just assistant rows: a step-1
    # tool-only call anchors to the user input row, and a turn whose only row
    # so far is that input row has zero assistant rows — keying off assistant
    # ids alone would resolve to an empty scope and silently drop its chips.
    turn_scope_ids = Transcript.turn_scope_ids(
        [cast("int", r['id']) for r in rows]
    )
    # One query for all tool calls across the whole turn scope (chips + segments).
    turn_calls = _fetch_tool_calls_for_transcripts(turn_scope_ids)
    # Group by transcript_id for per-row chip lookup.
    calls_by_transcript = _group_calls_by_transcript(turn_calls)

    for r in rows:
        msg = _base_message(r)
        chips = _tool_call_chips(calls_by_transcript.get(cast("int", r['id']), []))
        if chips:
            msg["tool_calls"] = chips
        if r['role'] == 'user':
            _apply_user_fields(msg, r, attachments_by_id)
        else:
            _apply_assistant_fields(msg, r, turn_calls)
        messages.append(msg)

    return messages


def _bulk_gists(channel: str, turn_ids: list[int]) -> dict[int, str]:
    """Thread labels for a batch of turn_ids, reached through an inert MP's
    ``gist_service`` (cut-over §2.7-style: :meth:`GistService.bulk_get` takes
    ``channel``/``turn_ids`` explicitly and never touches ``mp.config``, so any
    config satisfies construction — zero side-effects, I2)."""
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

    mp = MessageProcessor(ConfigTypeEnum.get_by_type(_TYPE))
    return mp.gist_service.bulk_get(channel, turn_ids)


def serialize_turn(channel: str, turn_id: int) -> dict[str, object]:
    """The single turn-block getter — the REST single/batch reads and the WS
    refetch all flow through here, so one fetch fully determines a turn's render
    with no signal memory.

    Returns the WHOLE turn (no settle0 floor) projected into messages, with every
    row PAST settle0 tagged ``thread_message: true`` — the reply continuation the
    main spine drops (it renders only through settle0) and whose mere presence
    makes the turn a thread (the feed shows the opener). The collapsed-feed
    metadata (gist, preview, last activity) and the turn-level render state
    (``working`` — an open ``turn_executions`` row for this (channel, turn_id),
    i.e. a currently in-flight execution, not merely "never settled" — and
    ``duration_ms``, derived from the row span) are folded in."""
    from services.time_utils import parse_utc

    rows = Transcript.by_turn(channel, turn_id)
    messages = _rows_to_messages(rows)

    settle = Transcript.settle0(channel, turn_id)
    for m in messages:
        if settle is not None and int(cast("str", m["id"])) > settle:
            m["thread_message"] = True

    duration_ms = 0
    if len(rows) >= 2:
        span = parse_utc(cast("str", rows[-1]["created_at"])) - parse_utc(cast("str", rows[0]["created_at"]))
        duration_ms = int(span.total_seconds() * 1000)

    return {
        "turn_id": turn_id,
        "gist": _bulk_gists(channel, [turn_id]).get(turn_id),
        "preview": cast("str", rows[0]["content"] or "")[:_PREVIEW_CHARS] if rows else "",
        "last_activity_at": rows[-1]["created_at"] if rows else None,
        "working": TurnExecution.open_turn(channel, turn_id) is not None,
        "duration_ms": duration_ms,
        "messages": messages,
    }


def _thread_summaries(threads: list[dict[str, object]], channel: str) -> list[ThreadSummary]:
    """Project raw recent_threads rows into feed DTOs, bulk-injecting each thread's
    one-sentence gist (scoped to ``channel``). The internal latest-row-id recency
    key is dropped here."""
    if not threads:
        return []

    turn_ids = [cast("int", t["turn_id"]) for t in threads if t.get("turn_id") is not None]
    gists = _bulk_gists(channel, turn_ids)
    return [
        ThreadSummary(
            turn_id=cast("int | None", t.get("turn_id")),
            last_activity_at=cast("str | None", t.get("last_activity_at")),
            preview=cast("str", t.get("preview") or ""),
            row_count=cast("int", t["row_count"]),
            gist=gists.get(cast("int", t["turn_id"])) if t.get("turn_id") is not None else None,
            working=bool(t.get("working", False)),
        )
        for t in threads
    ]


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@threads_ns.route("/threads")
class ThreadsResource(Resource):
    @require_session
    @threads_ns.doc(
        description="The collapsed thread feed, most-recently-active first. A non-empty "
        "``q`` switches to keyword search (capped, unpaginated); otherwise limit/offset "
        "paginate.",
    )
    @threads_ns.response(200, "Thread feed", model=_T["ThreadFeed"])
    @threads_ns.response(422, "Validation failed", model=_T["Error"])
    @responds(ThreadFeed)
    @expects(ThreadFeedQuery, source="args")
    def get(self, dto: ThreadFeedQuery) -> ThreadFeed:
        """List threads with collapsed metadata (gist, preview, last activity)."""
        limit, offset = (_SEARCH_LIMIT, 0) if dto.q else (dto.limit, dto.offset)
        channel = ConfigTypeEnum.get_by_type(dto.type).channel
        threads, has_more, threads_returned = Transcript.recent_threads(
            channel, exclude_roles=_THREAD_EXCLUDE, limit=limit, offset=offset, query=dto.q,
        )
        return ThreadFeed(
            threads=_thread_summaries(threads, channel),
            has_more=has_more,
            threads_returned=threads_returned,
        )


@threads_ns.route("/threads/batch")
class ThreadsBatchResource(Resource):
    @require_session
    @threads_ns.param("id[]", "Turn ids to fetch (repeatable)", _in="query", type="integer", action="append")
    @threads_ns.param("type", "Config type (default: user)", _in="query", type="string")
    @threads_ns.response(200, "Turn blocks", model=_T["ThreadBatch"])
    @responds(ThreadBatch)
    def get(self) -> ThreadBatch:
        """Many turn blocks in one round-trip — a pure concatenation of the
        single-turn getter over the requested ids. Non-numeric ids are ignored."""
        channel = ConfigTypeEnum.get_by_type(request.args.get("type", _TYPE)).channel
        ids = [int(t) for t in request.args.getlist("id[]") if t.isdigit()]
        return ThreadBatch(blocks=[TurnBlock.model_validate(serialize_turn(channel, t)) for t in ids])


# werkzeug's converter-arg grammar has no negative literals, so min=-1 can't be
# expressed here; signed alone admits -1, and the MessageProcessor constructor
# rejects every other id that names no existing turn.
@threads_ns.route("/thread/<int(signed=True):turn_id>")
class ThreadItemResource(Resource):
    @require_session
    @threads_ns.param("turn_id", "Turn id")
    @threads_ns.param("type", "Config type (default: user)", _in="query", type="string")
    @threads_ns.response(200, "Turn block", model=_T["TurnBlock"])
    @responds(TurnBlock)
    def get(self, turn_id: int) -> TurnBlock:
        """One turn's full block — the WS-refetch + expand-on-click read. ``type``
        defaults to ``user``."""
        return TurnBlock.model_validate(serialize_turn(ConfigTypeEnum.get_by_type(request.args.get("type", _TYPE)).channel, turn_id))

    @require_session
    @threads_ns.param("turn_id", "Turn id (-1 creates a new thread)")
    @threads_ns.param("text", "Message text", _in="formData", type="string", required=True)
    @threads_ns.param("type", "Config type (default: user)", _in="formData", type="string")
    @threads_ns.param("files", "Attachments (multipart, repeatable, max 10)", _in="formData", type="file")
    @threads_ns.response(200, "Send accepted", model=_T["TurnExecutionDTO"])
    @threads_ns.response(422, "Empty message and no files", model=_T["Error"])
    @responds(TurnExecutionDTO, code=200)
    @expects(ThreadSendRequest, source="form")
    def post(self, turn_id: int, dto: ThreadSendRequest) -> "TurnExecutionDTO | ResponseReturnValue":
        """Send — the one chat-dispatch chokepoint. ``-1`` is the MessageProcessor's
        own unset sentinel and starts a new thread (a fresh turn_id is allocated
        synchronously); a real id replies INTO that thread (FORK view; the id is
        echoed back). A supplied id that names no existing turn is rejected by the
        constructor. Returns 200 with the turn's freshly-opened turn_execution row —
        the constructor opens it synchronously so the FE holds the handle with no WS
        round-trip, and live output then surfaces via the ``updated`` signal → REST
        pull plus ``turn_execution`` lifecycle frames."""
        from api.chat import _stage_chat_uploads  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        attachments = _stage_chat_uploads(cast("list[object]", request.files.getlist("files")[:_MAX_FILES]))
        text = dto.text.strip()
        if not text and not attachments:
            return error(_MSG_REQUIRED, 422)
        if not text:
            text = _FILE_PLACEHOLDER

        # ``type`` selects the ProcessorConfig directly (user → UserConfig,
        # scheduled → ScheduledConfig, its own growing thread on the schedule channel).
        # forked-ness is derived internally by the MessageProcessor from the turn_id.
        config = ConfigTypeEnum.get_by_type(dto.type)
        mp = MessageProcessor.process(config, text, {"attachments": attachments}, turn_id)
        if mp.execution is None:
            return error("Failed to open turn execution", 500)
        return TurnExecutionDTO.model_validate(mp.execution.to_dict())

    @require_session
    @threads_ns.param("turn_id", "Turn id")
    @threads_ns.param("type", "Config type (default: user)", _in="query", type="string")
    @threads_ns.response(200, "Interrupt ack", model=_T["TurnExecutionDTO"])
    @threads_ns.response(400, "Invalid type", model=_T["Error"])
    @responds(code=200)
    def delete(self, turn_id: int) -> "TurnExecutionDTO | Interrupted | ResponseReturnValue":
        """Interrupt the running turn for this turn_id — the FE stop button's call.
        Flips ``cancel_requested`` on the turn's open execution row; the turn polls
        it cooperatively, stops its step loop and stamps itself cancelled — every
        row it already wrote stays, nothing is deleted. The
        id is the same one the send response handed the surface, so no body beyond
        the turn_id is needed; ``type`` resolves the channel the row was opened
        under (turn_id is only unique per channel). An idle/finished turn_id (no
        open row on that channel) is a harmless ``no_active_turn`` ack."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        try:
            config = ConfigTypeEnum.get_by_type(request.args.get("type", _TYPE))
        except ValueError:
            return error("Invalid type", 400)
        mp = MessageProcessor(config, turn_id)  # inert (I2): 0 db, 0 ws at construction
        execution = mp.turn_execution_service.cancel()
        if execution is None:
            return Interrupted(reason="no_active_turn")
        logger.info("[Threads API] cancel requested for turn %s channel=%s", turn_id, config.channel)
        return TurnExecutionDTO.model_validate(execution.to_dict())

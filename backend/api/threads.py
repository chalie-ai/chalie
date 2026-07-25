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
import uuid
from collections.abc import Sequence
from typing import cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from configs.channels import config_for
from services.turn_serializer_service import get_service as _serializer
from models.transcript import Transcript
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


def _stage_chat_uploads(files: Sequence[object]) -> list[object]:
    """Returns temp paths that _seed_turn_zero feeds to FileParserService.ingest.

    The files are saved to tmp-storage, then ingested (extracted, copied flat
    to data/documents/uploads/, indexed) by FileParserService during turn-zero
    seeding. No file blob ever reaches the act-trail; only the extracted text
    or vision description does.
    """
    from services.filename_utils import safe_filename  # noqa: PLC0415
    from services.tmp_storage import new_tmp_path  # noqa: PLC0415

    paths: list[object] = []
    for f in files:
        if not f or not getattr(f, 'filename', None):
            continue
        name = safe_filename(getattr(f, 'filename')) or "attachment"
        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{name}")
        getattr(f, 'save')(tmp_path)
        paths.append(tmp_path)
    return paths


_TYPE = "user"
_THREAD_EXCLUDE = ("subagent_return",)
_SEARCH_LIMIT = 5  # search collapses the feed to a single capped page (no pagination)
_MAX_FILES = 10
_MSG_REQUIRED = "message required"
_FILE_PLACEHOLDER = "[File attached]"


def _thread_summaries(threads: list[dict[str, object]], channel: str, config_type: str) -> list[ThreadSummary]:
    """Project raw recent_threads rows into feed DTOs, bulk-injecting each thread's
    one-sentence gist (scoped to ``channel``). ``config_type`` is the ConfigType
    identity string ``channel`` was resolved from and is stamped onto every
    summary as ``type``, so a client refetching a thread found via the feed can
    carry the right type forward. The internal latest-row-id recency key is
    dropped here."""
    if not threads:
        return []

    turn_ids = [cast("int", t["turn_id"]) for t in threads if t.get("turn_id") is not None]
    gists = _serializer().bulk_gists(channel, turn_ids)
    return [
        ThreadSummary(
            turn_id=cast("int | None", t.get("turn_id")),
            last_activity_at=cast("str | None", t.get("last_activity_at")),
            preview=cast("str", t.get("preview") or ""),
            row_count=cast("int", t["row_count"]),
            gist=gists.get(cast("int", t["turn_id"])) if t.get("turn_id") is not None else None,
            working=bool(t.get("working", False)),
            type=config_type,
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
        channel = config_for(dto.type).channel
        threads, has_more, threads_returned = Transcript.recent_threads(
            channel, exclude_roles=_THREAD_EXCLUDE, limit=limit, offset=offset, query=dto.q,
        )
        return ThreadFeed(
            threads=_thread_summaries(threads, channel, dto.type),
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
        config_type = request.args.get("type", _TYPE)
        channel = config_for(config_type).channel
        ids = [int(t) for t in request.args.getlist("id[]") if t.isdigit()]
        return ThreadBatch(
            blocks=[TurnBlock.model_validate(_serializer().serialize(channel, t, config_type)) for t in ids]
        )


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
        config_type = request.args.get("type", _TYPE)
        channel = config_for(config_type).channel
        return TurnBlock.model_validate(_serializer().serialize(channel, turn_id, config_type))

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
        config = config_for(dto.type)
        mp = MessageProcessor.process(config, text, {"attachments": attachments, "thinking_level": dto.thinking_level}, turn_id)
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
        Stamps the turn's open execution row CANCELLED right here, synchronously
        (:meth:`TurnExecutionService.cancel` — the single authority for a turn's
        terminal state) and broadcasts that row on the same lifecycle WS channel
        every other state flip uses, so the surface's 'cancelled' frame fires the
        instant this call returns rather than waiting on the still-running step
        loop. The loop itself keeps running the in-flight provider call to
        completion — there is no mid-flight abort — but its own cancel checkpoint
        (§ ``MessageProcessor._step``) observes ``cancel_requested`` and discards
        the response instead of storing it; every row the turn already wrote before
        that point stays, nothing is deleted. The id is the same one the send
        response handed the surface, so no body beyond the turn_id is needed;
        ``type`` resolves the channel the row was opened under (turn_id is only
        unique per channel). An idle/finished turn_id (no open row on that
        channel) is a harmless ``no_active_turn`` ack."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        try:
            config = config_for(request.args.get("type", _TYPE))
        except ValueError:
            return error("Invalid type", 400)
        mp = MessageProcessor(config, turn_id)  # inert (I2): 0 db, 0 ws at construction
        execution = mp.turn_execution_service.cancel()
        if execution is None:
            return Interrupted(reason="no_active_turn")
        logger.info("[Threads API] cancel requested for turn %s channel=%s", turn_id, config.channel)
        return TurnExecutionDTO.model_validate(execution.to_dict())


@threads_ns.route("/thread/<int(signed=True):turn_id>/thinking-level")
class ThinkingLevelResource(Resource):
    @require_session
    @threads_ns.param("turn_id", "Turn id", _in="path", type="integer")
    @threads_ns.param("type", "Config type (default: user)", _in="query", type="string")
    @threads_ns.response(200, "Current thinking level")
    def get(self, turn_id: int) -> dict[str, str]:
        """Return the thread's current thinking level — the literal value
        persisted on the input row (auto/medium/high), or 'auto' when none is
        set. ``turn_id == -1`` (new/spine thread) spans every turn of the
        channel; a real id reads that turn's own row."""
        config_type = request.args.get("type", _TYPE)
        channel = config_for(config_type).channel
        level = Transcript.latest_thinking_level(channel, turn_id if turn_id != -1 else None)
        return {"level": level if level is not None else "auto"}

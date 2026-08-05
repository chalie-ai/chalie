"""Turn serialization service — projects transcript rows into the conversation
message shape used by the thread feed, thread-expand, and thread-batch endpoints.

Owns turn serialisation outright, so the routes stay thin controllers over an
Endpoint/Action contract and the business logic lives in a service class that
non-HTTP callers can reuse without spinning up Flask.

Public API:
* :meth:`serialize` — the single turn-block getter (used by the REST single/
  batch reads and the WS refetch)
* :meth:`bulk_gists` — thread labels for a batch of turn_ids (used by the
  collapsed feed)
"""

from __future__ import annotations

import logging
import mimetypes
import os
import sqlite3
import urllib.parse
from typing import cast

from services.database import Database
from services.file_mapper_service import FileMapperService
from services.locale_service import CHAT_DAY_FMT, CHAT_TIMESTAMP_FMT, format_date
from services.rich_media_parser import parse as _parse_rich_media
from configs.channels import config_for
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.transcript_thinking import TranscriptThinking
from models.turn_execution import TurnExecution
from models.voice_transcript import VoiceTranscript

logger = logging.getLogger(__name__)

_TYPE = "user"
_PREVIEW_CHARS = 200
_COMPACTOR_TOOL = "chat_history_compactor"


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
    """Attachment rows for the given transcript ids, read from ``transcript_files``.

    Each row carries a path relative to ``FileMapperService.get_documents_path()``.
    The absolute path is reconstructed, and the row is silently dropped when the
    file no longer exists on disk — so a removed file does not render, matching
    the prior soft-delete semantics.

    Each attachment carries the inline-serving ``/api/files/preview/<relpath>`` URL.
    """
    if not transcript_ids:
        return {}
    placeholders = ",".join("?" * len(transcript_ids))
    rows = conn.execute(
        f"SELECT transcript_id, path FROM transcript_files "
        f"WHERE transcript_id IN ({placeholders}) "
        f"ORDER BY rowid",
        tuple(transcript_ids),
    ).fetchall()
    by_id: dict[int, list[dict[str, object]]] = {}
    for tid, relpath in rows:
        abs_path = os.path.join(str(FileMapperService.get_documents_path()), relpath)
        if not os.path.exists(abs_path):
            continue
        mime = mimetypes.guess_type(relpath)[0] or ""
        by_id.setdefault(cast("int", tid), []).append({
            "filename": os.path.basename(relpath),
            "mime_type": mime,
            "is_image": mime.startswith("image/"),
            "url": f"/api/files/preview/{urllib.parse.quote(relpath)}",
        })
    return by_id


def _group_calls_by_transcript(calls: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    """Bucket flat tool-call rows by the transcript row that owns each call."""
    by_transcript: dict[int, list[dict[str, object]]] = {}
    for c in calls:
        by_transcript.setdefault(cast("int", c["transcript_id"]), []).append(c)
    return by_transcript


def _base_message(r: dict[str, object]) -> dict[str, object]:
    """The role/content/timestamp/day shape common to every projected message."""
    return {
        "id": str(cast("int", r['id'])),
        "role": r['role'],
        "content": r['content'] or "",
        "timestamp": format_date(cast("str", r['created_at']), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
        "day": format_date(cast("str", r['created_at']), CHAT_DAY_FMT, for_ui=True) or "",
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
    cycle_calls: list[dict[str, object]],
) -> None:
    content = msg["content"]
    segments = _parse_rich_media(str(content), cycle_calls)
    if not segments and content:
        segments = [{"type": "text", "content": content}]
    msg["segments"] = segments


def _voice_states(rows: list[dict[str, object]]) -> dict[int, str]:
    """The recorded speech pre-synthesis outcome per settled assistant row, in
    one query. ``ready`` once the audio is stored, ``failed`` once the pipeline
    gave up; a settled row missing from this map has no outcome yet — history
    predating the feature, or a job lost to a restart — and the client's first
    speaker press starts the pipeline through the playback route."""
    settled_ids = [
        cast("int", r["id"]) for r in rows
        if r["role"] == "assistant" and r["settled"]
    ]
    return {
        row.transcript_id: "ready" if row.file_path is not None else "failed"
        for row in VoiceTranscript.for_transcripts(settled_ids)
    }


def _thinking_states(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    """Thinking traces per transcript row, in one batch query.

    Traces anchor to WHATEVER row drove the tools — the user input row for
    step-1 tool-only calls, or an assistant row for later steps — so the id
    list includes ALL rows, not just assistant rows. Returns a map from
    transcript_id to a list of trace dicts (traces in row order); rows with no
    thinking rows are absent from the map so the serializer can omit the
    field entirely.
    """
    all_ids = [cast("int", r["id"]) for r in rows]
    if not all_ids:
        return {}
    rows_data = TranscriptThinking.for_transcripts(all_ids)
    by_id: dict[int, list[dict[str, object]]] = {}
    for row in rows_data:
        by_id.setdefault(row.transcript_id, []).append({
            "thinking_trace": row.thinking_trace,
            "duration_ms": row.duration_ms,
            "tokens": row.tokens,
        })
    return by_id


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
    # One query for all tool calls across the whole turn scope (chips per row,
    # segments per cycle — see the cycle-scoping note below).
    turn_calls = _fetch_tool_calls_for_transcripts(turn_scope_ids)
    # Group by transcript_id for per-row chip lookup.
    calls_by_transcript = _group_calls_by_transcript(turn_calls)

    # Rich-media spans resolve PER ACT CYCLE, not per turn. A turn_id spans the
    # whole thread — many user requests — and each request runs its own ACT loop
    # with its own rich-media ordinal counter that restarts at 1 (DispatchService
    # is constructed per request). So two image_preview calls in the same turn
    # both carry ``image_preview_1``; resolving an assistant row's span against
    # the FLAT turn scope returns the turn's FIRST such call for every card
    # (every image card rendered the first image). Scope each assistant
    # row to its cycle instead: reset at every user row (the cycle boundary),
    # accumulate any assistant-anchored calls, and resolve the span within that
    # window — the one scope where the per-request ordinal is unique.
    voice_states = _voice_states(rows)
    thinking_states = _thinking_states(rows)
    cycle_calls: list[dict[str, object]] = []
    for r in rows:
        msg = _base_message(r)
        own = calls_by_transcript.get(cast("int", r['id']), [])
        chips = _tool_call_chips(own)
        if chips:
            msg["tool_calls"] = chips
        own_thinking = thinking_states.get(cast("int", r['id']))
        if own_thinking is not None:
            msg["thinking"] = {
                "traces": [t["thinking_trace"] for t in own_thinking],
                "duration_ms": sum(cast("int", t["duration_ms"]) for t in own_thinking),
                "tokens": sum(cast("int", t["tokens"]) for t in own_thinking),
            }
        if r['role'] == 'user':
            cycle_calls = list(own)
            _apply_user_fields(msg, r, attachments_by_id)
        else:
            cycle_calls = cycle_calls + own
            _apply_assistant_fields(msg, cycle_calls)
            msg["settled"] = bool(r['settled'])
            msg["voice_state"] = voice_states.get(cast("int", r['id']))
        messages.append(msg)

    return messages


def _bulk_gists(channel: str, turn_ids: list[int]) -> dict[int, str]:
    """Thread labels for a batch of turn_ids, reached through an inert MP's
    ``gist_service`` (cut-over §2.7-style: :meth:`GistService.bulk_get` takes
    ``channel``/``turn_ids`` explicitly and never touches ``mp.config``, so any
    config satisfies construction — zero side-effects, I2)."""
    from controllers.message_processor import MessageProcessor

    mp = MessageProcessor(config_for(_TYPE))
    return mp.gist_service.bulk_get(channel, turn_ids)


def _drop_trailing_cancelled_orphan(
    rows: list[dict[str, object]], latest: TurnExecution | None,
) -> list[dict[str, object]]:
    """A cancel observed at :meth:`MessageProcessor._step`'s checkpoint
    discards the in-flight response before any reply row is stored (§2.7),
    leaving one or more trailing, reply-less user rows behind — filtered out
    here so a cancelled exchange never surfaces as a dangling, unanswered
    bubble on refetch/refresh. Normally the FE gates this to exactly one
    trailing user row, but a direct API/automation POST into an open
    turn_id can append a second (or third) reply-less user row before the
    cancel lands, so every CONSECUTIVE trailing user row is stripped, not
    just the last — the loop stops the instant it hits a non-user row,
    since a cancel observed mid-turn, after real assistant/tool content was
    already written, leaves that content in place per existing doctrine
    ('every row it already wrote stays'). Rows are dropped only when the
    turn's own most recent execution (``latest``, fetched once by the caller)
    actually ended cancelled, never on role alone."""
    if not rows or rows[-1]["role"] != "user":
        return rows
    if latest is None or latest.state != TurnExecution.CANCELLED:
        return rows
    cutoff = len(rows)
    while cutoff > 0 and rows[cutoff - 1]["role"] == "user":
        cutoff -= 1
    return rows[:cutoff]


class TurnSerializerService:
    """Turn serialization service — projects transcript rows into the
    conversation message shape used by the thread feed, thread-expand, and
    thread-batch endpoints.

    ``config_type`` is the ConfigType identity string the caller resolved
    ``channel`` from (the same value passed to :func:`config_for`) and is
    stamped onto the returned block as ``type`` — the one piece of state
    a refetching client needs to keep addressing the right channel, since
    ``channel`` alone isn't recoverable from a turn_id (a thread refetched
    without the right ``type`` would otherwise silently resolve to the wrong
    channel and render as an empty block).

    Returns the WHOLE turn (no floor) projected into messages, with every row
    from the turn's SECOND user-role row onward tagged ``thread_message: true``
    — the reply continuation the main spine drops (it renders only the opener)
    and whose mere presence makes the turn a thread (the feed shows the
    opener). The opener is one user row plus every assistant row that follows
    it (interim "let me check…" rows AND the final settled reply alike) up to
    (not including) the next user row a reply appends — so the boundary keys
    on the second user row's id, the one thing about a reply that is both
    structural (a fresh input row, not a column flip) and immutable once
    written. Neither ``Transcript.settle0`` NOR "first assistant row" work
    here: settle0 is deliberately mutable (a reply's own tool activity
    unsettles the ORIGINAL exchange's row via ``TranscriptService.unsettle()``,
    so re-querying it retroactively erases every row's tag), and "first
    assistant row" wrongly tags a single-exchange turn's own final reply once
    an interim assistant row (a tool-using turn's "let me check…" row) precedes
    it. The collapsed-feed metadata (gist, preview, last activity) and the
    turn-level render state (``working`` — an open ``turn_executions`` row for
    this (channel, turn_id), i.e. a currently in-flight execution, not merely
    "never settled" — and ``duration_ms``, derived from the row span) are
    folded in."""

    def serialize(self, channel: str, turn_id: int, config_type: str = _TYPE) -> dict[str, object]:
        """The single turn-block getter — the REST single/batch reads and the WS
        refetch all flow through here, so one fetch fully determines a turn's render
        with no signal memory.

        ``config_type`` is the ConfigType identity string the caller resolved
        ``channel`` from (the same value passed to :func:`config_for`)
        and is stamped onto the returned block as ``type`` — the one piece of state
        a refetching client needs to keep addressing the right channel, since
        ``channel`` alone isn't recoverable from a turn_id (a thread refetched
        without the right ``type`` would otherwise silently resolve to the wrong
        channel and render as an empty block).

        Returns the WHOLE turn (no floor) projected into messages, with every row
        from the turn's SECOND user-role row onward tagged ``thread_message: true``
        — the reply continuation the main spine drops (it renders only the opener)
        and whose mere presence makes the turn a thread (the feed shows the
        opener). The opener is one user row plus every assistant row that follows
        it (interim "let me check…" rows AND the final settled reply alike) up to
        (not including) the next user row a reply appends — so the boundary keys
        on the second user row's id, the one thing about a reply that is both
        structural (a fresh input row, not a column flip) and immutable once
        written. Neither ``Transcript.settle0`` NOR "first assistant row" work
        here: settle0 is deliberately mutable (a reply's own tool activity
        unsettles the ORIGINAL exchange's row via ``TranscriptService.unsettle()``,
        so re-querying it retroactively erases every row's tag), and "first
        assistant row" wrongly tags a single-exchange turn's own final reply once
        an interim assistant row (a tool-using turn's "let me check…" row) precedes
        it. The collapsed-feed metadata (gist, preview, last activity) and the
        turn-level render state (``working`` — an open ``turn_executions`` row for
        this (channel, turn_id), i.e. a currently in-flight execution, not merely
        "never settled" — and ``duration_ms``, derived from the row span) are
        folded in."""
        from services.time_utils import parse_utc

        latest = TurnExecution.latest(channel, turn_id)
        rows = Transcript.by_turn(channel, turn_id)
        rows = _drop_trailing_cancelled_orphan(rows, latest)
        messages = _rows_to_messages(rows)

        user_ids = [cast("int", r["id"]) for r in rows if r["role"] == "user"]
        boundary = user_ids[1] if len(user_ids) > 1 else None
        for m in messages:
            if boundary is not None and int(cast("str", m["id"])) >= boundary:
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
            "type": config_type,
            # True when the turn's most recent execution ended CRASHED — an unhandled
            # step exception, or a process death the boot sweep stamped. A crash that
            # produced no reply row is otherwise indistinguishable from a normal empty
            # turn on refetch (both settle to working:false), so the FE needs this to
            # render an "ended unexpectedly" note instead of a bare tool-trace footer.
            # stop_reason is deliberately NOT exposed: it is raw str(exc)/"process
            # death", never user-displayable, so the FE renders a fixed message.
            "crashed": bool(latest and latest.state == TurnExecution.CRASHED),
        }

    def bulk_gists(self, channel: str, turn_ids: list[int]) -> dict[int, str]:
        return _bulk_gists(channel, turn_ids)


# Module-level singleton, mirroring the pattern in other services.
_default_service: TurnSerializerService | None = None


def get_service() -> TurnSerializerService:
    """Return the module-level singleton."""
    global _default_service
    if _default_service is None:
        _default_service = TurnSerializerService()
    return _default_service

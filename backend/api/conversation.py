"""Conversation blueprint — the thread feed.

GET /api/threads — id list + collapsed metadata (gist, preview, last activity).
GET /api/thread/<turn_id> — one turn's full block (the WS-refetch + expand read).
POST /api/threads/batch — many blocks, a pure concatenation of the single-turn getter.
"""

import logging
import sqlite3
from typing import cast
from flask import Blueprint, request, jsonify
from flask.typing import ResponseReturnValue

from .auth import require_session
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.rich_media_parser import parse as _parse_rich_media, resolve_tool_call_transcript_ids as _resolve_ids

logger = logging.getLogger(__name__)
conversation_bp = Blueprint('conversation', __name__)

_THREAD_EXCLUDE = ('subagent_return',)


def _fetch_tool_calls_for_transcripts(conn: sqlite3.Connection, transcript_ids: list[int]) -> list[dict[str, object]]:
    """Tool-call rows anchored to any of the given transcript ids, ordered by id.

    Rows within the 7-day retention window carry the rich-media payloads the
    parser uses to pair span tags with cards, and the act_summary the frontend
    prints as the blue box under each tool chip. Ordered by ``id`` (not
    ``created_at``, which is 1-second granular and ambiguous within a batch) so
    multiple calls on one row keep their emission order.
    """
    if not transcript_ids:
        return []
    placeholders = ",".join("?" * len(transcript_ids))
    tc_rows = conn.execute(
        f"SELECT tool_name, params, result, summary, created_at FROM tool_calls "
        f"WHERE transcript_id IN ({placeholders}) ORDER BY id",
        tuple(transcript_ids),
    ).fetchall()
    return [
        {
            "tool_name": r[0],
            "params": r[1],
            "result": r[2] or "",
            "summary": r[3] or "",
            "created_at": r[4],
        }
        for r in tc_rows
    ]


def _fetch_attachments_for_transcripts(conn: sqlite3.Connection, transcript_ids: list[int]) -> dict[int, list[dict[str, object]]]:
    """Soft-deleted docs are filtered out, so a removed file silently does not
    render. Each attachment carries the inline-serving
    ``/documents/<id>/preview`` URL.
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
            "url": f"/documents/{doc_id}/preview",
        })
    return by_id


def _rows_to_messages(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project raw transcript rows (oldest-first) into the conversation message
    shape — attachments for user rows, tool-call chips + rich-media segments for
    assistant rows. Shared by recent-history, thread-list and thread-expand."""
    from services.database_service import get_shared_db_service

    messages: list[dict[str, object]] = []
    db = get_shared_db_service()
    with db.connection() as conn:
        attachments_by_id = _fetch_attachments_for_transcripts(
            conn, [cast("int", r['id']) for r in rows if r['role'] == 'user']
        )

        for r in rows:
            transcript_id = cast("int", r['id'])
            role, content = r['role'], r['content'] or ""
            ts = format_date(cast("str", r['created_at']), CHAT_TIMESTAMP_FMT, for_ui=True) or ""
            msg: dict[str, object] = {
                "id": str(transcript_id),
                "role": role,
                "content": content,
                "timestamp": ts,
                "turn_id": r['turn_id'],
            }

            if role == 'user':
                attachments = attachments_by_id.get(transcript_id)
                if attachments:
                    msg["attachments"] = attachments
            else:
                own_calls = _fetch_tool_calls_for_transcripts(conn, [transcript_id])
                chips = [
                    {"tool_name": c["tool_name"], "summary": c["summary"]}
                    for c in own_calls
                    if c["tool_name"] != "chat_history_compactor"
                ]
                if chips:
                    msg["tool_calls"] = chips
                turn_calls = _fetch_tool_calls_for_transcripts(conn, _resolve_ids(transcript_id, conn))
                segments = _parse_rich_media(str(content), turn_calls)
                if not segments and content:
                    segments = [{"type": "text", "content": content}]
                msg["segments"] = segments
            messages.append(msg)

    return messages


def serialize_turn(channel: str, turn_id: int) -> dict[str, object]:
    """The single turn-block getter — the REST single/batch reads and the WS
    refetch all flow through here, so one fetch fully determines a turn's render
    with no signal memory.

    Returns the WHOLE turn (no settle0 floor) projected into messages, with the
    settle0 row tagged ``settled: 0`` (the only main-spine presentation cut), the
    collapsed-feed metadata (gist, preview, last activity) and the turn-level
    render state (``working`` — unsettled/in-flight — and ``duration_ms``,
    derived from the row span) folded in."""
    from services.transcript_service import Transcript
    from services.thread_gist_service import get_thread_gist_service
    from services.time_utils import parse_utc

    rows = Transcript.thread_rows(channel, turn_id, exclude_roles=_THREAD_EXCLUDE)
    messages = _rows_to_messages(rows)

    settle = Transcript.settle0(channel, turn_id)
    for m in messages:
        if settle is not None and m["id"] == str(settle):
            m["settled"] = 0
            break

    duration_ms = 0
    if len(rows) >= 2:
        span = parse_utc(cast("str", rows[-1]["created_at"])) - parse_utc(cast("str", rows[0]["created_at"]))
        duration_ms = int(span.total_seconds() * 1000)

    return {
        "turn_id": turn_id,
        "gist": get_thread_gist_service().bulk_get(channel, [turn_id]).get(turn_id),
        "preview": cast("str", rows[0]["content"] or "")[:200] if rows else "",
        "last_activity_at": rows[-1]["created_at"] if rows else None,
        "working": settle is None,
        "duration_ms": duration_ms,
        "messages": messages,
    }


@conversation_bp.route('/api/threads', methods=['GET'])
@require_session
def conversation_threads() -> ResponseReturnValue:
    """List threads (turns) with collapsed metadata — the thread feed.

    Returns the ``limit`` most-recently-active threads, each with a preview
    (first content), row count, last activity timestamp and turn_id. Scroll-up
    pagination via ``offset``.
    """
    from services.transcript_service import Transcript

    try:
        limit = max(1, min(120, int(request.args.get("limit", 20))))
    except (ValueError, TypeError):
        limit = 20
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (ValueError, TypeError):
        offset = 0

    threads, has_more, threads_returned = Transcript.recent_threads(
        'user', exclude_roles=_THREAD_EXCLUDE, limit=limit, offset=offset,
    )
    if threads:
        from services.thread_gist_service import get_thread_gist_service  # noqa: PLC0415
        gist_turn_ids = [cast("int", t["turn_id"]) for t in threads if t.get("turn_id") is not None]
        gists = get_thread_gist_service().bulk_get('user', gist_turn_ids)
        for t in threads:
            tid = t.get("turn_id")
            if tid is not None:
                t["gist"] = gists.get(cast("int", tid))
    return jsonify({"threads": threads, "has_more": has_more, "threads_returned": threads_returned})


@conversation_bp.route('/api/thread/<int:turn_id>', methods=['GET'])
@require_session
def conversation_thread(turn_id: int) -> ResponseReturnValue:
    """One turn's full block — the WS-refetch + expand-on-click read."""
    return jsonify(serialize_turn('user', turn_id))


@conversation_bp.route('/api/threads/batch', methods=['POST'])
@require_session
def conversation_threads_batch() -> ResponseReturnValue:
    """Many turn blocks in one round-trip — a pure concatenation of the single
    -turn getter over the requested ids (the FE paginates the feed ~20/page)."""
    body = request.get_json(silent=True) or {}
    turn_ids = cast("list[int]", body.get("turn_ids") or [])
    return jsonify({"blocks": [serialize_turn('user', int(t)) for t in turn_ids]})

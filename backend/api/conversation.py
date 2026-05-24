"""Conversation blueprint — GET /conversation/recent."""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.rich_media_parser import parse as _parse_rich_media, resolve_tool_call_transcript_ids as _resolve_ids

logger = logging.getLogger(__name__)
conversation_bp = Blueprint('conversation', __name__)


def _fetch_tool_calls_for_transcript(conn, transcript_id: int) -> list[dict]:
    """Fetch all tool_calls rows for a transcript, including ephemeral=1 rows.

    Ephemeral rows carry the rich-media instruction trailer that the parser
    uses to pair span tags with their payloads.  Filtering them out would
    break card reconstruction on page refresh.
    """
    tc_rows = conn.execute(
        "SELECT tool_name, params, result, ephemeral, created_at FROM tool_calls "
        "WHERE transcript_id = ? ORDER BY created_at",
        (transcript_id,),
    ).fetchall()
    return [
        {
            "tool_name": r[0],
            "params": r[1],
            "result": r[2] or "",
            "ephemeral": r[3],
            "created_at": r[4],
        }
        for r in tc_rows
    ]


def get_recent_history(limit=12, offset=0):
    """Fetch recent messages from transcript. Returns (messages, has_more)."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM transcript "
            "WHERE channel = 'user' AND role NOT IN ('subagent_return') "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        # Build messages and attach rich-media segments for assistant rows.
        # _resolve_ids() queries the DB for the user-input transcript row that
        # precedes each assistant row — the same lookup used by the WS-send path.
        # Both paths call the same helper so they can never silently diverge.
        messages = []

        for row in reversed(rows):
            transcript_id, role, content, created_at = row[0], row[1], row[2] or "", row[3]

            ts = format_date(created_at, CHAT_TIMESTAMP_FMT, for_ui=True) or ""

            if role == 'user':
                messages.append({
                    "id": str(transcript_id),
                    "role": role,
                    "content": content,
                    "timestamp": ts,
                })
            else:
                msg = {
                    "id": str(transcript_id),
                    "role": role,
                    "content": content,
                    "timestamp": ts,
                }
                # Attach segments for assistant rows so the frontend can
                # reconstruct rich-media cards on page refresh.
                input_ids = _resolve_ids(transcript_id, conn)
                if input_ids:
                    tool_calls = _fetch_tool_calls_for_transcript(conn, input_ids[0])
                else:
                    tool_calls = []
                segments = _parse_rich_media(content, tool_calls)
                if not segments:
                    segments = [{"type": "text", "content": content}]
                msg["segments"] = segments
                messages.append(msg)

    return messages, len(rows) == limit


@conversation_bp.route('/conversation/recent', methods=['GET'])
@require_session
def conversation_recent():
    try:
        limit = max(1, min(120, int(request.args.get("limit", 12))))
    except (ValueError, TypeError):
        limit = 12
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (ValueError, TypeError):
        offset = 0

    messages, has_more = get_recent_history(limit, offset)
    return jsonify({"messages": messages, "has_more": has_more})

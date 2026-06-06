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


def _fetch_attachments_for_transcripts(conn, transcript_ids: list[int]) -> dict:
    """Map each user transcript_id -> its uploaded attachments for refresh render.

    One batch join of transcript_docs -> documents (soft-deleted docs filtered out,
    so a removed file silently does not render).  Each attachment carries the
    inline-serving ``/documents/<id>/preview`` URL the renderer uses as the
    <img>/chip source — the same media the live blob: preview showed before reload.
    Written by message_processor._seed_upload_attachment.  (TKT-842)

    Ordered by ``td.rowid`` (insertion order).  Uploads fan out across a thread
    pool, so for a multi-attachment turn this stable order may differ from the
    user's original composer order — a cosmetic interleave, never a lost/dup row.
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
    by_id: dict = {}
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


def get_recent_history(limit=12, offset=0):
    """Fetch recent messages from transcript. Returns (messages, has_more)."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM transcript "
            "WHERE channel = 'user' AND role NOT IN ('subagent_return', 'compaction') "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        # Build messages and attach rich-media segments for assistant rows.
        # _resolve_ids() queries the DB for the user-input transcript row that
        # precedes each assistant row — the same lookup used by the WS-send path.
        # Both paths call the same helper so they can never silently diverge.
        messages = []

        # Re-render uploaded attachments on refresh: batch-fetch the doc links for
        # every user row in this page (the live blob: preview is gone after reload).
        attachments_by_id = _fetch_attachments_for_transcripts(
            conn, [row[0] for row in rows if row[1] == 'user']
        )

        for row in reversed(rows):
            transcript_id, role, content, created_at = row[0], row[1], row[2] or "", row[3]

            ts = format_date(created_at, CHAT_TIMESTAMP_FMT, for_ui=True) or ""

            if role == 'user':
                msg = {
                    "id": str(transcript_id),
                    "role": role,
                    "content": content,
                    "timestamp": ts,
                }
                attachments = attachments_by_id.get(transcript_id)
                if attachments:
                    msg["attachments"] = attachments
                messages.append(msg)
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

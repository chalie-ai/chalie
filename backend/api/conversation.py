"""Conversation blueprint — GET /conversation/recent."""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.rich_media_parser import parse as _parse_rich_media, resolve_tool_call_transcript_ids as _resolve_ids

logger = logging.getLogger(__name__)
conversation_bp = Blueprint('conversation', __name__)


def _fetch_tool_calls_for_transcripts(conn, transcript_ids: list[int]) -> list[dict]:
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


def _fetch_attachments_for_transcripts(conn, transcript_ids: list[int]) -> dict:
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
        # Under the chain model a tool turn is many rows (input → step rows that
        # emitted the tools → final synthesis row). Each assistant row shows its
        # OWN tool chips (anchored to that row), while rich-media spans resolve
        # TURN-WIDE via _resolve_ids() — the span sits on the final row but the
        # tool ran on a step row. The WS-send path calls the same helper, so the
        # two paths can never silently diverge.
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
                # Per-row chips: the tools THIS row emitted (anchored to its own
                # id), each with the ability's persisted act_summary — the blue
                # box the frontend prints. Set only when the row drove tools; the
                # framework compaction marker never surfaces as a chip.
                own_calls = _fetch_tool_calls_for_transcripts(conn, [transcript_id])
                chips = [
                    {"tool_name": c["tool_name"], "summary": c["summary"]}
                    for c in own_calls
                    if c["tool_name"] != "chat_history_compactor"
                ]
                if chips:
                    msg["tool_calls"] = chips
                # Segments: rich-media spans live on the final row but the tool
                # ran on a step row, so pair TURN-WIDE — gather every tool call
                # across the turn's transcript rows, then parse THIS row's content.
                turn_ids = _resolve_ids(transcript_id, conn)
                turn_calls = _fetch_tool_calls_for_transcripts(conn, turn_ids)
                segments = _parse_rich_media(content, turn_calls)
                if not segments and content:
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

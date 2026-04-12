"""Conversation blueprint — GET /conversation/recent."""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session
from services.blocks_render_service import BlocksRenderService

_blocks = BlocksRenderService()
logger = logging.getLogger(__name__)
conversation_bp = Blueprint('conversation', __name__)


def get_recent_history(limit=12, offset=0):
    """Fetch recent messages from transcript. Returns (messages, has_more)."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM transcript "
            "WHERE channel = 'user' ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    messages = [
        {
            "id": str(row[0]),
            "role": row[1],
            "content": row[2],
            "blocks": _blocks.from_markdown(row[2]) if row[1] == 'assistant' else [],
            "timestamp": row[3],
        }
        for row in reversed(rows)
    ]

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

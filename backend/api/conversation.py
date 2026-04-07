"""
Conversation blueprint — /conversation/recent.

The /chat endpoint has been replaced by the WebSocket handler in api/websocket.py.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session
from services.blocks_render_service import BlocksRenderService

_blocks_svc = BlocksRenderService()

logger = logging.getLogger(__name__)

conversation_bp = Blueprint('conversation', __name__)


WORKING_MEMORY_SIZE = 12


@conversation_bp.route('/conversation/recent', methods=['GET'])
@require_session
def conversation_recent():
    """Return paginated conversation from the transcript table."""
    try:
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService

        # Parse and clamp query params
        try:
            limit = max(1, min(120, int(request.args.get("limit", 12))))
        except (ValueError, TypeError):
            limit = 12
        try:
            offset = max(0, int(request.args.get("offset", 0)))
        except (ValueError, TypeError):
            offset = 0

        # Resolve active channel
        store = MemoryClientService.create_connection()
        channel = store.get('active_channel:default')
        if isinstance(channel, bytes):
            channel = channel.decode()
        if not channel:
            channel = 'web:default:1'

        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

            # Total count for this channel
            cursor.execute(
                "SELECT COUNT(*) FROM transcript WHERE channel = ? AND internal = 0 AND role IN ('user', 'assistant')",
                (channel,),
            )
            total = cursor.fetchone()[0]

            if total == 0:
                return jsonify({
                    "thread_id": channel,
                    "exchanges": [],
                    "total": 0,
                    "has_more": False,
                    "working_memory_count": WORKING_MEMORY_SIZE,
                    "from_expired": False,
                }), 200

            # Fetch paginated rows — order by id DESC, then reverse to chronological
            cursor.execute(
                """
                SELECT id, role, content, created_at
                FROM transcript
                WHERE channel = ? AND internal = 0 AND role IN ('user', 'assistant')
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (channel, limit, offset),
            )
            rows = list(reversed(cursor.fetchall()))

        # Pair user+assistant turns into exchange-like objects
        formatted = []
        i = 0
        pair_index = 0
        while i < len(rows):
            row = rows[i]
            row_id, role, content, created_at = row[0], row[1], row[2], row[3]

            if role == 'user':
                # Look ahead for matching assistant turn
                response_text = ''
                response_time = ''
                if i + 1 < len(rows) and rows[i + 1][1] == 'assistant':
                    response_text = rows[i + 1][2]
                    response_time = rows[i + 1][3]
                    i += 2
                else:
                    i += 1

                distance_from_end = total - (offset + pair_index + 1)
                in_working_memory = distance_from_end < WORKING_MEMORY_SIZE

                formatted.append({
                    "id": str(row_id),
                    "prompt": content,
                    "blocks": _blocks_svc.from_markdown(response_text) if response_text else [],
                    "topic": channel,
                    "timestamp": created_at,
                    "in_working_memory": in_working_memory,
                })
                pair_index += 1
            else:
                # Orphaned assistant turn (no preceding user turn in this page)
                i += 1

        has_more = (offset + limit) < total

        return jsonify({
            "thread_id": channel,
            "exchanges": formatted,
            "total": total,
            "has_more": has_more,
            "working_memory_count": WORKING_MEMORY_SIZE,
            "from_expired": False,
        }), 200

    except Exception as e:
        logger.error(f"[REST API] conversation/recent error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve conversation"}), 500

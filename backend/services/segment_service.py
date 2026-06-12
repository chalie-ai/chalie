"""
SegmentService — builds rich media segments from transcript content and tool_calls rows.

Extracted from api/websocket.py so the parsing logic is not entangled with
WebSocket-specific handler code.
"""

import logging

logger = logging.getLogger(__name__)


class SegmentService:
    """Builds rich media segments from transcript content and tool_calls rows."""

    @staticmethod
    def build(content: str, transcript_ids: list) -> list:
        """Build segments for a message event.

        Fetches tool_calls rows by exact transcript IDs then runs the rich
        media parser.  Always returns at least one plain text segment so the
        frontend never receives an empty array.

        When ``transcript_ids`` is empty or None a plain text segment is
        returned immediately — no DB round-trip, no silent degradation.

        Args:
            content: Sanitised assistant text (transcript.content).
            transcript_ids: List of transcript row IDs for this turn.

        Returns:
            List of segment dicts.  Always at least one element.
        """
        if not transcript_ids:
            logger.warning(
                "[SEGMENT] build: no transcript_ids — emitting plain text segment"
            )
            return [{"type": "text", "content": content}]

        from services.rich_media_parser import parse as _parse_rich_media

        tool_calls = SegmentService._fetch_tool_calls(transcript_ids)
        segments = _parse_rich_media(content, tool_calls)
        return segments or [{"type": "text", "content": content}]

    @staticmethod
    def _fetch_tool_calls(transcript_ids: list) -> list:
        """Fetch all tool_calls rows for a list of transcript IDs.

        All rows are durable; the rich-media parser receives the full set so
        span tags can be paired with their payloads on both the live broadcast
        and page-refresh paths.

        Args:
            transcript_ids: List of transcript.id values from the turn that
                produced the response.

        Returns:
            Tool_calls rows ordered by created_at, id.  Empty list on any error.
        """
        if not transcript_ids:
            return []
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            placeholders = ','.join('?' * len(transcript_ids))
            with db.connection() as conn:
                tc_rows = conn.execute(
                    f"SELECT tool_name, params, result, created_at "
                    f"FROM tool_calls "
                    f"WHERE transcript_id IN ({placeholders}) "
                    f"ORDER BY created_at, id",
                    transcript_ids,
                ).fetchall()
            return [
                {
                    "tool_name": r[0],
                    "params": r[1],
                    "result": r[2] or "",
                    "created_at": r[3],
                }
                for r in tc_rows
            ]
        except Exception as exc:
            logger.debug("[SEGMENT] _fetch_tool_calls failed: %s", exc)
            return []

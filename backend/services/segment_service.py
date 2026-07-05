

import logging

from services.database import Database

logger = logging.getLogger(__name__)


class SegmentService:

    @staticmethod
    def build(content: str, transcript_ids: list[int]) -> list[dict[str, object]]:
        """Always returns at least one plain text segment. When transcript_ids is empty, no DB query is issued."""
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
    def _fetch_tool_calls(transcript_ids: list[int]) -> list[dict[str, object]]:
        if not transcript_ids:
            return []
        try:
            placeholders = ','.join('?' * len(transcript_ids))
            tc_rows = Database.conn().execute(
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



import logging

from models.tool_call import ToolCall

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
            return [
                {
                    "tool_name": tc.tool_name,
                    "params": tc.params,
                    "result": tc.result or "",
                    "created_at": tc.created_at,
                }
                for tc in ToolCall.by_transcripts(transcript_ids)
            ]
        except Exception as exc:
            logger.debug("[SEGMENT] _fetch_tool_calls failed: %s", exc)
            return []

"""
Tool Call Service — unified API for recording tool call audit entries.

All writes to the tool_calls table MUST go through this service.
"""

import json
import logging

from services.time_utils import utc_now
from services.database_service import get_shared_db_service

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TOOL CALLS]"


class ToolCallService:
    """Unified API for recording tool call audit entries."""

    def store(self, transcript_id, tool_name, params, result, ephemeral=False):
        """Store a single tool call record.

        Args:
            transcript_id: ID of the transcript entry this call belongs to.
            tool_name: Name of the tool invoked.
            params: Dict of parameters — serialized to JSON internally.
            result: String result from the tool invocation.
            ephemeral: If True, marks the record as ephemeral (not surfaced in history).
        """
        db = get_shared_db_service()
        params_str = json.dumps(params) if isinstance(params, dict) else (params or '{}')
        ephemeral_int = 1 if ephemeral else 0
        now = utc_now().isoformat()

        try:
            with db.connection() as conn:
                conn.execute(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, ephemeral, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (transcript_id, tool_name, params_str, result, ephemeral_int, now),
                )
                conn.commit()
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to store tool call: tool={tool_name!r} transcript={transcript_id}")

    def get_by_transcript(self, transcript_id, include_ephemeral=True):
        """Get all tool calls for a transcript entry.

        Args:
            transcript_id: Transcript entry ID to look up.
            include_ephemeral: If False, only returns non-ephemeral records.

        Returns:
            List of dicts with all tool_calls columns.
        """
        db = get_shared_db_service()
        if include_ephemeral:
            sql = "SELECT * FROM tool_calls WHERE transcript_id = ? ORDER BY created_at"
            params = (transcript_id,)
        else:
            sql = "SELECT * FROM tool_calls WHERE transcript_id = ? AND ephemeral = 0 ORDER BY created_at"
            params = (transcript_id,)

        try:
            return db.fetch_all(sql, params)
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to fetch tool calls for transcript={transcript_id}")
            return []

    def get_by_transcript_ids(self, transcript_ids: list, include_ephemeral=True) -> dict:
        """Get tool calls grouped by transcript_id for a list of IDs.

        Args:
            transcript_ids: List of transcript entry IDs to look up.
            include_ephemeral: If False, only returns non-ephemeral records.

        Returns:
            Dict mapping transcript_id (int) → list of tool_call dicts.
        """
        if not transcript_ids:
            return {}

        db = get_shared_db_service()
        placeholders = ','.join('?' for _ in transcript_ids)

        if include_ephemeral:
            sql = (
                f"SELECT id, transcript_id, tool_name, params, result, "
                f"ephemeral, created_at "
                f"FROM tool_calls WHERE transcript_id IN ({placeholders}) "
                f"ORDER BY created_at"
            )
        else:
            sql = (
                f"SELECT id, transcript_id, tool_name, params, result, "
                f"ephemeral, created_at "
                f"FROM tool_calls WHERE transcript_id IN ({placeholders}) AND ephemeral = 0 "
                f"ORDER BY created_at"
            )

        try:
            rows = db.fetch_all(sql, transcript_ids)
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to fetch tool calls for transcripts={transcript_ids}")
            return {}

        grouped: dict = {}
        for row in rows:
            tid = row.get('transcript_id')
            if tid not in grouped:
                grouped[tid] = []
            grouped[tid].append(dict(row))
        return grouped

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

    def store_batch(self, transcript_id, tool_calls, results, ephemeral=True):
        """Store multiple tool call records from a single LLM response.

        Args:
            transcript_id: ID of the transcript entry this call belongs to.
            tool_calls: List of dicts with {'id': str, 'name': str, 'input': dict}.
            results: List of dispatcher result dicts with 'action_type', 'status', 'result', etc.
            ephemeral: If True, marks the records as ephemeral.
        """
        if not tool_calls:
            return

        db = get_shared_db_service()
        ephemeral_int = 1 if ephemeral else 0
        now = utc_now().isoformat()

        rows = []
        for tc, r in zip(tool_calls, results):
            tool_name = tc['name']

            # Special case: find_tools — store discovered tool names as params
            # The dispatcher unwraps the handler dict; _discovered_tools is a top-level key
            if tool_name == 'find_tools' and r.get('_discovered_tools'):
                params_str = json.dumps({'discovered': r['_discovered_tools']})
            else:
                params_str = json.dumps(tc.get('input', {}))

            result_str = str(r.get('result', ''))
            rows.append((transcript_id, tool_name, params_str, result_str, ephemeral_int, now))

        try:
            with db.connection() as conn:
                conn.executemany(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, ephemeral, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to store batch of {len(rows)} tool calls for transcript={transcript_id}")

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

    def get_by_timerange(self, center_dt, buffer_minutes=5):
        """Get tool calls within ±buffer_minutes of center_dt.

        Args:
            center_dt: ISO datetime string at the centre of the window.
            buffer_minutes: Half-width of the window in minutes.

        Returns:
            List of dicts with all tool_calls columns (including ephemeral).
        """
        from services.time_utils import parse_utc
        from datetime import timedelta

        center = parse_utc(center_dt)
        lo = (center - timedelta(minutes=buffer_minutes)).isoformat()
        hi = (center + timedelta(minutes=buffer_minutes)).isoformat()

        db = get_shared_db_service()
        sql = (
            "SELECT * FROM tool_calls "
            "WHERE created_at >= ? AND created_at <= ? "
            "ORDER BY created_at"
        )
        try:
            return db.fetch_all(sql, (lo, hi))
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to fetch tool calls for timerange center={center_dt}")
            return []

    def get_find_tools_results(self, transcript_ids) -> list:
        """Get all find_tools results for tool compounding.

        Args:
            transcript_ids: A single transcript entry ID (int) or a list of IDs.

        Returns:
            Flat deduped list of all discovered tool names from find_tools invocations.
        """
        if isinstance(transcript_ids, int):
            transcript_ids = [transcript_ids]
        if not transcript_ids:
            return []

        db = get_shared_db_service()
        placeholders = ','.join('?' for _ in transcript_ids)
        sql = (
            f"SELECT params FROM tool_calls "
            f"WHERE transcript_id IN ({placeholders}) AND tool_name = 'find_tools'"
        )
        try:
            rows = db.fetch_all(sql, transcript_ids)
        except Exception:
            logger.exception(f"{LOG_PREFIX} Failed to fetch find_tools results for transcripts={transcript_ids}")
            return []

        discovered = []
        seen = set()
        for row in rows:
            try:
                data = json.loads(row.get('params') or '{}')
                for name in data.get('discovered', []):
                    if name not in seen:
                        seen.add(name)
                        discovered.append(name)
            except (json.JSONDecodeError, TypeError):
                continue

        return discovered

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

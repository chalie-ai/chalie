"""Pure SQL helpers for compaction state, backed by the `tool_calls` table.

The `compactions` table is dropped (see migrations/migration_001_drop_compactions.py). Compaction summaries are
now append-only rows in `tool_calls` with `tool_name='compaction'`. Failure
rows are written for audit but filtered out by the canonical lookup so they
never poison the prompt.

Callers:
  - MessageProcessor.get_previous_messages  (watermark + prepend)
  - MessageProcessor._wrap_with_checkpoint (checkpoint envelope)
  - MessageProcessor._run_full_compaction  (orchestrator reads prior state)
"""

import logging
from typing import Optional, Dict, List

from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"


def get_compaction(channel: str) -> Optional[Dict]:
    """Return the latest success compaction for a channel, or None.

    Canonical lookup — joins tool_calls → transcript, filters status=success,
    orders by tool_calls.id DESC so append-only history is honoured.
    Failure rows are invisible to this query.

    Returns dict with: compacted_text, compacted_up_to_id, tool_call_id,
    created_at (the row's UTC timestamp, used by the Brain observability tab).
    Returns None if no success compaction exists for the channel.
    Never raises — DB errors are logged and treated as "no compaction".
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT tc.result, tc.params, tc.id, tc.created_at
                FROM tool_calls tc
                JOIN transcript t ON t.id = tc.transcript_id
                WHERE tc.tool_name = 'compaction'
                  AND t.channel = ?
                  AND json_extract(tc.params, '$.status') = 'success'
                ORDER BY tc.id DESC
                LIMIT 1
                """,
                (channel,),
            ).fetchone()

        if not row:
            return None

        params = parse_json_column(row[1])
        return {
            'compacted_text': row[0],
            'compacted_up_to_id': params.get('compacted_up_to_id', 0),
            'tool_call_id': row[2],
            'created_at': row[3],
        }
    except Exception as exc:
        logger.warning("%s Failed to get compaction for %s: %s", LOG_PREFIX, channel, exc)
        return None


def get_entries_since(channel: str, watermark: int = 0, limit: int = 2000) -> List[Dict]:
    """Read transcript entries with id > watermark for a channel.

    Returns entries in chronological order (oldest first).
    """
    from services import transcript_service
    return transcript_service.get_recent(channel, limit=limit, since_id=watermark)

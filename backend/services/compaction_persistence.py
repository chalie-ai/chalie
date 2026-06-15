"""Pure SQL helpers for compaction state.

The canonical watermark home (design §3.6) is the transcript table: a row with
role='compaction' whose OWN id is the watermark (compacted_up_to_id). Downstream
`id > watermark` reads naturally exclude that row and everything before it.

Callers:
  - MessageProcessor._previous_rows         (watermark for `id > watermark` reads)
  - MessageProcessor._wrap_with_checkpoint  (checkpoint envelope prepend)
  - transcript_service.cleanup_unlinked_entries (watermark-bounded cleanup)
  - api.system compaction observability     (Brain read-only view)
"""

import logging
from typing import Optional, Dict


logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"


def get_compaction(channel: str) -> Optional[Dict]:
    """Latest history-compaction summary for a channel, or None.

    Canonical watermark reader (design §3.6): the newest transcript row with
    role='compaction'. Its OWN id is the watermark (compacted_up_to_id), so
    _previous_rows()'s `id > watermark` naturally excludes it and everything
    before it. Never raises — DB errors are logged and treated as 'no compaction'."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, content, created_at FROM transcript
                WHERE channel = ? AND role = 'compaction'
                ORDER BY id DESC LIMIT 1
                """,
                (channel,),
            ).fetchone()
        if not row:
            return None
        return {
            "compacted_text": row[0 + 1],         # content
            "compacted_up_to_id": row[0],         # the row's own id
            "tool_call_id": None,                 # legacy field; no longer a tool_call
            "created_at": row[2],
        }
    except Exception as exc:
        logger.warning("%s Failed to get compaction for %s: %s", LOG_PREFIX, channel, exc)
        return None

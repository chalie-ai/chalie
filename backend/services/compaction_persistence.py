"""Pure SQL helpers for the `compactions` table.

Lives outside CompactionMessageProcessor because it's pure persistence,
not an LLM call. Consumed by:
  - MessageProcessor.getPreviousMessages (watermark + prepend)
  - MessageProcessor._wrap_with_checkpoint (checkpoint envelope)
  - MessageProcessor._run_full_compaction orchestrator
"""

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"


def get_compaction(channel: str, _context=None) -> Optional[Dict]:
    """Read the current compaction row for a channel.

    Returns None if no compaction exists.
    Returns dict with: compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content
                FROM compactions
                WHERE channel = ?
                """,
                (channel,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return None

        return {
            'compacted_text': row[0],
            'compacted_up_to_id': row[1],
            'token_count': row[2],
            'updated_at': row[3],
            'overflow_content': row[4],
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to get compaction for {channel}: {e}")
        return None


def get_entries_since(channel: str, watermark: int = 0, limit: int = 2000) -> List[Dict]:
    """Read transcript entries with id > watermark for a channel.

    Returns entries in chronological order (oldest first).
    """
    from services import transcript_service
    return transcript_service.get_recent(channel, limit=limit, since_id=watermark)

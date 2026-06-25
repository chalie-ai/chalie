"""Pure SQL helpers for compaction state.

The canonical watermark home (design §3.6) is the transcript table: a row
with role='compaction' whose OWN id is the watermark (compacted_up_to_id).
Downstream `id > watermark` reads naturally exclude that row and everything
before it.

Compaction is **thread-scoped** (workstream C): the watermark keys off
``(channel, turn_id)``. A thread-starter pass ``turn_id`` to scope the
watermark to that thread; a ``None`` turn_id falls back to the legacy
channel-wide latest compaction (housekeeping channels that never carry a
thread id).

Callers: ``MessageProcessor._previous_rows`` (watermark for `id >
watermark` reads), ``MessageProcessor._wrap_with_checkpoint`` (checkpoint
envelope prepend), ``transcript_service.cleanup_unlinked_entries``
(watermark-bounded cleanup), ``api.system`` compaction observability
(Brain read-only view), ``turn_zero_flashback._living_doc_now``.
"""

import logging
from typing import Optional, Dict


logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"


def get_compaction(channel: str, turn_id: "int | None" = None) -> Optional[Dict[str, object]]:
    """Latest history-compaction summary, or None. Never raises — DB errors are
    logged and treated as 'no compaction'.

    ``turn_id`` scopes the watermark to one thread (workstream C): the latest
    ``role='compaction'`` row for ``(channel, turn_id)``. When ``None`` the
    latest compaction for the whole channel is returned (the legacy
    housekeeping path — channels that never allocate a thread).
    """
    try:
        from services.transcript_service import Transcript
        if turn_id is not None:
            rows = Transcript.by_turn(channel, turn_id)
            rows = [r for r in rows if r.get('role') == 'compaction']
            if not rows:
                return None
            row = rows[-1]
        else:
            rows = Transcript.get_recent(channel, limit=1, role='compaction')
            if not rows:
                return None
            row = rows[0]
        return {
            "compacted_text": row['content'],
            "compacted_up_to_id": row['id'],
            "tool_call_id": None,
            "created_at": row['created_at'],
        }
    except Exception as exc:
        logger.warning("%s Failed to get compaction for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)
        return None

"""Pure SQL helpers for compaction state.

A compaction is a row in the dedicated ``compactions`` table — never a
transcript row. Keeping it off the transcript spine means firing one never
moves a ``turn_id``, so the turn boundary survives a mid-turn collapse.

The watermark is two-axis, selected by ``for_turn_id`` (the conversation view):

* **MAIN** view → ``for_turn_id IS NULL``; ``compacted_up_to`` is a **turn_id**.
  The spine read drops turns whose ``turn_id <= compacted_up_to``.
* **FORK** view → ``for_turn_id = T``; ``compacted_up_to`` is a **transcript.id**.
  The thread read drops rows of turn ``T`` whose ``id <= compacted_up_to``.

The two axes never collide, so a MAIN compaction can't hide a fork reply and a
FORK compaction can't hide the spine.

Callers: ``MessageProcessor._previous_rows`` (watermark for the spine / fork
read), ``MessageProcessor._wrap_with_checkpoint`` (checkpoint envelope prepend),
``chat_history_compactor`` (writes the checkpoint), ``transcript_service
.cleanup_unlinked_entries`` (scope-aware GC), ``api.system`` compaction
observability (Brain read-only view).
"""

import logging
from typing import Optional, Dict

from services.database import Database


logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"


def get_compaction(channel: str, for_turn_id: "int | None") -> Optional[Dict[str, object]]:
    """Latest checkpoint for one ``(channel, for_turn_id)`` scope, or None. Never
    raises — DB errors are logged and treated as 'no compaction'.

    ``for_turn_id`` selects the watermark axis: ``None`` is the MAIN spine
    checkpoint (``compacted_up_to`` is a turn_id), an integer is that thread's
    FORK checkpoint (``compacted_up_to`` is a transcript.id). The newest row in
    the scope wins (``id DESC``).
    """
    try:
        conn = Database.conn()
        row = conn.execute(
            "SELECT compacted_up_to, content, created_at FROM compactions "
            "WHERE channel = ? AND for_turn_id IS ? ORDER BY id DESC LIMIT 1",
            (channel, for_turn_id),
        ).fetchone()
        if not row:
            return None
        return {
            "compacted_text": row[1],
            "compacted_up_to": row[0],
            "created_at": row[2],
        }
    except Exception as exc:
        logger.warning("%s Failed to get compaction for %s scope=%s: %s", LOG_PREFIX, channel, for_turn_id, exc)
        return None


def write_compaction(channel: str, for_turn_id: "int | None", compacted_up_to: int, content: str) -> None:
    """Append a checkpoint for one ``(channel, for_turn_id)`` scope.

    ``compacted_up_to`` is a turn_id for a MAIN compaction (``for_turn_id`` None)
    or a transcript.id for a FORK compaction. ``created_at`` is filled by the
    table default (UTC ``datetime('now')``), matching the transcript writers.
    """
    conn = Database.conn()
    conn.execute(
        "INSERT INTO compactions (channel, for_turn_id, compacted_up_to, content) "
        "VALUES (?, ?, ?, ?)",
        (channel, for_turn_id, compacted_up_to, content),
    )

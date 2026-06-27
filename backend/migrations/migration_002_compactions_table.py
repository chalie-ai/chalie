"""Migration 002 — move compaction state into the ``compactions`` table.

The Thread Context & Compaction Model replaces the old ``role='compaction'``
transcript rows with a dedicated ``compactions`` table carrying a two-axis
watermark (MAIN cuts on the turn_id axis, FORK on the transcript.id axis). This
migration:

  1. Creates the ``compactions`` table (idempotent — ``schema.sql`` also
     declares it, so a fresh boot's schema convergence creates it; this mirror
     lets an operator apply the change to a live database without a restart).
  2. Purges every legacy ``role='compaction'`` transcript row, plus any
     tool_calls anchored to one (the FK has no ON DELETE CASCADE, so children
     go first).

Delete-only, no backfill: the old per-turn checkpoints were written under the
single-turn model and do not map cleanly onto either new watermark axis. The
first compaction after this migration writes a fresh checkpoint into the new
table; until then a thread simply reads its full history.

Idempotent — safe to run more than once.

Usage: ``python backend/migrations/migration_002_compactions_table.py``
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS compactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT NOT NULL,
    for_turn_id     INTEGER,
    compacted_up_to INTEGER NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_compactions_scope "
    "ON compactions(channel, for_turn_id, id)"
)


def apply(db_path: str) -> None:
    """Create the compactions table and purge legacy compaction rows."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        # Children before parent: tool_calls FK-references transcript with no
        # ON DELETE CASCADE, so drop any audit rows anchored to a compaction
        # row before the row itself.
        conn.execute(
            "DELETE FROM tool_calls WHERE transcript_id IN "
            "(SELECT id FROM transcript WHERE role = 'compaction')"
        )
        purged = conn.execute(
            "DELETE FROM transcript WHERE role = 'compaction'"
        ).rowcount
        conn.commit()
        print(
            f"[migration_002] compactions table ready; purged {purged} "
            f"legacy compaction row(s) ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

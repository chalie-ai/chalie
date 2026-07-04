"""Migration 002 — move compaction state into the ``compactions`` table.

The Thread Context & Compaction Model replaces the old ``role='compaction'``
transcript rows with a dedicated ``compactions`` table carrying a two-axis
watermark (MAIN cuts on the turn_id axis, FORK on the transcript.id axis). This
migration purges every legacy ``role='compaction'`` transcript row, plus any
tool_calls anchored to one (the FK has no ON DELETE CASCADE, so children go
first).

Delete-only — migrations never create schema. The ``compactions`` table is
declared in ``schema.sql`` and created by the boot schema convergence, and
compaction self-heals its data: the first compaction after this purge writes a
fresh checkpoint into that table; until then a thread simply reads its full
history. No backfill — the old per-turn checkpoints were written under the
single-turn model and do not map cleanly onto either new watermark axis.

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


def apply(db_path: str) -> None:
    """Purge legacy ``role='compaction'`` transcript rows (delete-only)."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
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
            f"[migration_002] purged {purged} legacy compaction row(s) ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

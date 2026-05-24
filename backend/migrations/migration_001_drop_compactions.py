"""Migration 001 — drop the compactions table.

The `compactions` table is replaced by append-only `tool_calls` rows with
`tool_name='compaction'`.  `schema.sql` no longer declares `compactions`, so
SchemaConvergenceService will drop it automatically on the next boot
(CHALIE_SCHEMA_ALLOW_DESTRUCTIVE=1, which is the default).

This file documents the intent and provides a standalone idempotent script
for operators who want to apply the drop manually (e.g. in production before
restarting with the new code).

No backfill is needed — every historical compaction summary is already present
in `tool_calls` as an ephemeral=0 audit row written by `_run_full_compaction`.
The canonical lookup in `compaction_persistence.get_compaction()` now reads
from that table.

Usage (manual, standalone):
    python backend/migrations/migration_001_drop_compactions.py [path/to/chalie.db]
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
    """Drop the compactions table.  Idempotent — safe to run more than once."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("DROP TABLE IF EXISTS compactions")
        conn.commit()
        print(f"[migration_001] DROP TABLE IF EXISTS compactions — done ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else str(FileMapperService.get_db_path())
    apply(_path)

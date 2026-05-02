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

import sqlite3
import sys
from pathlib import Path


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
    _default_db = str(
        Path(__file__).resolve().parent.parent / "data" / "chalie.db"
    )
    _path = sys.argv[1] if len(sys.argv) > 1 else _default_db
    apply(_path)

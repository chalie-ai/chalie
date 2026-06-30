"""Migration 004 — drop the scheduled_items quiet-hours window columns.

The `window_start` / `window_end` columns existed solely to gate firing during
an item's active window (quiet hours). That behaviour is removed — the scheduler
no longer honours quiet hours — so nothing reads either column. `schema.sql` no
longer declares them, so SchemaConvergenceService drops them automatically on the
next boot (`CHALIE_SCHEMA_ALLOW_DESTRUCTIVE=1`, the default). This file is the
standalone idempotent script for operators who want to apply the drop manually.

No backfill is needed — the dropped data gated removed behaviour and has no
remaining consumer.

Usage: `python backend/migrations/migration_004_drop_scheduled_window_columns.py [DB_PATH]`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_STALE_COLUMNS = ("window_start", "window_end")


def apply(db_path: str) -> None:
    """Drop the stale quiet-hours window columns. Idempotent — safe to re-run."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        live = {row[1] for row in conn.execute("PRAGMA table_info(scheduled_items)")}
        for col in _STALE_COLUMNS:
            if col in live:
                conn.execute(f"ALTER TABLE scheduled_items DROP COLUMN {col}")
                print(f"[migration_004] DROP COLUMN scheduled_items.{col} — done ({db_path})")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else str(FileMapperService.get_db_path())
    apply(_path)

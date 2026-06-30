"""Migration 005 — add scheduled_items.turn_id column.

Each prompt schedule is now a growing, inspectable thread on the `schedule`
channel: one `turn_id` per series (keyed by `COALESCE(group_id, id)`), allocated
on the first fire and shared by every occurrence and user reply. `schema.sql`
declares the column; SchemaConvergenceService adds it automatically on the next
boot. This file is the standalone idempotent script for operators applying the
change manually.

No backfill is needed — NULL is the correct initial value: `turn_id` is
allocated lazily on the first fire (a schedule that has never fired has no
thread). The gist is stored in `thread_gist` (keyed by channel+turn_id), NOT
on the scheduled_items row.

Usage: `python backend/migrations/migration_005_scheduled_thread_columns.py [DB_PATH]`
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
    """Add the schedule-thread turn_id column. Idempotent — safe to re-run."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        live = {row[1] for row in conn.execute("PRAGMA table_info(scheduled_items)")}
        if "turn_id" not in live:
            conn.execute("ALTER TABLE scheduled_items ADD COLUMN turn_id INTEGER")
            print(f"[migration_005] ADD COLUMN scheduled_items.turn_id — done ({db_path})")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else str(FileMapperService.get_db_path())
    apply(_path)

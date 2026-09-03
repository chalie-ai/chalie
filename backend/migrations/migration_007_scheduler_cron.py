"""Migration 007 — scheduler cron model.

The scheduler moved from a single keyword ``recurrence`` string + a
notification/prompt ``item_type`` toggle to a day-of-month / hour / minute
crontab plus a ``start_at`` activation floor. There is NO backwards
compatibility: the old rows have no cron representation, so this migration
simply wipes ``scheduled_items`` (and its vector shadow) and lets the scheduler
start fresh. The column reshape itself (add start_at/cron_*, drop
recurrence/is_prompt) is declarative — the release's database file is built from
``schema.sql`` — and this script only clears the now-incompatible rows. The wipe is UNCONDITIONAL — every call deletes every scheduled_items row —
so it must run exactly once; the run-once gate (the migrations/runner.py ledger)
lives in the caller, not here, and needed() only answers True while the table
still carries the pre-cron TEXT ``id`` shape. Do not re-invoke it against a
database whose rows the scheduler has already re-materialised.

Usage: `python backend/migrations/migration_007_scheduler_cron.py`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from migrations.connection import connect  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402


def needed(conn: sqlite3.Connection) -> bool:
    """Rows present under the pre-cron scheduler shape? migration_008's rebuild
    declares ``id`` INTEGER PRIMARY KEY and nothing retypes a live column, so a
    TEXT ``id`` reliably marks rows that predate the cron model. A current-shape
    or empty table needs no wipe."""
    cols = {
        row[1]: str(row[2] or "").upper()
        for row in conn.execute("PRAGMA table_info(scheduled_items)")
    }
    if cols.get("id") != "TEXT":
        return False
    return conn.execute("SELECT 1 FROM scheduled_items LIMIT 1").fetchone() is not None


def apply(db_path: str) -> None:
    """Wipe ALL scheduled_items rows unconditionally (no back-compat reshape).

    NOT self-idempotent: re-running deletes any rows the scheduler has since
    created. The run-once gate lives in the boot caller, not this script.
    """
    conn = connect(db_path)
    try:
        conn.execute("DELETE FROM scheduled_items")
        # Clear the vector shadow table too, if present, so no orphan embeddings
        # survive the wipe.
        has_vec = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='scheduled_items_vec'"
        ).fetchone()
        if has_vec:
            conn.execute("DELETE FROM scheduled_items_vec")
        conn.commit()
        print(f"[migration_007] Wiped scheduled_items for fresh cron model ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

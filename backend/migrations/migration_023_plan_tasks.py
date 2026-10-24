"""Migration 023 — add the ``plan_tasks`` table.

Each message the agent works on gets a short plan: one row per task, keyed by
the message's input transcript row. Databases provisioned before plans existed
lack the table; fresh installs get it from ``schema.sql``.

Idempotent — the CREATE is guarded by a sqlite_master presence check, so fresh
installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_023_plan_tasks.py``
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

_CREATE = """
CREATE TABLE IF NOT EXISTS plan_tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcript(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    title         TEXT    NOT NULL,
    task          TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'todo'
                  CHECK (status IN ('todo', 'in_progress', 'done', 'cancelled')),
    UNIQUE (transcript_id, position)
)
"""


def needed(conn: sqlite3.Connection) -> bool:
    """Table still absent? Only databases predating plans carry the gap."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'plan_tasks'"
    ).fetchone()
    return row is None


def apply(db_path: str) -> None:
    """Create ``plan_tasks``. Idempotent — no-op when already present."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute(_CREATE)
            conn.commit()
            print(f"[migration_023] created plan_tasks ({db_path})")
        else:
            print(f"[migration_023] plan_tasks already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

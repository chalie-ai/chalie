"""Migration 024 — add the ``agents`` table.

One row per subagent spawned: the number the model uses for it, who spawned it,
its task and how it ended. Databases provisioned before the table existed lack
it; fresh installs get it from ``schema.sql``.

Idempotent — the CREATE is guarded by a sqlite_master presence check, so fresh
installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_024_agents.py``
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
CREATE TABLE IF NOT EXISTS agents (
    agent_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    spawned_by_channel TEXT    NOT NULL,
    spawned_in_turn    INTEGER NOT NULL,
    type               TEXT    NOT NULL,
    prompt             TEXT    NOT NULL,
    status             TEXT    NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running', 'done', 'cancelled', 'failed')),
    response_id        INTEGER REFERENCES transcript(id) ON DELETE SET NULL,
    turn_id            INTEGER NOT NULL
)
"""

_INDEX = "CREATE INDEX IF NOT EXISTS idx_agents_type_turn ON agents(type, turn_id)"


def needed(conn: sqlite3.Connection) -> bool:
    """Table still absent? Only databases predating it carry the gap."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agents'").fetchone()
    return row is None


def apply(db_path: str) -> None:
    """Create ``agents`` and its index. Idempotent — no-op when already present."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute(_CREATE)
            conn.execute(_INDEX)
            conn.commit()
            print(f"[migration_024] created agents ({db_path})")
        else:
            print(f"[migration_024] agents already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

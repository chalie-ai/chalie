"""Migration 011 — drop the zombie ``tool_calls.invoked_by`` column.

No current code writes or reads ``invoked_by``; it lingers only on databases
created while ``schema.sql`` still declared it. Formalises the former inline
boot wrapper (ledger key ``tool-calls-drop-invoked-by-v1``).

Idempotent — the DROP is guarded by a column-presence check.

Usage: ``python backend/migrations/migration_011_drop_tool_calls_invoked_by.py``
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
    """Zombie column still present? Only databases predating its removal from
    ``schema.sql`` carry it."""
    return "invoked_by" in {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}


def apply(db_path: str) -> None:
    """Drop ``tool_calls.invoked_by``. Idempotent — no-op when already gone."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute("ALTER TABLE tool_calls DROP COLUMN invoked_by")
            conn.commit()
            print(f"[migration_011] dropped zombie tool_calls.invoked_by ({db_path})")
        else:
            print(f"[migration_011] tool_calls.invoked_by already gone — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

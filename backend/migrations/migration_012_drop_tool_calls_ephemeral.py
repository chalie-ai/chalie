"""Migration 012 — drop the legacy ``tool_calls.ephemeral`` column.

Every tool call is now durable: rows older than the retention window are
removed by ``DecayEngineService._purge_tool_calls()`` instead of being flagged
ephemeral at write time. The column lingers only on databases created while
``schema.sql`` still declared it. Formalises the former inline boot wrapper
(ledger key ``tool-calls-drop-ephemeral-v1``).

Idempotent — the DROP is guarded by a column-presence check.

Usage: ``python backend/migrations/migration_012_drop_tool_calls_ephemeral.py``
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
    """Legacy column still present? Only databases predating durable retention
    carry it."""
    return "ephemeral" in {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}


def apply(db_path: str) -> None:
    """Drop ``tool_calls.ephemeral``. Idempotent — no-op when already gone."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute("ALTER TABLE tool_calls DROP COLUMN ephemeral")
            conn.commit()
            print(f"[migration_012] dropped tool_calls.ephemeral (durable retention) ({db_path})")
        else:
            print(f"[migration_012] tool_calls.ephemeral already gone — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

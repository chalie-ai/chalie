"""Migration 022 — add ``transcript.joined`` column.

A message sent into a thread while its turn is still working joins that turn
instead of opening a new one; its user row is stamped ``joined = 1`` so readers
can tell it from the user row that opens a reply. Every existing row predates
joins, so the ``0`` default is correct for all of them.

Idempotent — the ADD COLUMN is guarded by a PRAGMA table_info presence check,
so fresh installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_022_transcript_joined.py``
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
    """Column still absent? Only databases predating joined messages carry the gap."""
    return "joined" not in {row[1] for row in conn.execute("PRAGMA table_info(transcript)")}


def apply(db_path: str) -> None:
    """Add ``transcript.joined``. Idempotent — no-op when already present."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute(
                "ALTER TABLE transcript ADD COLUMN joined INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
            print(f"[migration_022] added transcript.joined (default 0) ({db_path})")
        else:
            print(f"[migration_022] transcript.joined already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

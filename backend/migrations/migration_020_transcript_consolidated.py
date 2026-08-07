"""Migration 020 — add transcript.consolidated column.

The consolidator no longer gates on memory_map provenance. Instead each
transcript row carries its own ``consolidated`` flag (0 = not yet distilled,
1 = the consolidator has already run over it). This column lets the
consolidator service query a window of unconsolidated rows directly, without
cross-table read-through ``memory_map.sourced_from``.

``schema.sql`` declares the column; SchemaConvergenceService adds it
automatically on the next boot. This file is the standalone idempotent
script for operators applying the change manually.

Usage: ``python backend/migrations/migration_020_transcript_consolidated.py``
"""

from __future__ import annotations

import os
import sqlite3
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from migrations.connection import connect  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402


def needed(conn: sqlite3.Connection) -> bool:
    """Column still absent? Only databases predating the consolidator redesign
    carry the gap."""
    return "consolidated" not in {row[1] for row in conn.execute("PRAGMA table_info(transcript)")}


def apply(db_path: str) -> None:
    """Add transcript.consolidated. Idempotent — no-op when already present."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute(
                "ALTER TABLE transcript ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
            print(f"[migration_020] added transcript.consolidated ({db_path})")
        else:
            print(f"[migration_020] transcript.consolidated already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

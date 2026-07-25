
"""Migration 016 — add ``providers.context_window`` column.

Every provider now carries its model's maximum context window (in tokens) so
the runtime can size requests without guessing. The column is nullable: a NULL
value means "ask the client" (Gemini and Ollama already query their APIs for
it at call time; OpenAI falls back to its 128k default).

Idempotent — the ADD COLUMN is guarded by a PRAGMA table_info presence check,
so fresh installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_016_providers_context_window.py``
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
    """Column still absent? Only databases predating the context-window field
    carry the gap."""
    return "context_window" not in {row[1] for row in conn.execute("PRAGMA table_info(providers)")}


def apply(db_path: str) -> None:
    """Add ``providers.context_window``. Idempotent — no-op when already present."""
    conn = connect(db_path)
    try:
        if needed(conn):
            conn.execute(
                "ALTER TABLE providers ADD COLUMN context_window INTEGER"
            )
            conn.commit()
            print(f"[migration_016] added providers.context_window (nullable) ({db_path})")
        else:
            print(f"[migration_016] providers.context_window already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

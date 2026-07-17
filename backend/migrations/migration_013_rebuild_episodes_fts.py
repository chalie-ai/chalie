"""Migration 013 — rebuild ``episodes_fts`` from its content table.

One-time re-sync of the external-content FTS5 index with ``episodes`` rows
written before the index (re)shipped. Formalises the former inline boot
wrapper (ledger key ``episodes-fts-rebuild-v1``).

Idempotent — the FTS5 ``rebuild`` command reconstructs the whole index from
the content table on every run.

Usage: ``python backend/migrations/migration_013_rebuild_episodes_fts.py``
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
    """Any episodes to index? An empty content table has nothing to rebuild,
    so fresh installs skip this."""
    return conn.execute("SELECT 1 FROM episodes LIMIT 1").fetchone() is not None


def apply(db_path: str) -> None:
    """Rebuild the ``episodes_fts`` index from the ``episodes`` content table."""
    conn = connect(db_path)
    try:
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
        conn.commit()
        print(f"[migration_013] episodes_fts rebuilt from content table ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

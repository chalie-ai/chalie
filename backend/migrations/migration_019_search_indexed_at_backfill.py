"""Migration 019 — backfill ``indexed_at`` on ``episodes`` and ``data_graph``.

Provisioning rebuilds ``episodes_fts`` and ``data_graph_fts`` from their content
tables (the old ``search_queries`` column was replaced by ``indexed_at TEXT
DEFAULT NULL``). The rebuild re-posts every row from the content table, but the
new ``indexed_at`` column is NULL on every replayed row — the rebuild only writes
FTS5 postings, not the base-table timestamp.

``SearchExpanderService``'s self-heal invariant is: ``indexed_at IS NULL``
means the row was never posted. Left NULL, its self-heal would INSERT a second
FTS5 posting for every row the rebuild already indexed, producing duplicates
that corrupt future external-content deletes (the delete-before-first-insert
path in ``_mark_indexed``). This migration closes that window.

Runs AFTER provisioning (so the FTS rebuilds are done) and BEFORE the worker's
self-heal (so the invariant holds by the time the worker starts).

Idempotent: a re-run matches nothing (all live rows already have a non-NULL
``indexed_at``).

Usage: ``python backend/migrations/migration_019_search_indexed_at_backfill.py``
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

_UPDATE_SQL = (
    "UPDATE {table} SET indexed_at = datetime('now') "
    "WHERE indexed_at IS NULL AND deleted_at IS NULL"
)


def needed(conn: sqlite3.Connection) -> bool:
    """Any live row in ``episodes`` or ``data_graph`` still has ``indexed_at IS
    NULL``? Such a row would trigger the self-heal's duplicate-posting path."""
    for table in ("episodes", "data_graph"):
        if conn.execute(
            f"SELECT 1 FROM {table} "
            "WHERE indexed_at IS NULL AND deleted_at IS NULL LIMIT 1"
        ).fetchone() is not None:
            return True
    return False


def apply(db_path: str) -> None:
    """Stamp ``indexed_at = datetime('now')`` on every live row in both
    tables that still carries NULL. Idempotent: already-stamped rows don't
    match on re-run."""
    conn = connect(db_path)
    try:
        total: int = 0
        for table in ("episodes", "data_graph"):
            cursor = conn.execute(_UPDATE_SQL.format(table=table))
            total += cursor.rowcount
            print(
                f"[migration_019] Updated {cursor.rowcount} row(s) in {table} "
                f"({db_path})"
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

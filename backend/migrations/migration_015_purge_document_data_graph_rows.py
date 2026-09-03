"""Migration 015 — purge orphaned ``data_graph`` rows with ``kind='document'``.

The documents subsystem code was removed in the previous commit; the
``data_graph`` rows themselves survive because they live in the generic
``data_graph`` table, not in the dropped document-specific tables. This
migration removes every such row plus its search-index footprint (FTS5
postings + key/value vec shadows) and any edges referencing it.

The teardown uses ``DataGraphRow._purge_search_index`` (the shared method in
``models/data_graph.py``) to mirror the idiom the deleted ``DocumentRow``
vertical used: per-row FTS delete (with the original column values captured
before deletion) followed by the base table DELETE. This is critical because
FTS5 external-content delete must be told the indexed values *before* the
source row disappears, and the key/value vec shadow lanes (``data_graph_key_vec``,
``data_graph_value_vec``) must also be cleared — orphaned vec rows whose rowids
could later alias onto new ``data_graph`` rows and poison future searches.

Edges are deleted FIRST (before any ``data_graph`` row dies) because the
subquery ``WHERE from_id IN (SELECT id FROM data_graph WHERE kind='document')``
would match nothing after the parent rows are gone.

Idempotent — ``needed()`` is False once all document rows are purged, so
fresh installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_015_purge_document_data_graph_rows.py``
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
    """True when any ``data_graph`` row carries ``kind='document'`` — the
    orphaned rows this migration exists to remove. Fresh installs (no
    document rows) and already-migrated databases answer False."""
    return conn.execute(
        "SELECT 1 FROM data_graph WHERE kind = 'document' LIMIT 1"
    ).fetchone() is not None


def apply(db_path: str) -> None:
    """Purge every ``data_graph`` row with ``kind='document'`` plus its
    search-index footprint and referencing edges. Idempotent: already-purged
    databases match no rows."""
    conn = connect(db_path)
    try:
        # Delete edges FIRST — once the parent rows are gone these
        # subqueries match nothing.
        cursor = conn.execute(
            "DELETE FROM data_graph_edges "
            "WHERE from_id IN (SELECT id FROM data_graph WHERE kind = 'document') "
            "OR to_id IN (SELECT id FROM data_graph WHERE kind = 'document')"
        )
        edges_purged = cursor.rowcount

        # Per-row teardown: mirror DocumentRow.purge_by_document_id's idiom.
        # SELECT the rowid + indexed column values, call _purge_search_index
        # (FTS + vec lanes), then DELETE by rowid. This is critical because
        # FTS5 external-content delete must be told the values before the
        # source row disappears.
        from models.data_graph import DataGraphRow

        rows = conn.execute(
            "SELECT rowid, key, value, kind "
            "FROM data_graph WHERE kind = 'document'"
        ).fetchall()

        for rowid, key, value, kind in rows:
            DataGraphRow._purge_search_index(
                conn, rowid,
                {"key": key, "value": value, "kind": kind},
            )
            conn.execute("DELETE FROM data_graph WHERE rowid = ?", (rowid,))

        conn.commit()
        print(
            f"[migration_015] Purged {len(rows)} document kind data_graph row(s), "
            f"{edges_purged} edge(s) ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

"""Migration 010 — mark pre-existing episodes as already FTS-indexed.

Episodes now route their full-text posting through ``SearchExpanderService``
(the declaration-driven engine ``data_graph`` already uses): the new
``episodes.search_queries`` column holds the doc2query keyword variants and the
``episodes_fts`` table gains a matching ``search_queries`` column. Schema
convergence performs the structural change at boot — it adds the column and,
because the FTS column list changed, DROP+CREATE+``rebuild``s ``episodes_fts``,
repopulating the ``gist`` postings from the content table. That rebuild leaves
every pre-existing row's ``search_queries`` NULL.

The engine's invariant is ``search_queries IS NULL`` ⟺ *never indexed*, and its
self-heal re-enqueues NULL rows with a delete-before-first-insert guard keyed on
that column. If the convergence rebuild (which already wrote the ``gist``
postings) leaves ``search_queries`` NULL, self-heal would treat those rows as
un-indexed and INSERT a *second* ``gist`` posting — a duplicate that corrupts
future external-content deletes. This migration closes that window: it stamps
every still-NULL live episode's ``search_queries`` to '' — a non-NULL
"indexed, no variants" marker that matches the empty ``search_queries`` value the
rebuild indexed (NULL and '' both tokenize to nothing, so a later hard-delete
supplying '' removes the posting cleanly). Self-heal then skips these rows; new
episodes flow through the engine and earn their doc2query variants.

Runs AFTER schema convergence and BEFORE the worker's self-heal (run.py boot
order: converge → startup migrations → workers). Idempotent: a re-run matches no
rows (they are no longer NULL). The run-once sentinel gate lives in
backend/run.py, not here.

Usage: `python backend/migrations/migration_010_episode_search_queries.py`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_MARK_INDEXED_SQL = (
    "UPDATE episodes SET search_queries = '' "
    "WHERE search_queries IS NULL AND deleted_at IS NULL"
)


def apply(db_path: str) -> None:
    """Mark still-NULL live episodes as FTS-indexed to block a self-heal
    double-post. Idempotent: already-stamped rows no longer match."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cursor = conn.execute(_MARK_INDEXED_SQL)
        conn.commit()
        print(
            f"[migration_010] Marked {cursor.rowcount} pre-existing episode(s) "
            f"as FTS-indexed (search_queries='') ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

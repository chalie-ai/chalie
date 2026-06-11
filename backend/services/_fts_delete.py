"""Shared FTS5 external-content delete idiom.

External-content FTS5 tables (``content='<table>'``) read the indexed columns
from their source table. A plain ``DELETE FROM <fts> WHERE rowid=?`` therefore
re-reads the source row to locate postings; once the source row is gone (or
mismatched) the index strands stale postings and can corrupt. The correct
removal is the FTS5 ``'delete'`` command, which is told the indexed values
explicitly so it never touches the source table.

This helper is the single home for that idiom. It is consumed by:
- ``DataGraphService._delete_fts`` (data_graph_service.py) for ``data_graph_fts``.
- ``DecayEngineService._hard_delete_episode`` (decay_engine_service.py) for
  ``episodes_fts``.
Keep both call sites pointed here so the production-safe path can never drift
between memory tables.
"""

import logging

logger = logging.getLogger(__name__)


def fts5_external_delete(conn, table: str, rowid: int, columns: dict) -> None:
    """Remove one rowid from an external-content FTS5 table.

    Issues the FTS5 ``'delete'`` command with the indexed column values (required
    for external-content tables in production). Falls back to a plain DELETE for
    standalone FTS tables (e.g. test fixtures that omit ``content=``).

    Args:
        conn: Open SQLite connection.
        table: FTS5 table name (e.g. ``episodes_fts``).
        rowid: Source rowid whose postings must be removed.
        columns: Ordered mapping of indexed column name -> stored value. The
            order MUST match the FTS5 column order; values may be ``None`` and
            are coerced to empty strings.
    """
    names = list(columns.keys())
    values = [columns[name] if columns[name] is not None else '' for name in names]
    placeholders = ', '.join(['?'] * (len(names) + 1))
    col_list = ', '.join([table] + names)
    try:
        conn.execute(
            f"INSERT INTO {table}({col_list}) VALUES({placeholders})",
            ['delete', rowid, *values],
        )
    except Exception:
        try:
            conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
        except Exception as e:
            logger.warning("[FTS] delete failed for %s rowid=%s: %s", table, rowid, e)

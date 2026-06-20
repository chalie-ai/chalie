"""Shared FTS5 external-content delete idiom.

A plain ``DELETE FROM <fts> WHERE rowid=?`` on an external-content FTS5
table re-reads the source row to locate postings; once the source row is
gone (or mismatched) the index strands stale postings and can corrupt.
The correct removal is the FTS5 ``'delete'`` command, which is told the
indexed values explicitly so it never touches the source table.

Consumed by ``DataGraphService._delete_fts`` (data_graph_fts) and
``DecayEngineService._hard_delete_episode`` (episodes_fts). Keep both call
sites pointed here so the production-safe path can never drift between
memory tables.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)


def fts5_external_delete(conn: sqlite3.Connection, table: str, rowid: int, columns: dict[str, str | None]) -> None:
    """Issues the FTS5 ``'delete'`` command with the indexed column values
    (required for external-content tables in production). Falls back to a
    plain DELETE for standalone FTS tables (e.g. test fixtures that omit
    ``content=``). columns order MUST match the FTS5 column order; values
    may be None and are coerced to empty strings.
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

"""Shared vec0 embedding upsert idiom.

``INSERT OR REPLACE`` on a vec0 virtual table raises ``UNIQUE constraint
failed on <table> primary key`` whenever the rowid already exists — vec0
implements no conflict resolution (verified against sqlite-vec v0.1.9), so
a "REPLACE" write never updates anything and the table silently keeps the
previous vector once the caller's warning-net absorbs the error. The
correct upsert is DELETE then INSERT, which vec0 supports natively.

Consumed by every ``*_vec`` shadow-table writer (``lists_vec``,
``scheduled_items_vec``). Keep all call sites pointed here so the
production-safe path can never drift between memory tables.
"""

import sqlite3


def vec0_upsert(conn: sqlite3.Connection, table: str, rowid: int, blob: bytes) -> None:
    """Replace ``table``'s embedding row for ``rowid``: DELETE then INSERT
    (vec0 rejects ``INSERT OR REPLACE``). Raises on failure — callers own
    their own warning/transaction policy."""
    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    conn.execute(
        f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
        (rowid, blob),
    )

"""Shared connection opener for migration modules.

Migrations own their connections (standalone-runnable contract), but a bare
``sqlite3.connect`` is not enough on databases whose triggers or shadow tables
reference the ``vec0`` virtual-table module: schema-validating DDL — e.g.
``ALTER TABLE … DROP COLUMN`` re-parses every trigger in the schema — and any
statement touching a ``*_vec`` table fail with "no such module: vec0" unless
the sqlite-vec extension is loaded, exactly as the Database gateway loads it.
The load is best-effort: on installs where sqlite-vec was never available, no
vec objects exist and the plain connection suffices.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)


def connect(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` with the sqlite-vec extension loaded best-effort."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception as exc:
        logger.warning(f"[migrations] sqlite-vec not available: {exc}")
    return conn

"""Migration 014 — restore ``tool_calls.id`` as an AUTOINCREMENT primary key.

On databases created after ``schema.sql`` declared
``id INTEGER PRIMARY KEY AUTOINCREMENT`` the column is a real rowid alias and
every insert hands back an id. On databases that predate it, boot could only
``ALTER TABLE tool_calls ADD COLUMN id INTEGER`` — SQLite cannot add a PRIMARY KEY
(or AUTOINCREMENT) with ALTER — so ``id`` landed as a plain, nullable, non-key
column that nothing ever populates. Every row's id stays
NULL, so ``ToolCallService.start()`` returns None, ``finish()`` no-ops, tool
results are never written back, and the turn re-feeds empty results to the
model — which then repeats the call until the runaway-loop guard trips.

This rebuilds the table into the canonical ``schema.sql`` shape and backfills
each row's id from its stable ``rowid`` (unique by construction), so every
existing call gets a distinct id and future inserts autoincrement past the
high-water mark. ``tool_calls`` is a leaf table — nothing foreign-keys to it —
and carries no triggers or views, so the standard 12-step rebuild is safe.

Idempotent — ``needed()`` is False once ``id`` is the primary key, so fresh
installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_014_tool_calls_id_primary_key.py``
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

# Canonical target, verbatim from schema.sql (structure only — the column
# comments live in schema.sql, the one source of truth for the shape).
_NEW_TABLE_DDL = """
CREATE TABLE tool_calls_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL,
    tool_name     TEXT NOT NULL,
    params        TEXT DEFAULT '{}',
    result        TEXT DEFAULT '',
    summary       TEXT DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at      TEXT,
    state         TEXT NOT NULL DEFAULT 'started'
                  CHECK (state IN ('started', 'done', 'error')),
    FOREIGN KEY (transcript_id) REFERENCES transcript(id)
)
"""

# Every canonical column except id, which is backfilled from rowid. Copied by
# name (never position) so live column order — which drift left different from
# schema.sql — never matters; any declared column the live table happens to
# lack is skipped and takes the new table's default.
_CARRY_COLUMNS = (
    "transcript_id",
    "tool_name",
    "params",
    "result",
    "summary",
    "created_at",
    "ended_at",
    "state",
)


def needed(conn: sqlite3.Connection) -> bool:
    """True when ``tool_calls.id`` is not the primary key — the ADD-COLUMN
    drift this migration exists to undo. Fresh installs (id already the PK)
    and already-migrated databases answer False."""
    id_col = next(
        (row for row in conn.execute("PRAGMA table_info(tool_calls)") if row[1] == "id"),
        None,
    )
    # row[5] is the pk flag: 1 for the primary key column, 0 otherwise.
    return id_col is None or id_col[5] == 0


def apply(db_path: str) -> None:
    """Rebuild ``tool_calls`` with ``id`` as an AUTOINCREMENT primary key,
    backfilling id from rowid. Atomic: on any failure the transaction rolls
    back and the original table is left untouched for the next boot to retry."""
    conn = connect(db_path)
    try:
        if not needed(conn):
            print(f"[migration_014] tool_calls.id already primary key — no-op ({db_path})")
            return

        live_columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
        col_list = ", ".join(c for c in _CARRY_COLUMNS if c in live_columns)

        # Baseline orphan count BEFORE the rebuild. Retention purges transcript
        # rows out from under the tool_calls that referenced them, so a live DB
        # legitimately carries dangling transcript_ids under this same FK. The
        # rebuild copies transcript_id verbatim and so is FK-neutral; the guard
        # below fails loudly only if the rebuild *introduced* new violations —
        # never on pre-existing orphans, which would brick the migration.
        orphans_before = len(conn.execute("PRAGMA foreign_key_check(tool_calls)").fetchall())

        conn.isolation_level = None  # explicit transaction control below
        conn.execute("PRAGMA foreign_keys=OFF")  # must precede BEGIN — a no-op inside a transaction
        conn.execute("BEGIN")
        try:
            conn.execute(_NEW_TABLE_DDL)
            conn.execute(
                f"INSERT INTO tool_calls_new (id, {col_list}) "
                f"SELECT rowid, {col_list} FROM tool_calls"
            )
            conn.execute("DROP TABLE tool_calls")
            conn.execute("ALTER TABLE tool_calls_new RENAME TO tool_calls")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_transcript ON tool_calls(transcript_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC)")
            orphans_after = len(conn.execute("PRAGMA foreign_key_check(tool_calls)").fetchall())
            if orphans_after > orphans_before:
                raise RuntimeError(
                    f"rebuild introduced FK violations: {orphans_before} -> {orphans_after}"
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("PRAGMA foreign_keys=ON")

        rebuilt = conn.execute(
            "SELECT COUNT(*), COUNT(id), COUNT(DISTINCT id) FROM tool_calls"
        ).fetchone()
        print(
            f"[migration_014] tool_calls rebuilt with id primary key — "
            f"{rebuilt[0]} rows, {rebuilt[1]} non-null ids, {rebuilt[2]} distinct ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

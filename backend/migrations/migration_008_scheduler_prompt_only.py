"""Migration 008 — scheduler prompt-only reshape.

The scheduler drops every lifecycle/CalDAV column it ever carried
(``item_type``, ``due_at``, ``status``, ``group_id``, ``turn_id``, ``source``,
``external_uid``, ``metadata``, ``hidden``) and its ``id`` column changes type
from ``TEXT PRIMARY KEY`` to ``INTEGER PRIMARY KEY AUTOINCREMENT`` so that
``id`` can double as the schedule's ``turn_id`` on the ``'schedule'`` channel.
The cron columns also change: the three ``INTEGER`` NULL-means-every fields
(``cron_dom``/``cron_hour``/``cron_minute``) become five ``TEXT`` crontab-
expression fields (adding ``cron_month``/``cron_dow``, ``'*'`` = every), so the
``schedule`` ability supports real crontab (``*/5``, ranges, lists, day-of-week).
SchemaConvergenceService reshapes columns declaratively from ``schema.sql``,
but neither a primary-key type change (TEXT -> INTEGER) nor an in-place column
retype (INTEGER -> TEXT) is within what convergence performs — it ADDs/DROPs
columns and only logs type mismatches, it does not rebuild the table. This
migration does the rebuild explicitly: drop the table and recreate it with
the exact prompt-only DDL. Data is discarded by design — no cron/thread rows
survive the reshape; there is no backwards-compatible mapping from a TEXT id
to a fresh AUTOINCREMENT integer id, nor from an INTEGER cron field to a
crontab expression.

It also purges every ``channel='schedule'`` row from the thread-system tables
(``transcript``, ``thread_gist``, ``turn_executions``, ``compactions``).
Pre-rewrite, the scheduler fired through ``MessageProcessor`` with the ``-1``
sentinel, so ``Transcript.next_turn_id`` allocated a per-channel monotonic
turn_id (1, 2, 3, …) on the ``'schedule'`` channel; those turn rows survive the
``scheduled_items`` rebuild. Under the new model ``turn_id`` IS the schedule
``id``, and the rebuilt ``AUTOINCREMENT`` id restarts at 1 — so without this
purge the first fresh schedule (id=1) would find the stale turn_id=1 row via
``turn_exists()`` and FORK into a dead, unrelated thread instead of opening its
own MAIN turn. Clearing the old ``'schedule'`` turn history closes that
collision. Same "wipe, no backwards compatibility" stance as the table rebuild.

NOT self-idempotent: re-running drops and recreates the table again, deleting
any rows created since the last run. The run-once gate (the migrations/runner.py
ledger) lives in the caller, not here; needed() only answers True while the
table still carries a pre-rebuild column shape.

Usage: `python backend/migrations/migration_008_scheduler_prompt_only.py`
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

_CREATE_TABLE_SQL = """
CREATE TABLE scheduled_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    start_at TEXT NOT NULL,
    cron_minute TEXT NOT NULL DEFAULT '*',
    cron_hour   TEXT NOT NULL DEFAULT '*',
    cron_dom    TEXT NOT NULL DEFAULT '*',
    cron_month  TEXT NOT NULL DEFAULT '*',
    cron_dow    TEXT NOT NULL DEFAULT '*',
    enabled INTEGER NOT NULL DEFAULT 1,
    channel TEXT,
    created_by_session TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_scheduled_items_enabled "
    "ON scheduled_items(enabled, start_at)"
)

# Thread-system tables whose per-channel turn rows must be purged for the
# 'schedule' channel, so the rebuilt AUTOINCREMENT schedule id — which now IS
# the turn_id — cannot collide with a pre-rewrite turn. The delete keys on
# ``channel`` alone (a schedule's whole turn history lives on that channel), so
# the differing turn_id column names (turn_id vs for_turn_id) don't matter here.
# A table missing at apply time is skipped.
_SCHEDULE_TURN_TABLES = ("transcript", "thread_gist", "turn_executions", "compactions")


def needed(conn: sqlite3.Connection) -> bool:
    """Table still in a pre-prompt-only shape? Convergence ADDs/DROPs columns
    but never retypes, so a non-INTEGER ``id`` (the pre-cron TEXT primary key)
    or a non-TEXT ``cron_minute`` (the INTEGER cron fields this rebuild's v2
    re-fire exists to replace) marks a table needing the rebuild. An absent
    table needs nothing — convergence creates it in target shape."""
    cols = {
        row[1]: str(row[2] or "").upper()
        for row in conn.execute("PRAGMA table_info(scheduled_items)")
    }
    if not cols:
        return False
    return cols.get("id") != "INTEGER" or cols.get("cron_minute") != "TEXT"


def apply(db_path: str) -> None:
    """Rebuild scheduled_items as the prompt-only table (TEXT id -> INTEGER id)
    and purge stale ``channel='schedule'`` thread-system turn rows.

    NOT self-idempotent: every call drops and recreates scheduled_items and
    re-purges the schedule turn history, so any rows created since the last run
    are discarded. The run-once gate lives in the boot caller, not this script.
    """
    conn = connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS scheduled_items")
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        has_vec = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='scheduled_items_vec'"
        ).fetchone()
        if has_vec:
            conn.execute("DELETE FROM scheduled_items_vec")
        # Clear the old 'schedule' turn history so fresh ids can't re-enter dead
        # threads. The turn_id column name differs per table; the WHERE on
        # channel makes each delete a no-op where no schedule rows exist.
        for table in _SCHEDULE_TURN_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists:
                conn.execute(f"DELETE FROM {table} WHERE channel='schedule'")
        conn.commit()
        print(
            f"[migration_008] Rebuilt scheduled_items as prompt-only and purged "
            f"stale 'schedule' turn history ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))

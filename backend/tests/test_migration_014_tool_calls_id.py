"""Feature tests — migration 014 (tool_calls id → AUTOINCREMENT primary key).

Recreates the real upgrade drift on a provisioned database: boot could only
``ALTER TABLE tool_calls ADD COLUMN id INTEGER`` on installs that predated the
primary-key declaration, leaving ``id`` a plain nullable non-key
column that every row leaves NULL — so ``start()`` gets no id back, ``finish()``
no-ops, and results never persist. The exact drifted DDL below is copied from
the live server table. Real migration module, real SQLite, zero mocks.
"""

import sqlite3

import pytest

from migrations import migration_014_tool_calls_id_primary_key as mig
from migrations.runner import run_all
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

# Verbatim from the live drifted server table: id is a plain nullable column
# sitting mid-table (ADD COLUMN appends it), the primary key never took, and
# state carries no CHECK. Column order deliberately differs from schema.sql.
_DRIFTED_DDL = """
CREATE TABLE "tool_calls" (
    transcript_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    params TEXT DEFAULT '{}',
    result TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    id INTEGER,
    summary TEXT DEFAULT '',
    ended_at TEXT,
    state TEXT DEFAULT 'started' NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES transcript(id)
)
"""


def _db_path() -> str:
    return str(FileMapperService.get_db_path())


def _id_column(conn: sqlite3.Connection) -> tuple[object, ...]:
    return next(row for row in conn.execute("PRAGMA table_info(tool_calls)") if row[1] == "id")


def _drift(db: sqlite3.Connection, rows: int = 25) -> list[int]:
    """Replace schema.sql's tool_calls with the drifted shape and seed ``rows``
    calls (all with NULL id) against real transcript FK targets. Returns the
    transcript ids used."""
    db.execute("DROP TABLE tool_calls")
    db.execute(_DRIFTED_DDL)
    db.execute("CREATE INDEX idx_tool_calls_transcript ON tool_calls(transcript_id)")
    db.execute("CREATE INDEX idx_tool_calls_created ON tool_calls(created_at DESC)")

    transcript_ids: list[int] = []
    for i in range(3):
        cur = db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', ?)",
            (f"msg {i}",),
        )
        assert cur.lastrowid is not None
        transcript_ids.append(cur.lastrowid)

    states = ("started", "done", "error")
    for i in range(rows):
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, summary, "
            "created_at, ended_at, state, id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                transcript_ids[i % len(transcript_ids)],
                f"tool_{i}",
                f'{{"n": {i}}}',
                f"result-{i}",
                f"doing {i}",
                f"2026-07-13T00:{i:02d}:00",
                None if states[i % 3] == "started" else f"2026-07-13T00:{i:02d}:05",
                states[i % 3],
            ),
        )
    db.commit()
    return transcript_ids


class TestNeeded:
    def test_true_on_drifted_shape(self, db: sqlite3.Connection) -> None:
        _drift(db)
        assert mig.needed(db) is True

    def test_false_on_fresh_schema_shape(self, db: sqlite3.Connection) -> None:
        # The template db already has id as the primary key — nothing to do.
        assert mig.needed(db) is False


class TestApply:
    def test_id_becomes_autoincrement_primary_key(self, db: sqlite3.Connection) -> None:
        _drift(db)
        mig.apply(_db_path())

        id_col = _id_column(db)
        assert id_col[2] == "INTEGER"  # type
        assert id_col[5] == 1  # pk flag — id is now the primary key
        # AUTOINCREMENT is active: sqlite tracks the high-water mark.
        assert db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone() is not None

    def test_every_row_gets_a_unique_non_null_id(self, db: sqlite3.Connection) -> None:
        _drift(db, rows=40)
        mig.apply(_db_path())

        total, non_null, distinct = db.execute(
            "SELECT COUNT(*), COUNT(id), COUNT(DISTINCT id) FROM tool_calls"
        ).fetchone()
        assert total == 40
        assert non_null == 40  # zero NULLs — the whole point
        assert distinct == 40  # every id unique

    def test_row_values_survive_rebuild(self, db: sqlite3.Connection) -> None:
        _drift(db, rows=10)
        before = db.execute(
            "SELECT tool_name, params, result, summary, created_at, ended_at, state "
            "FROM tool_calls ORDER BY rowid"
        ).fetchall()

        mig.apply(_db_path())

        after = db.execute(
            "SELECT tool_name, params, result, summary, created_at, ended_at, state "
            "FROM tool_calls ORDER BY id"
        ).fetchall()
        assert after == before

    def test_indexes_and_integrity_preserved(self, db: sqlite3.Connection) -> None:
        _drift(db)
        mig.apply(_db_path())

        index_names = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tool_calls'"
            )
        }
        assert {"idx_tool_calls_transcript", "idx_tool_calls_created"} <= index_names
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    def test_new_insert_receives_autoincremented_id(self, db: sqlite3.Connection) -> None:
        # The real bug's blast radius: after the fix, an insert without an id
        # (what ToolCallService.start() issues) must hand one back so finish()
        # can write the result. Before the fix lastrowid pointed at rowid but
        # id stayed NULL; now id IS rowid.
        transcript_ids = _drift(db)
        mig.apply(_db_path())

        cur = db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, state) VALUES (?, 'web_search', 'started')",
            (transcript_ids[0],),
        )
        new_id = cur.lastrowid
        db.commit()
        assert new_id is not None
        stored = db.execute("SELECT id FROM tool_calls WHERE id = ?", (new_id,)).fetchone()
        assert stored is not None and stored[0] == new_id
        # finish() can now find the row and write its result.
        db.execute("UPDATE tool_calls SET result = 'hit', state = 'done' WHERE id = ?", (new_id,))
        db.commit()
        written = db.execute("SELECT result, state FROM tool_calls WHERE id = ?", (new_id,)).fetchone()
        assert tuple(written) == ("hit", "done")


class TestIdempotence:
    def test_apply_twice_is_stable(self, db: sqlite3.Connection) -> None:
        _drift(db, rows=15)
        mig.apply(_db_path())
        first = db.execute("SELECT id FROM tool_calls ORDER BY id").fetchall()

        assert mig.needed(db) is False  # second pass sees nothing to do
        mig.apply(_db_path())
        second = db.execute("SELECT id FROM tool_calls ORDER BY id").fetchall()
        assert second == first

    def test_fresh_db_is_untouched(self, db: sqlite3.Connection) -> None:
        before = _id_column(db)
        mig.apply(_db_path())
        assert _id_column(db) == before  # no-op — id was already the pk


class TestRunnerIntegration:
    def test_runner_applies_and_ledgers_the_step(self, db: sqlite3.Connection) -> None:
        _drift(db)
        run_all(_db_path())

        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'tool-calls-id-primary-key-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "applied"
        assert _id_column(db)[5] == 1  # id is the primary key after the runner pass

    def test_runner_noops_on_fresh_db(self, db: sqlite3.Connection) -> None:
        run_all(_db_path())
        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'tool-calls-id-primary-key-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "noop"

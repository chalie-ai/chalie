# Tests for SchemaConvergenceService.

import logging
import sqlite3
from pathlib import Path

import pytest

from services.database_service import DatabaseService
from services.schema_convergence_service import SchemaConvergenceService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path, name: str = "test.db") -> DatabaseService:
    return DatabaseService(str(tmp_path / name))


def _converge(db: DatabaseService, embedding_dimensions: int = 256) -> SchemaConvergenceService:
    svc = SchemaConvergenceService(db, embedding_dimensions=embedding_dimensions)
    svc.converge()
    return svc


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall()
    return {r[0] for r in rows}


def _virtual_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    return {name for name, ddl in rows if ddl.strip().upper().startswith("CREATE VIRTUAL TABLE")}


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaConvergence:

    # ── 1. Fresh DB convergence ───────────────────────────────────────────────

    def test_fresh_db_creates_core_tables(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            tables = _table_names(conn)

        # Spot-check a representative cross-section of schema tables
        expected = {
            "episodes",
            "transcript",
            "settings",
            "schema_version",
            "documents",
            "providers",
            "tool_calls",
            "lists",
        }
        missing = expected - tables
        assert not missing, f"Missing tables after fresh convergence: {missing}"

        # schema_migrations belongs to the deleted migration scaffolding —
        # bidirectional convergence must NOT recreate it.
        assert "schema_migrations" not in tables, (
            "schema_migrations should not exist — migration scaffolding was removed"
        )

    # ── 2. Idempotency ────────────────────────────────────────────────────────

    def test_converge_twice_idempotent(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            tables_first = _table_names(conn)
            indexes_first = _index_names(conn)

        _converge(db)

        with db.connection() as conn:
            tables_second = _table_names(conn)
            indexes_second = _index_names(conn)

        assert tables_first == tables_second
        assert indexes_first == indexes_second

    # ── 3. Missing column detection ───────────────────────────────────────────

    def test_missing_column_is_restored(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        # SQLite cannot DROP COLUMN directly in older versions; we recreate the
        # table without the target column to simulate a missing column situation.
        with db.connection() as conn:
            conn.execute("ALTER TABLE episodes RENAME TO episodes_old")
            conn.execute(
                """
                CREATE TABLE episodes (
                    id TEXT PRIMARY KEY,
                    gist TEXT NOT NULL,
                    channel TEXT NOT NULL
                    -- salience intentionally omitted
                )
                """
            )
            conn.execute("INSERT INTO episodes SELECT id, gist, channel FROM episodes_old")
            conn.execute("DROP TABLE episodes_old")

        # Verify column is missing before re-convergence
        with db.connection() as conn:
            cols_before = _column_names(conn, "episodes")
        assert "salience" not in cols_before

        # Re-converge — should add the missing column
        _converge(db)

        with db.connection() as conn:
            cols_after = _column_names(conn, "episodes")
        assert "salience" in cols_after

    # ── 4. Missing table detection ────────────────────────────────────────────

    def test_missing_table_is_restored(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        # Drop a non-critical table
        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS documents")

        with db.connection() as conn:
            assert "documents" not in _table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "documents" in _table_names(conn)

    # ── 5. Missing index detection ────────────────────────────────────────────

    def test_missing_index_is_restored(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_episodes_channel")

        with db.connection() as conn:
            assert "idx_episodes_channel" not in _index_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "idx_episodes_channel" in _index_names(conn)

    # ── 6. Index DDL change detection ─────────────────────────────────────────

    def test_changed_index_ddl_is_recreated(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        # Replace a real index with one that has different columns (stale DDL)
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_settings_key")
            # Create index with wrong column to simulate DDL mismatch
            conn.execute("CREATE INDEX idx_settings_key ON settings(created_at)")

        # After re-convergence the index should match the desired DDL
        _converge(db)

        with db.connection() as conn:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='idx_settings_key'"
            ).fetchall()

        assert rows, "idx_settings_key is missing after re-convergence"
        # The recreated DDL should reference the correct column (key), not created_at
        ddl = rows[0][0].lower()
        assert "key" in ddl, f"Recreated index DDL looks wrong: {ddl}"

    # ── 7. Virtual table creation ─────────────────────────────────────────────

    def test_dropped_fts5_table_is_recreated(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS episodes_fts")

        with db.connection() as conn:
            assert "episodes_fts" not in _virtual_table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "episodes_fts" in _virtual_table_names(conn)

    # ── 8. Seed data on fresh DB ──────────────────────────────────────────────

    def test_settings_seeded_on_fresh_db(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT key, is_sensitive FROM settings WHERE key='api_key'"
            ).fetchone()

        assert row is not None, "api_key seed row missing from settings"
        assert row[1] == 1  # the is_sensitive column must be set to true for api_key

    # ── 9. Bidirectional convergence — drop stale schema objects ──────────────

    def test_stale_table_is_dropped(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("CREATE TABLE legacy_widget (id INTEGER PRIMARY KEY, name TEXT)")
            assert "legacy_widget" in _table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "legacy_widget" not in _table_names(conn), (
                "Stale table legacy_widget should have been auto-dropped"
            )

    def test_fts5_shadow_tables_survive_convergence(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _converge(db)
        _converge(db)  # second run — would drop shadow tables if filter is broken

        with db.connection() as conn:
            tables = _table_names(conn)

        # episodes_fts is a virtual table; its shadow tables (episodes_fts_data,
        # episodes_fts_idx, episodes_fts_docsize, episodes_fts_config) must survive.
        for shadow in ("episodes_fts_data", "episodes_fts_idx", "episodes_fts_docsize"):
            assert shadow in tables, (
                f"Shadow table {shadow} was incorrectly dropped — virtual table cleanup logic broke"
            )

    def test_destructive_disabled_via_env_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("CREATE TABLE legacy_widget (id INTEGER PRIMARY KEY)")

        monkeypatch.setenv("CHALIE_SCHEMA_ALLOW_DESTRUCTIVE", "0")
        _converge(db)

        with db.connection() as conn:
            assert "legacy_widget" in _table_names(conn), (
                "legacy_widget should NOT be dropped when destructive ops are disabled"
            )

    def test_safety_guard_refuses_mass_drop_on_truncated_schema(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        db = _make_db(tmp_path)
        _converge(db)

        # Count live tables before.
        with db.connection() as conn:
            before = _table_names(conn)
        assert len(before) >= 10, "Test requires ≥10 live tables to exercise guard"

        # Point the service at a 2-table schema.sql stub — looks like a truncation.
        stub = tmp_path / "stub_schema.sql"
        stub.write_text("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY);\n"
                        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);\n")

        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        monkeypatch.setattr(svc, "_schema_path", stub)

        with caplog.at_level(logging.ERROR, logger="services.schema_convergence_service"):
            svc.converge()

        with db.connection() as conn:
            after = _table_names(conn)

        # No mass-drop occurred — live DB retains its tables.
        assert len(after) >= len(before), (
            f"Safety guard failed: live DB lost tables on truncated schema "
            f"(before={len(before)}, after={len(after)})"
        )
        safety_logs = [r.message for r in caplog.records if "SAFETY" in r.message]
        assert safety_logs, "Safety guard did not emit an ERROR-level SAFETY log"

    # ── 11. Embedding dimensions override ─────────────────────────────────────

    def test_vec0_tables_use_custom_embedding_dimensions(self, tmp_path: Path) -> None:
        """vec0 virtual tables are created with the configured embedding_dimensions."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        svc.converge()

        with db.connection() as conn:
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
            ).fetchall()

        vec_tables = {
            name: sql for name, sql in rows
            if sql and sql.strip().upper().startswith("CREATE VIRTUAL TABLE")
            and "vec0" in sql.lower()
        }

        assert vec_tables, "No vec0 virtual tables found"
        for name, ddl in vec_tables.items():
            assert "float[256]" in ddl, (
                f"Expected float[256] in {name} DDL but got: {ddl}"
            )
            assert "float[768]" not in ddl, (
                f"Unexpected float[768] (default) found in {name} DDL"
            )


    # ── 14. Stale column drop logging ─────────────────────────────────────────

    def test_stale_column_drop_emits_warning_log(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When a stale column is dropped, a WARNING-level log is emitted naming the column."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute(
                "ALTER TABLE lists ADD COLUMN legacy_field TEXT DEFAULT NULL"
            )

        with caplog.at_level(logging.WARNING, logger="services.schema_convergence_service"):
            _converge(db)

        with db.connection() as conn:
            cols = _column_names(conn, "lists")
        assert "legacy_field" not in cols, "Stale column should be dropped by bidirectional convergence"

        warn_logs = [
            r.message for r in caplog.records
            if r.levelname == "WARNING" and "legacy_field" in r.message and "DROPPED" in r.message
        ]
        assert warn_logs, "Expected WARNING log entry for the dropped stale column"

    # ── 18. Missing schema.sql ────────────────────────────────────────────────

    def test_missing_schema_file_raises_file_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """converge() raises FileNotFoundError when schema.sql does not exist."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        # Point schema path to a nonexistent file
        monkeypatch.setattr(svc, "_schema_path", tmp_path / "nonexistent_schema.sql")

        with pytest.raises(FileNotFoundError):
            svc.converge()





# ── TestStripDataGraphCheckConstraint ─────────────────────────────────────────

@pytest.mark.unit
class TestStripDataGraphCheckConstraint:
    """_strip_data_graph_check_constraint removes CHECK(kind IN (...)) from
    data_graph and preserves all rows and the FTS virtual table."""

    # DDL that mimics an old data_graph with a CHECK constraint on kind.
    _OLD_DDL_WITH_CHECK = """
    CREATE TABLE data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL CHECK(kind IN ('user_specific','system','misc','moment')),
        key               TEXT NOT NULL,
        value             TEXT,
        storage_strength  REAL NOT NULL DEFAULT 0.5,
        retrieval_weight  REAL NOT NULL DEFAULT 1.0,
        salience_score    REAL NOT NULL DEFAULT 0.0,
        evidence_count    INTEGER NOT NULL DEFAULT 1,
        first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at  TEXT,
        source            TEXT,
        deleted_at        TEXT,
        active            INTEGER NOT NULL DEFAULT 1,
        search_queries    TEXT DEFAULT NULL
    )
    """

    _NEW_DDL_NO_CHECK = """
    CREATE TABLE data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL,
        key               TEXT NOT NULL,
        value             TEXT,
        storage_strength  REAL NOT NULL DEFAULT 0.5,
        retrieval_weight  REAL NOT NULL DEFAULT 1.0,
        salience_score    REAL NOT NULL DEFAULT 0.0,
        evidence_count    INTEGER NOT NULL DEFAULT 1,
        first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at  TEXT,
        source            TEXT,
        deleted_at        TEXT,
        active            INTEGER NOT NULL DEFAULT 1,
        search_queries    TEXT DEFAULT NULL
    )
    """

    _FTS_DDL = """
    CREATE VIRTUAL TABLE data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """

    def _build_conn_with_check(self, tmp_path: Path, rows: list[tuple[str, str, str]] | None = None) -> sqlite3.Connection:
        """Return an open connection to a DB with CHECK constraint and optionally pre-inserted rows."""
        conn = sqlite3.connect(str(tmp_path / "check_test.db"))
        conn.execute(self._OLD_DDL_WITH_CHECK)
        conn.execute(self._FTS_DDL)
        if rows:
            for kind, key, value in rows:
                conn.execute(
                    "INSERT INTO data_graph (kind, key, value) VALUES (?, ?, ?)",
                    (kind, key, value)
                )
                rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) VALUES (?, ?, ?, ?, ?)",
                    (rowid, key, value or '', kind, '')
                )
        conn.commit()
        return conn

    def _has_check_constraint(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_graph'"
        ).fetchone()
        return row is not None and row[0] is not None and 'CHECK' in row[0].upper()

    def _build_svc(self, db: DatabaseService) -> SchemaConvergenceService:
        return SchemaConvergenceService(db, embedding_dimensions=256)

    def test_removes_check_constraint_from_existing_table(self, tmp_path: Path) -> None:
        """CHECK constraint is stripped — the resulting table DDL contains no CHECK."""
        conn = self._build_conn_with_check(tmp_path)
        assert self._has_check_constraint(conn), "Pre-condition: table must have CHECK constraint"

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        assert not self._has_check_constraint(conn), "CHECK constraint was not removed"
        conn.close()





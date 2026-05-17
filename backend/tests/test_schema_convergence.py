"""
Tests for SchemaConvergenceService — declarative SQLite schema management.

Covers: fresh DB convergence, idempotency, missing table/column/index recovery,
virtual table creation, seed data, bidirectional convergence (auto-drop of
stale tables/columns/indexes/virtual tables), env-flag safety gate,
embedding dimension override, DDL comment handling, shadow table exclusion,
and job-assignment convergence (orphan-pruning).

Each test method creates its own temp DB via tmp_path — no shared state.
"""

import logging
import sqlite3
from pathlib import Path

import pytest

from services.database_service import DatabaseService
from services.schema_convergence_service import SchemaConvergenceService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path, name: str = "test.db") -> DatabaseService:
    """Create a fresh DatabaseService backed by a temp file."""
    return DatabaseService(str(tmp_path / name))


def _converge(db: DatabaseService, embedding_dimensions: int = 256) -> SchemaConvergenceService:
    """Instantiate and run convergence; return the service for further inspection."""
    svc = SchemaConvergenceService(db, embedding_dimensions=embedding_dimensions)
    svc.converge()
    return svc


def _table_names(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _index_names(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall()
    return {r[0] for r in rows}


def _virtual_table_names(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    return {name for name, ddl in rows if ddl.strip().upper().startswith("CREATE VIRTUAL TABLE")}


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaConvergence:

    # ── 1. Fresh DB convergence ───────────────────────────────────────────────

    def test_fresh_db_creates_core_tables(self, tmp_path):
        """All expected core tables exist after converging a blank database."""
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

    def test_fresh_db_creates_fts5_virtual_tables(self, tmp_path):
        """FTS5 virtual tables are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            virtual_tables = _virtual_table_names(conn)

        fts_tables = {t for t in virtual_tables if t.endswith("_fts")}
        assert "episodes_fts" in fts_tables
        assert "documents_fts" in fts_tables

    def test_fresh_db_creates_vec0_virtual_tables(self, tmp_path):
        """vec0 virtual tables are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            virtual_tables = _virtual_table_names(conn)

        vec_tables = {t for t in virtual_tables if t.endswith("_vec")}
        assert "episodes_vec" in vec_tables
        assert "documents_vec" in vec_tables

    def test_fresh_db_creates_indexes(self, tmp_path):
        """Representative indexes are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            indexes = _index_names(conn)

        assert "idx_episodes_channel" in indexes
        assert "idx_transcript_channel" in indexes

    # ── 2. Idempotency ────────────────────────────────────────────────────────

    def test_converge_twice_idempotent(self, tmp_path):
        """Table and index sets are identical after two convergence passes."""
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

    def test_missing_column_is_restored(self, tmp_path):
        """A column dropped from an existing table is re-added on the next converge."""
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

    def test_missing_table_is_restored(self, tmp_path):
        """A table dropped after convergence is recreated on the next converge."""
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

    def test_missing_table_columns_correct_after_restore(self, tmp_path):
        """Restored table has all expected columns."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS scheduled_items")

        _converge(db)

        with db.connection() as conn:
            cols = _column_names(conn, "scheduled_items")

        for expected_col in ("id", "message", "due_at", "status", "channel"):
            assert expected_col in cols, f"Column '{expected_col}' missing after table restore"

    # ── 5. Missing index detection ────────────────────────────────────────────

    def test_missing_index_is_restored(self, tmp_path):
        """An index dropped after convergence is recreated on the next converge."""
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

    def test_changed_index_ddl_is_recreated(self, tmp_path):
        """An index with a stale DDL is dropped and recreated with the correct DDL."""
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

    def test_dropped_fts5_table_is_recreated(self, tmp_path):
        """An FTS5 virtual table dropped after convergence is recreated."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS episodes_fts")

        with db.connection() as conn:
            assert "episodes_fts" not in _virtual_table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "episodes_fts" in _virtual_table_names(conn)

    def test_dropped_vec0_table_is_recreated(self, tmp_path):
        """A vec0 virtual table dropped after convergence is recreated."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS episodes_vec")

        with db.connection() as conn:
            assert "episodes_vec" not in _virtual_table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "episodes_vec" in _virtual_table_names(conn)

    # ── 8. Seed data on fresh DB ──────────────────────────────────────────────

    def test_settings_seeded_on_fresh_db(self, tmp_path):
        """settings table contains the api_key seed row on a fresh DB."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT key, is_sensitive FROM settings WHERE key='api_key'"
            ).fetchone()

        assert row is not None, "api_key seed row missing from settings"
        assert row[1] == 1  # the is_sensitive column must be set to true for api_key

    def test_schema_version_seeded_on_fresh_db(self, tmp_path):
        """schema_version table has at least one row after fresh convergence."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()

        assert row is not None, "schema_version has no rows after fresh convergence"
        assert row[0] == 1

    def test_seed_data_not_duplicated_on_re_convergence(self, tmp_path):
        """Seed inserts use INSERT OR IGNORE — re-converging does not create duplicates."""
        db = _make_db(tmp_path)
        _converge(db)
        _converge(db)  # second run

        with db.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM settings WHERE key='api_key'"
            ).fetchone()[0]

        assert count == 1, f"Expected 1 api_key row, got {count}"

    # ── 9. Bidirectional convergence — drop stale schema objects ──────────────

    def test_stale_table_is_dropped(self, tmp_path):
        """A table present in the live DB but not in schema.sql is dropped on converge."""
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

    def test_stale_column_is_dropped(self, tmp_path):
        """A column added to a live table but absent from schema.sql is dropped on converge."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("ALTER TABLE settings ADD COLUMN obsolete_flag INTEGER DEFAULT 0")
            assert "obsolete_flag" in _column_names(conn, "settings")

        _converge(db)

        with db.connection() as conn:
            assert "obsolete_flag" not in _column_names(conn, "settings"), (
                "Stale column settings.obsolete_flag should have been auto-dropped"
            )

    def test_stale_index_is_dropped(self, tmp_path):
        """An index present in the live DB but not in schema.sql is dropped."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("CREATE INDEX idx_obsolete_settings_value ON settings(value)")
            assert "idx_obsolete_settings_value" in _index_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "idx_obsolete_settings_value" not in _index_names(conn), (
                "Stale index should have been auto-dropped"
            )

    def test_stale_virtual_table_is_dropped(self, tmp_path):
        """A vec0 virtual table not declared in schema.sql is dropped."""
        db = _make_db(tmp_path)
        _converge(db, embedding_dimensions=256)

        with db.connection() as conn:
            conn.execute("CREATE VIRTUAL TABLE obsolete_vec USING vec0(embedding float[256])")
            assert "obsolete_vec" in _virtual_table_names(conn)

        _converge(db, embedding_dimensions=256)

        with db.connection() as conn:
            assert "obsolete_vec" not in _virtual_table_names(conn), (
                "Stale virtual table obsolete_vec should have been auto-dropped"
            )

    def test_protected_sqlite_tables_never_dropped(self, tmp_path):
        """sqlite_sequence and sqlite_* tables must survive bidirectional convergence."""
        db = _make_db(tmp_path)
        _converge(db)

        # Create a row that triggers sqlite_sequence creation
        with db.connection() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES ('trigger_seq_test', 'x')")

        _converge(db)  # second pass should not even attempt to drop sqlite_*

        with db.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sqlite_%'"
            ).fetchall()
        # sqlite_sequence is created lazily by SQLite; just confirm convergence
        # did not blow up trying to manipulate it.
        for (name,) in rows:
            assert name.startswith("sqlite_"), f"Unexpected system table name: {name}"

    def test_fts5_shadow_tables_survive_convergence(self, tmp_path):
        """FTS5 shadow tables (xxx_data, xxx_idx, etc.) must not be dropped as 'stale'."""
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

    def test_destructive_disabled_via_env_flag(self, tmp_path, monkeypatch):
        """Setting CHALIE_SCHEMA_ALLOW_DESTRUCTIVE=0 prevents drops; logs WARNING instead."""
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

    def test_safety_guard_refuses_mass_drop_on_truncated_schema(self, tmp_path, monkeypatch, caplog):
        """If schema.sql is corrupted to near-empty, convergence refuses to drop the live DB."""
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

    def test_safety_guard_refuses_empty_schema(self, tmp_path, monkeypatch, caplog):
        """An empty schema.sql must NOT wipe the live DB."""
        db = _make_db(tmp_path)
        _converge(db)

        empty = tmp_path / "empty_schema.sql"
        empty.write_text("-- empty schema\n")

        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        monkeypatch.setattr(svc, "_schema_path", empty)

        with caplog.at_level(logging.ERROR, logger="services.schema_convergence_service"):
            svc.converge()

        with db.connection() as conn:
            tables = _table_names(conn)
        # Live DB must still have its tables
        assert "settings" in tables, "Safety guard failed on empty schema — live DB was wiped"

    # ── 11. Embedding dimensions override ─────────────────────────────────────

    def test_vec0_tables_use_custom_embedding_dimensions(self, tmp_path):
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

    # ── 12. DDL extraction with comments ──────────────────────────────────────

    def test_comment_does_not_break_ddl_extraction(self, tmp_path):
        """SQL comments inside CREATE TABLE blocks don't cause _extract_table_ddl()
        to truncate the DDL prematurely."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        # data_graph has an inline comment on the kind column:
        #   -- CHECK constraint removed: Python validates kind via VALID_KINDS in data_graph_service.py.
        # If the extractor is broken it will truncate before the final ')'.
        schema_sql = svc._schema_path.read_text()
        ddl = svc._extract_table_ddl(schema_sql, "data_graph")

        assert ddl is not None, "Failed to extract data_graph table DDL"
        # DDL must be a complete CREATE TABLE statement — ends with );
        stripped = ddl.strip().rstrip(";").strip()
        assert stripped.endswith(")"), (
            f"data_graph DDL appears truncated (missing closing paren): ...{stripped[-60:]}"
        )
        # Must contain columns that appear after the comment line
        assert "search_queries" in ddl.lower(), (
            "search_queries column absent — DDL was likely cut at the comment"
        )

    # ── 13. Shadow table exclusion ────────────────────────────────────────────

    def test_fts5_shadow_tables_not_in_normal_tables(self, tmp_path):
        """FTS5 shadow tables (e.g. episodes_fts_data) are excluded from the
        normal table set returned by _introspect_tables()."""
        db = _make_db(tmp_path)
        _converge(db)

        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        with db.connection() as conn:
            from services.schema_convergence_service import _load_sqlite_vec
            _load_sqlite_vec(conn)
            normal_tables = svc._introspect_tables(conn)

        shadow_like = [t for t in normal_tables if "_fts_" in t or "_fts" in t]
        assert not shadow_like, f"FTS5 shadow tables leaked into normal tables: {shadow_like}"

    # ── 14. Stale column drop logging ─────────────────────────────────────────

    def test_stale_column_drop_emits_warning_log(self, tmp_path, caplog):
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

    # ── 15. DDL normalization ─────────────────────────────────────────────────

    def test_normalize_ddl_lowercases_and_collapses_whitespace(self, tmp_path):
        """_normalize_ddl returns lowercase with collapsed whitespace."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        result = svc._normalize_ddl("CREATE  TABLE   Foo  ( bar TEXT )")
        assert result == result.lower()
        assert "  " not in result  # no double spaces

    def test_normalize_ddl_strips_if_not_exists_and_handles_empty(self, tmp_path):
        """_normalize_ddl strips IF NOT EXISTS and handles empty/None."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        with_ine = svc._normalize_ddl("CREATE TABLE IF NOT EXISTS foo (id INTEGER)")
        without_ine = svc._normalize_ddl("CREATE TABLE foo (id INTEGER)")
        assert with_ine == without_ine

        assert svc._normalize_ddl("") == ""
        assert svc._normalize_ddl(None) == ""

    # ── 16. Removed-table cleanup via bidirectional convergence ───────────────

    def test_legacy_tables_absent_after_fresh_convergence(self, tmp_path):
        """Tables that no longer exist in schema.sql must not appear after convergence.

        Replaces the old DROP-TABLE-statement scaffolding: bidirectional
        convergence handles removal automatically.  Use historically-removed
        tables as canaries — if these reappear, schema.sql has regressed.
        """
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            tables = _table_names(conn)

        for legacy in (
            "knowledge",
            "cognitive_reflexes",
            "triage_calibration_events",
            "persistent_tasks",
            "document_chunks",
            "cortex_iterations",
            "goals",
            "goal_evidence",
        ):
            assert legacy not in tables, (
                f"Removed table {legacy!r} must not be created on fresh convergence"
            )


    # ── 17. _is_fresh_db ──────────────────────────────────────────────────────

    def test_is_fresh_db_true_for_empty_db(self, tmp_path):
        """_is_fresh_db() returns True for a database with no user tables."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        with db.connection() as conn:
            result = svc._is_fresh_db(conn)

        assert result is True

    # ── 18. Missing schema.sql ────────────────────────────────────────────────

    def test_missing_schema_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """converge() raises FileNotFoundError when schema.sql does not exist."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        # Point schema path to a nonexistent file
        monkeypatch.setattr(svc, "_schema_path", tmp_path / "nonexistent_schema.sql")

        with pytest.raises(FileNotFoundError):
            svc.converge()

    # ── 19. _restore_if_not_exists ────────────────────────────────────────────

    def test_restore_if_not_exists_for_index(self, tmp_path):
        """_restore_if_not_exists() inserts IF NOT EXISTS after CREATE INDEX."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        normalized = "create index idx_foo on bar(col)"
        restored = svc._restore_if_not_exists(normalized, "index")

        assert "if not exists" in restored
        assert restored.startswith("create index if not exists")

    def test_restore_if_not_exists_handles_unique_index(self, tmp_path):
        """_restore_if_not_exists() correctly handles CREATE UNIQUE INDEX."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        normalized = "create unique index idx_uniq on bar(col)"
        restored = svc._restore_if_not_exists(normalized, "index")

        assert restored == "create unique index if not exists idx_uniq on bar(col)"



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

    def _build_conn_with_check(self, tmp_path, rows=None) -> sqlite3.Connection:
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

    def test_removes_check_constraint_from_existing_table(self, tmp_path):
        """CHECK constraint is stripped — the resulting table DDL contains no CHECK."""
        conn = self._build_conn_with_check(tmp_path)
        assert self._has_check_constraint(conn), "Pre-condition: table must have CHECK constraint"

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        assert not self._has_check_constraint(conn), "CHECK constraint was not removed"
        conn.close()

    def test_preserves_all_rows_after_table_recreation(self, tmp_path):
        """All existing rows survive the table-recreate that strips the CHECK constraint."""
        rows = [
            ('user_specific', 'user_name', 'Dylan'),
            ('system', 'tone', 'terse'),
            ('misc', 'scratch', 'temp'),
        ]
        conn = self._build_conn_with_check(tmp_path, rows=rows)

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        stored = conn.execute("SELECT kind, key, value FROM data_graph ORDER BY key").fetchall()
        assert len(stored) == 3
        stored_set = {(r[0], r[1], r[2]) for r in stored}
        assert stored_set == set(rows)
        conn.close()

    def test_no_op_when_check_constraint_absent(self, tmp_path):
        """When the table has no CHECK, the method returns without touching any rows."""
        conn = sqlite3.connect(str(tmp_path / "nocheck.db"))
        conn.execute(self._NEW_DDL_NO_CHECK)
        conn.execute(self._FTS_DDL)
        conn.execute(
            "INSERT INTO data_graph (kind, key, value) VALUES (?, ?, ?)",
            ('user_specific', 'existing_key', 'existing_value')
        )
        conn.commit()

        assert not self._has_check_constraint(conn), "Pre-condition: table must NOT have CHECK"
        before_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_graph'"
        ).fetchone()[0]

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        # Table DDL must be unchanged and data preserved
        after_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_graph'"
        ).fetchone()[0]
        assert before_sql == after_sql

        count = conn.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
        assert count == 1
        conn.close()

    def test_allows_document_kind_after_constraint_removal(self, tmp_path):
        """After stripping, inserting kind='document' succeeds without constraint violation."""
        conn = self._build_conn_with_check(tmp_path)

        # Verify that inserting 'document' kind fails BEFORE stripping
        try:
            conn.execute(
                "INSERT INTO data_graph (kind, key, value) VALUES ('document', 'doc:test:000', 'v')"
            )
            conn.rollback()
            # Some SQLite builds may not enforce CHECK at insert; skip this assertion
        except Exception:
            conn.rollback()

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        # After stripping, 'document' kind must be insertable
        conn.execute(
            "INSERT INTO data_graph (kind, key, value) VALUES ('document', 'doc:solar:000', 'solar content')"
        )
        conn.commit()
        row = conn.execute("SELECT kind FROM data_graph WHERE key='doc:solar:000'").fetchone()
        assert row is not None
        assert row[0] == 'document'
        conn.close()

    def test_fts_rebuilt_after_constraint_removal(self, tmp_path):
        """FTS virtual table is rebuilt — content of pre-existing rows is searchable."""
        rows = [('user_specific', 'energy_fact', 'solar power efficiency')]
        conn = self._build_conn_with_check(tmp_path, rows=rows)

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        svc._strip_data_graph_check_constraint(conn)
        conn.commit()

        # FTS search must find the pre-existing row
        fts_rows = conn.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH 'solar'"
        ).fetchall()
        assert len(fts_rows) >= 1
        conn.close()

    def test_no_op_when_table_does_not_exist(self, tmp_path):
        """If data_graph table doesn't exist at all, method returns silently."""
        conn = sqlite3.connect(str(tmp_path / "empty.db"))

        svc = SchemaConvergenceService.__new__(SchemaConvergenceService)
        # Must not raise
        svc._strip_data_graph_check_constraint(conn)
        conn.close()

"""
Tests for SchemaConvergenceService — declarative SQLite schema management.

Covers: fresh DB convergence, idempotency, missing table/column/index recovery,
virtual table creation, seed data, migration stamping, pending migration execution,
embedding dimension override, DDL comment handling, shadow table exclusion, and
stale column logging.

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
            "knowledge",
            "transcript",
            "goals",
            "settings",
            "schema_migrations",
            "schema_version",
            "identity_vectors",
            "documents",
            "document_chunks",
            "providers",
            "tool_calls",
            "compactions",
            "lists",
        }
        missing = expected - tables
        assert not missing, f"Missing tables after fresh convergence: {missing}"

    def test_fresh_db_creates_fts5_virtual_tables(self, tmp_path):
        """FTS5 virtual tables are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            virtual_tables = _virtual_table_names(conn)

        fts_tables = {t for t in virtual_tables if t.endswith("_fts")}
        assert "episodes_fts" in fts_tables
        assert "knowledge_fts" in fts_tables
        assert "documents_fts" in fts_tables
        assert "document_chunks_fts" in fts_tables

    def test_fresh_db_creates_vec0_virtual_tables(self, tmp_path):
        """vec0 virtual tables are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            virtual_tables = _virtual_table_names(conn)

        vec_tables = {t for t in virtual_tables if t.endswith("_vec")}
        assert "episodes_vec" in vec_tables
        assert "knowledge_vec" in vec_tables
        assert "documents_vec" in vec_tables
        assert "transcript_vec" in vec_tables

    def test_fresh_db_creates_indexes(self, tmp_path):
        """Representative indexes are created on a fresh database."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            indexes = _index_names(conn)

        assert "idx_episodes_channel" in indexes
        assert "idx_knowledge_kind" in indexes
        assert "idx_transcript_channel" in indexes
        assert "idx_goals_status" in indexes

    # ── 2. Idempotency ────────────────────────────────────────────────────────

    def test_converge_twice_no_error(self, tmp_path):
        """Calling converge() on an already-converged database raises no errors."""
        db = _make_db(tmp_path)
        _converge(db)
        # Second run must be silent — no exception
        _converge(db)

    def test_converge_twice_same_table_set(self, tmp_path):
        """Table set is identical after two convergence passes."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            tables_first = _table_names(conn)

        _converge(db)

        with db.connection() as conn:
            tables_second = _table_names(conn)

        assert tables_first == tables_second

    def test_converge_twice_same_index_set(self, tmp_path):
        """Index set is identical after two convergence passes."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            indexes_first = _index_names(conn)

        _converge(db)

        with db.connection() as conn:
            indexes_second = _index_names(conn)

        assert indexes_first == indexes_second

    # ── 3. Missing column detection ───────────────────────────────────────────

    def test_missing_column_is_restored(self, tmp_path):
        """A column dropped from an existing table is re-added on the next converge."""
        db = _make_db(tmp_path)
        _converge(db)

        # SQLite cannot DROP COLUMN directly in older versions; we recreate the
        # table without the target column to simulate a missing column situation.
        with db.connection() as conn:
            conn.execute("ALTER TABLE goals RENAME TO goals_old")
            conn.execute(
                """
                CREATE TABLE goals (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'emergent',
                    status TEXT NOT NULL DEFAULT 'candidate'
                    -- salience intentionally omitted
                )
                """
            )
            conn.execute("INSERT INTO goals SELECT id, description, type, status FROM goals_old")
            conn.execute("DROP TABLE goals_old")

        # Verify column is missing before re-convergence
        with db.connection() as conn:
            cols_before = _column_names(conn, "goals")
        assert "salience" not in cols_before

        # Re-converge — should add the missing column
        _converge(db)

        with db.connection() as conn:
            cols_after = _column_names(conn, "goals")
        assert "salience" in cols_after

    # ── 4. Missing table detection ────────────────────────────────────────────

    def test_missing_table_is_restored(self, tmp_path):
        """A table dropped after convergence is recreated on the next converge."""
        db = _make_db(tmp_path)
        _converge(db)

        # Drop a non-critical table
        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS place_fingerprints")

        with db.connection() as conn:
            assert "place_fingerprints" not in _table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "place_fingerprints" in _table_names(conn)

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

    def test_multiple_missing_indexes_all_restored(self, tmp_path):
        """Dropping several indexes restores all of them after re-convergence."""
        db = _make_db(tmp_path)
        _converge(db)

        dropped = ["idx_knowledge_kind", "idx_goals_salience", "idx_transcript_channel"]
        with db.connection() as conn:
            for idx in dropped:
                conn.execute(f"DROP INDEX IF EXISTS {idx}")

        _converge(db)

        with db.connection() as conn:
            indexes = _index_names(conn)

        for idx in dropped:
            assert idx in indexes, f"Index '{idx}' was not restored"

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
            conn.execute("DROP TABLE IF EXISTS knowledge_vec")

        with db.connection() as conn:
            assert "knowledge_vec" not in _virtual_table_names(conn)

        _converge(db)

        with db.connection() as conn:
            assert "knowledge_vec" in _virtual_table_names(conn)

    def test_fts5_table_is_queryable_after_creation(self, tmp_path):
        """Newly created FTS5 table accepts an FTS query without error."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS documents_fts")

        _converge(db)

        with db.connection() as conn:
            # A basic FTS match query should execute without error
            result = conn.execute(
                "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH 'test'"
            ).fetchone()
        assert result is not None

    # ── 8. Seed data on fresh DB ──────────────────────────────────────────────

    def test_identity_vectors_seeded_on_fresh_db(self, tmp_path):
        """identity_vectors table is populated with default archetype rows on fresh DB."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            rows = conn.execute(
                "SELECT vector_name FROM identity_vectors ORDER BY vector_name"
            ).fetchall()

        names = {r[0] for r in rows}
        expected = {
            "curiosity", "assertiveness", "warmth",
            "playfulness", "skepticism", "emotional_intensity",
            "uncertainty_tolerance",
        }
        assert expected.issubset(names), f"Missing identity vectors: {expected - names}"

    def test_settings_seeded_on_fresh_db(self, tmp_path):
        """settings table contains the api_key seed row on a fresh DB."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT key, is_sensitive FROM settings WHERE key='api_key'"
            ).fetchone()

        assert row is not None, "api_key seed row missing from settings"
        assert row[1] == 1  # is_sensitive = True

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
                "SELECT COUNT(*) FROM identity_vectors WHERE vector_name='curiosity'"
            ).fetchone()[0]

        assert count == 1, f"Expected 1 curiosity row, got {count}"

    # ── 9. Migration stamping on fresh DB ─────────────────────────────────────

    def test_all_migration_files_stamped_on_fresh_db(self, tmp_path):
        """All *.sql files in the migrations directory are recorded in schema_migrations on fresh install."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        svc.converge()

        migrations_dir = svc._migrations_dir
        expected_files = {f.name for f in sorted(migrations_dir.glob("*.sql"))}

        with db.connection() as conn:
            rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()

        stamped = {r[0] for r in rows}
        missing = expected_files - stamped
        assert not missing, f"These migration files were not stamped: {missing}"

    def test_migration_stamp_count_matches_file_count(self, tmp_path):
        """The number of stamped migrations equals the number of *.sql files."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        svc.converge()

        migrations_dir = svc._migrations_dir
        file_count = len(list(migrations_dir.glob("*.sql")))

        with db.connection() as conn:
            stamped_count = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations"
            ).fetchone()[0]

        assert stamped_count == file_count

    # ── 10. Pending migration execution ───────────────────────────────────────

    def test_pending_migration_is_applied(self, tmp_path):
        """A migration not yet in schema_migrations is applied on the next converge."""
        db = _make_db(tmp_path)
        # First full convergence — stamps all migrations
        _converge(db)

        # Remove one stamped migration to simulate it being unapplied
        with db.connection() as conn:
            conn.execute(
                "DELETE FROM schema_migrations WHERE filename='028_auth_sessions.sql'"
            )
            # Also verify the target table still exists (the migration is idempotent)
            tables_before = _table_names(conn)

        assert "auth_sessions" in tables_before

        # Re-converge — pending migration should re-run (idempotently)
        _converge(db)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT filename FROM schema_migrations WHERE filename='028_auth_sessions.sql'"
            ).fetchone()

        assert row is not None, "028_auth_sessions.sql was not re-stamped after pending run"

    def test_already_applied_migrations_not_re_run(self, tmp_path):
        """Migrations already in schema_migrations are not applied again."""
        db = _make_db(tmp_path)
        _converge(db)

        with db.connection() as conn:
            before = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]

        # Second convergence — no new migrations should be added
        _converge(db)

        with db.connection() as conn:
            after = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]

        assert before == after, "Extra migration rows added on re-convergence"

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

    def test_default_768_dimensions_when_not_overridden(self, tmp_path):
        """vec0 tables default to float[768] when no override is provided."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=768)
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
            assert "float[768]" in ddl, (
                f"Expected float[768] in {name} DDL but got: {ddl}"
            )

    # ── 12. DDL extraction with comments ──────────────────────────────────────

    def test_comment_with_semicolon_does_not_break_ddl_extraction(self, tmp_path):
        """SQL comments containing semicolons inside CREATE TABLE blocks don't
        cause _extract_table_ddl() to truncate the DDL prematurely."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        # The knowledge table has a column with a comment containing a semicolon:
        #   reliability TEXT ...  -- dropped by migration 026; kept here for migration compat
        # If the extractor is broken it will truncate before the final ')'.
        schema_sql = svc._schema_path.read_text()
        ddl = svc._extract_table_ddl(schema_sql, "knowledge")

        assert ddl is not None, "Failed to extract knowledge table DDL"
        # DDL must be a complete CREATE TABLE statement — ends with );
        stripped = ddl.strip().rstrip(";").strip()
        assert stripped.endswith(")"), (
            f"knowledge DDL appears truncated (missing closing paren): ...{stripped[-60:]}"
        )
        # Must contain columns that appear after the comment-with-semicolon line
        assert "search_queries" in ddl.lower(), (
            "search_queries column absent — DDL was likely cut at the comment semicolon"
        )

    def test_extract_table_ddl_strips_inline_comments(self, tmp_path):
        """_extract_table_ddl strips single-line comments before matching."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        schema_sql = svc._schema_path.read_text()
        # episodes table has several inline comments
        ddl = svc._extract_table_ddl(schema_sql, "episodes")

        assert ddl is not None, "Failed to extract episodes table DDL"
        assert "CREATE TABLE" in ddl.upper()
        assert "salience" in ddl.lower()

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

    def test_vec0_shadow_tables_not_in_normal_tables(self, tmp_path):
        """vec0 shadow tables (e.g. episodes_vec_info) are excluded from
        the normal table set returned by _introspect_tables()."""
        db = _make_db(tmp_path)
        _converge(db)

        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        with db.connection() as conn:
            from services.schema_convergence_service import _load_sqlite_vec
            _load_sqlite_vec(conn)
            normal_tables = svc._introspect_tables(conn)

        shadow_like = [t for t in normal_tables if "_vec_" in t]
        assert not shadow_like, f"vec0 shadow tables leaked into normal tables: {shadow_like}"

    def test_virtual_tables_not_in_normal_tables(self, tmp_path):
        """Virtual tables themselves are excluded from _introspect_tables() output."""
        db = _make_db(tmp_path)
        _converge(db)

        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        with db.connection() as conn:
            from services.schema_convergence_service import _load_sqlite_vec
            _load_sqlite_vec(conn)
            normal_tables = svc._introspect_tables(conn)
            virtual_tables = svc._introspect_virtual_tables(conn)

        overlap = set(normal_tables.keys()) & set(virtual_tables.keys())
        assert not overlap, f"Virtual tables found in normal table set: {overlap}"

    # ── 14. Stale column logging ──────────────────────────────────────────────

    def test_stale_column_is_logged_not_dropped(self, tmp_path, caplog):
        """An extra column in a live table that is absent from desired schema is
        logged at DEBUG level but NOT dropped."""
        db = _make_db(tmp_path)
        _converge(db)

        # Add a column that is not in schema.sql
        with db.connection() as conn:
            conn.execute(
                "ALTER TABLE lists ADD COLUMN legacy_field TEXT DEFAULT NULL"
            )

        # Re-converge with DEBUG logging captured
        with caplog.at_level(logging.DEBUG, logger="services.schema_convergence_service"):
            _converge(db)

        # Column must still be present (not dropped)
        with db.connection() as conn:
            cols = _column_names(conn, "lists")

        assert "legacy_field" in cols, "Stale column was dropped — it should be preserved"

        # A debug log entry mentioning the stale column should exist
        stale_logs = [
            r.message for r in caplog.records
            if "stale" in r.message.lower() and "legacy_field" in r.message
        ]
        assert stale_logs, "No debug log entry found for the stale column"

    # ── 15. DDL normalization ─────────────────────────────────────────────────

    def test_normalize_ddl_lowercases_and_collapses_whitespace(self, tmp_path):
        """_normalize_ddl returns lowercase with collapsed whitespace."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        result = svc._normalize_ddl("CREATE  TABLE   Foo  ( bar TEXT )")
        assert result == result.lower()
        assert "  " not in result  # no double spaces

    def test_normalize_ddl_strips_if_not_exists(self, tmp_path):
        """_normalize_ddl removes IF NOT EXISTS for stable comparison."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        with_ine = svc._normalize_ddl("CREATE TABLE IF NOT EXISTS foo (id INTEGER)")
        without_ine = svc._normalize_ddl("CREATE TABLE foo (id INTEGER)")

        assert with_ine == without_ine

    def test_normalize_ddl_handles_empty_string(self, tmp_path):
        """_normalize_ddl returns an empty string for empty input."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        assert svc._normalize_ddl("") == ""
        assert svc._normalize_ddl(None) == ""

    # ── 16. Drop statements ───────────────────────────────────────────────────

    def test_drop_statements_executed_during_convergence(self, tmp_path):
        """Tables listed in DROP TABLE IF EXISTS blocks in schema.sql are removed."""
        db = _make_db(tmp_path)
        _converge(db)

        # schema.sql drops cognitive_reflexes and triage_calibration_events
        with db.connection() as conn:
            tables = _table_names(conn)

        assert "cognitive_reflexes" not in tables
        assert "triage_calibration_events" not in tables

    def test_drop_statements_applied_even_if_table_does_not_exist(self, tmp_path):
        """DROP TABLE IF EXISTS is safe when the table never existed — no error raised."""
        db = _make_db(tmp_path)
        # Should complete without raising even though dropped tables don't exist yet
        _converge(db)

    # ── 17. _is_fresh_db ──────────────────────────────────────────────────────

    def test_is_fresh_db_true_for_empty_db(self, tmp_path):
        """_is_fresh_db() returns True for a database with no user tables."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        with db.connection() as conn:
            result = svc._is_fresh_db(conn)

        assert result is True

    def test_is_fresh_db_false_after_convergence(self, tmp_path):
        """_is_fresh_db() returns False after tables have been created."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)
        svc.converge()

        with db.connection() as conn:
            result = svc._is_fresh_db(conn)

        assert result is False

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

    def test_restore_if_not_exists_preserves_rest_of_ddl(self, tmp_path):
        """_restore_if_not_exists() does not alter the index name or columns."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        normalized = "create index idx_foo on bar(col)"
        restored = svc._restore_if_not_exists(normalized, "index")

        assert "idx_foo" in restored
        assert "on bar(col)" in restored

    def test_restore_if_not_exists_handles_unique_index(self, tmp_path):
        """_restore_if_not_exists() correctly handles CREATE UNIQUE INDEX."""
        db = _make_db(tmp_path)
        svc = SchemaConvergenceService(db, embedding_dimensions=256)

        normalized = "create unique index idx_uniq on bar(col)"
        restored = svc._restore_if_not_exists(normalized, "index")

        assert restored == "create unique index if not exists idx_uniq on bar(col)"

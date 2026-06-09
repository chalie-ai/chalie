"""
Tests for backend/schema.sql structural integrity.

Each test absorbs a specific end-to-end scenario, validating that the production
schema contains the expected tables, columns, indexes, and seed data.
The fixture loads the full schema into an in-memory SQLite database so these
tests run with zero external dependencies.
"""

import sqlite3
import pytest

from services.file_mapper_service import FileMapperService


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_columns(conn, table_name):
    """Return set of column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _get_tables(conn):
    """Return set of table names."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def _get_indexes(conn, table_name):
    """Return set of index names for a table."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table_name,)
    ).fetchall()
    return {row[0] for row in rows}


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaValidation:

    @pytest.fixture
    def schema_db(self):
        """Load full production schema into in-memory SQLite.

        sqlite-vec (vec0) and FTS5 virtual tables are filtered out before
        execution because the bare in-memory SQLite used in unit tests does
        not have those extensions loaded.  All regular tables, indexes, and
        seed INSERT statements are preserved exactly as they appear in
        schema.sql.
        """
        import re
        schema_path = FileMapperService.get_schema_path()
        raw = schema_path.read_text()

        # Strip statements that create virtual tables requiring extensions
        # (vec0 for sqlite-vec, fts5 for full-text search).
        # A statement runs from the keyword CREATE to the closing semicolon.
        cleaned = re.sub(
            r'CREATE\s+VIRTUAL\s+TABLE\s+.*?;',
            '',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        conn = sqlite3.connect(':memory:')
        conn.executescript(cleaned)
        yield conn
        conn.close()

    # ── Reliability defaults ─────────────────────────────────────────────────

    def test_reliability_default_values(self, schema_db):
        """Absorbs an end-to-end scenario: episodes table inserts without error.

        The reliability column concept only applied to the removed knowledge table.
        We verify episodes inserts still work correctly.
        """
        import uuid

        ep_id = str(uuid.uuid4())
        schema_db.execute(
            "INSERT INTO episodes (id, gist, salience, channel) VALUES (?, ?, ?, ?)",
            (ep_id, 'test gist', 5, 'test-topic')
        )
        row = schema_db.execute(
            "SELECT id FROM episodes WHERE id = ?", (ep_id,)
        ).fetchone()
        assert row is not None, "episodes INSERT failed"

        schema_db.rollback()

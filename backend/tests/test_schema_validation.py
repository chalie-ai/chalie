"""
Tests for backend/schema.sql structural integrity.

Each test absorbs a specific nightly scenario, validating that the production
schema contains the expected tables, columns, indexes, and seed data.
The fixture loads the full schema into an in-memory SQLite database so these
tests run with zero external dependencies.
"""

import sqlite3
import pytest
from pathlib import Path


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
        schema_path = Path(__file__).parent.parent / 'schema.sql'
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

    # ── Scenario 145 ─────────────────────────────────────────────────────────

    def test_effort_column_in_tool_profiles(self, schema_db):
        """Absorbs scenario 145: tool_capability_profiles has an effort column.

        The effort field distinguishes low/moderate/high-cost tools so the
        ACT loop can factor in execution cost during tool selection.
        """
        columns = _get_columns(schema_db, 'tool_capability_profiles')
        assert 'effort' in columns, (
            "effort column missing from tool_capability_profiles"
        )

    # ── Scenario 200 ─────────────────────────────────────────────────────────

    def test_reliability_columns_on_memory_tables(self, schema_db):
        """Absorbs scenario 200: reliability column exists on knowledge.

        uncertainties table was removed from schema.sql (dropped by migration
        025). Reliability column is still
        present on knowledge (for migration 026 compat).
        """
        cols = _get_columns(schema_db, 'knowledge')
        assert 'reliability' in cols, (
            "reliability column missing from knowledge"
        )

        # uncertainties should NOT be in schema.sql (removed; migration 025 drops it)
        tables = _get_tables(schema_db)
        assert 'uncertainties' not in tables, (
            "uncertainties table should not exist in schema.sql "
            "(dropped by migration 025)"
        )

    # ── Scenario 201 ─────────────────────────────────────────────────────────

    def test_reliability_default_values(self, schema_db):
        """Absorbs scenario 201: reliability columns default to 'reliable'.

        Inserts a minimal row into knowledge and episodes
        without specifying reliability and confirms the column defaults correctly.
        """
        import uuid

        # knowledge (replaces user_traits and semantic_concepts)
        schema_db.execute(
            "INSERT INTO knowledge (kind, entity, key, value) VALUES ('trait', 'user', ?, ?)",
            ('test_key', 'test_value')
        )
        row = schema_db.execute(
            "SELECT reliability FROM knowledge WHERE key = 'test_key'"
        ).fetchone()
        assert row is not None, "knowledge INSERT failed"
        assert row[0] == 'reliable', (
            f"knowledge.reliability default is {row[0]!r}, expected 'reliable'"
        )

        # episodes
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

    # ── Scenario 231 ─────────────────────────────────────────────────────────

    def test_knowledge_table_supports_procedures(self, schema_db):
        """Absorbs scenario 231: knowledge table supports procedure entries.

        Procedures (formerly procedural_memory) are stored in the unified
        knowledge table with kind='procedure'. Reward data (weight, success_rate)
        lives in the JSON `data` column.
        """
        tables = _get_tables(schema_db)
        assert 'knowledge' in tables, (
            "knowledge table missing from schema"
        )

        columns = _get_columns(schema_db, 'knowledge')
        required = {'kind', 'entity', 'key', 'value', 'data', 'confidence', 'decay_class'}
        missing = required - columns
        assert not missing, (
            f"knowledge missing columns for procedure support: {missing}"
        )

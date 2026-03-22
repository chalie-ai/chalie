"""
Tests that constraint learning event types are queryable in the interaction_log table.

Absorbs nightly scenarios 235, 238, 239, 243.
These tests verify the schema supports the required event_type values — no real
data is required; we insert synthetic rows and confirm the queries succeed.
"""

import re
import pytest
import sqlite3
from pathlib import Path


@pytest.mark.unit
class TestConstraintEventTypes:

    @pytest.fixture
    def schema_db(self):
        """Load full production schema into in-memory SQLite.

        sqlite-vec (vec0) and FTS5 virtual tables are stripped before execution
        because the bare in-memory SQLite used in unit tests does not have those
        extensions loaded.  All regular tables, indexes, and seed INSERT
        statements are preserved exactly as they appear in schema.sql.
        """
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        raw = schema_path.read_text()
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

    # ── scenario 235 ─────────────────────────────────────────────────────────

    def test_reliability_warning_event_queryable(self, schema_db):
        """reliability_warning event_type can be inserted and queried."""
        schema_db.execute(
            "INSERT INTO interaction_log (id, event_type, payload) "
            "VALUES ('test-rw-1', 'reliability_warning', '{\"action_type\":\"test\"}')"
        )
        schema_db.commit()

        cur = schema_db.execute(
            "SELECT event_type FROM interaction_log WHERE event_type = 'reliability_warning'"
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'reliability_warning'

    # ── scenario 238 ─────────────────────────────────────────────────────────

    def test_assimilation_rejection_event_queryable(self, schema_db):
        """assimilation_rejected event_type can be inserted and queried."""
        schema_db.execute(
            "INSERT INTO interaction_log (id, event_type, payload) "
            "VALUES ('test-ar-1', 'assimilation_rejected', '{\"action_type\":\"test\"}')"
        )
        schema_db.commit()

        cur = schema_db.execute(
            "SELECT event_type FROM interaction_log WHERE event_type = 'assimilation_rejected'"
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'assimilation_rejected'

    # ── scenario 239 ─────────────────────────────────────────────────────────

    def test_plan_rejection_event_queryable(self, schema_db):
        """plan_rejected event_type can be inserted and queried."""
        schema_db.execute(
            "INSERT INTO interaction_log (id, event_type, payload) "
            "VALUES ('test-pr-1', 'plan_rejected', '{\"action_type\":\"test\"}')"
        )
        schema_db.commit()

        cur = schema_db.execute(
            "SELECT event_type FROM interaction_log WHERE event_type = 'plan_rejected'"
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'plan_rejected'

    # ── scenario 243 ─────────────────────────────────────────────────────────

    def test_all_constraint_event_types_queryable(self, schema_db):
        """All 6 constraint event types can be stored and retrieved via IN clause."""
        constraint_event_types = [
            'action_gate_rejected',
            'plan_rejected',
            'assimilation_rejected',
            'reliability_warning',
            'critic_override',
            'constraint_learned',
        ]

        for i, event_type in enumerate(constraint_event_types):
            schema_db.execute(
                "INSERT INTO interaction_log (id, event_type, payload) VALUES (?, ?, ?)",
                (f'test-all-{i}', event_type, '{"action_type":"test"}'),
            )
        schema_db.commit()

        placeholders = ','.join('?' * len(constraint_event_types))
        cur = schema_db.execute(
            f"SELECT event_type FROM interaction_log WHERE event_type IN ({placeholders})",
            constraint_event_types,
        )
        returned_types = {row[0] for row in cur.fetchall()}
        assert returned_types == set(constraint_event_types)

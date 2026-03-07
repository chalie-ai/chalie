"""Unit tests for CortexIterationService — id column in INSERT regression."""

import json
import sqlite3
import pytest
from unittest.mock import MagicMock
from contextlib import contextmanager

from services.cortex_iteration_service import CortexIterationService


pytestmark = pytest.mark.unit


def _make_db(schema_sql: str) -> tuple:
    """Create an in-memory SQLite db with the given schema, return (db_service, conn)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(schema_sql)
    conn.commit()

    @contextmanager
    def _connection():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    svc = MagicMock()
    svc.connection = _connection
    return svc, conn


SCHEMA = """
CREATE TABLE cortex_iterations (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    exchange_id TEXT,
    session_id TEXT,
    loop_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    execution_time_ms REAL,
    chosen_mode TEXT,
    chosen_confidence REAL,
    alternative_paths TEXT,
    iteration_cost REAL,
    diminishing_cost REAL,
    uncertainty_cost REAL,
    action_base_cost REAL,
    total_cost REAL,
    cumulative_cost REAL,
    efficiency_score REAL,
    expected_confidence_gain REAL,
    task_value REAL,
    future_leverage REAL,
    effort_estimate TEXT,
    effort_multiplier REAL,
    iteration_penalty REAL,
    exploration_bonus REAL,
    net_value REAL,
    decision_override INTEGER,
    overridden_mode TEXT,
    termination_reason TEXT,
    actions_executed TEXT,
    action_count INTEGER,
    action_success_count INTEGER,
    frontal_cortex_response TEXT,
    config_snapshot TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def _make_iteration(n: int = 1) -> dict:
    """Minimal iteration dict with required fields."""
    from services.time_utils import utc_now
    now = utc_now().isoformat()
    return {
        'iteration_number': n,
        'started_at': now,
        'completed_at': now,
        'execution_time_ms': 250.0,
        'chosen_mode': 'RESPOND',
    }


class TestCortexIterationServiceInsert:

    def test_log_iterations_batch_creates_rows_with_id(self):
        """Each logged iteration must have a non-null UUID id in the database."""
        db_svc, conn = _make_db(SCHEMA)
        svc = CortexIterationService(db_svc)

        svc.log_iterations_batch(
            loop_id='loop-abc',
            topic='test-topic',
            exchange_id='ex-001',
            session_id='sess-001',
            iterations=[_make_iteration(1), _make_iteration(2)],
        )

        rows = conn.execute("SELECT id, loop_id, iteration_number FROM cortex_iterations").fetchall()
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

        for row in rows:
            assert row['id'] is not None, "id must not be NULL"
            assert len(row['id']) == 36, f"id should be a UUID (36 chars), got {row['id']!r}"
            assert row['loop_id'] == 'loop-abc'

        ids = [r['id'] for r in rows]
        assert ids[0] != ids[1], "Each iteration must get a unique id"

    def test_log_iterations_batch_empty_does_not_crash(self):
        """Empty iterations list logs nothing and does not raise."""
        db_svc, conn = _make_db(SCHEMA)
        svc = CortexIterationService(db_svc)

        svc.log_iterations_batch(
            loop_id='loop-xyz',
            topic='topic',
            exchange_id='ex-002',
            session_id='sess-002',
            iterations=[],
        )

        rows = conn.execute("SELECT COUNT(*) FROM cortex_iterations").fetchone()
        assert rows[0] == 0

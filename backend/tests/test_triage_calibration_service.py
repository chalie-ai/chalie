"""Unit tests for TriageCalibrationService — log_triage_decision INSERT regression."""

import json
import sqlite3
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock

from services.triage_calibration_service import TriageCalibrationService
from services.cognitive_triage_service import TriageResult


pytestmark = pytest.mark.unit


SCHEMA = """
CREATE TABLE triage_calibration_events (
    id TEXT PRIMARY KEY,
    exchange_id TEXT,
    topic TEXT,
    triage_branch TEXT NOT NULL,
    triage_mode TEXT NOT NULL,
    tool_selected TEXT,
    confidence_internal REAL,
    confidence_tool_need REAL,
    reasoning TEXT,
    freshness_risk REAL,
    decision_entropy REAL,
    self_eval_override INTEGER DEFAULT 0,
    self_eval_reason TEXT,
    outcome_mode TEXT,
    outcome_tools_used TEXT,
    outcome_tool_success INTEGER,
    outcome_latency_ms REAL,
    tool_abstention INTEGER DEFAULT 0,
    signal_rephrase INTEGER DEFAULT 0,
    signal_correction INTEGER DEFAULT 0,
    signal_explicit_lookup INTEGER DEFAULT 0,
    signal_abandonment INTEGER DEFAULT 0,
    correctness_label TEXT,
    correctness_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()

    @contextmanager
    def _connection():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    db = MagicMock()
    db.execute = lambda sql, params=(): conn.execute(sql, params)
    db.connection = _connection
    return db, conn


def _make_triage_result():
    return TriageResult(
        branch='respond', mode='RESPOND', tools=[], skills=[],
        confidence_internal=0.8, confidence_tool_need=0.2,
        freshness_risk=0.1, decision_entropy=0.6,
        reasoning='test', triage_time_ms=12.0,
        fast_filtered=False, self_eval_override=False, self_eval_reason='',
    )


class TestLogTriageDecision:

    def test_log_triage_decision_creates_row_with_id(self):
        """log_triage_decision must persist a row with a non-null UUID id."""
        db, conn = _make_db()
        svc = TriageCalibrationService(db)

        svc.log_triage_decision('ex-001', 'test-topic', _make_triage_result())

        rows = conn.execute("SELECT id, exchange_id, triage_branch FROM triage_calibration_events").fetchall()
        assert len(rows) == 1
        assert rows[0]['id'] is not None
        assert len(rows[0]['id']) == 36  # UUID
        assert rows[0]['exchange_id'] == 'ex-001'
        assert rows[0]['triage_branch'] == 'respond'

    def test_log_triage_decision_multiple_events_unique_ids(self):
        """Each call must produce a distinct id — not conflict on PRIMARY KEY."""
        db, conn = _make_db()
        svc = TriageCalibrationService(db)

        svc.log_triage_decision('ex-001', 'topic', _make_triage_result())
        svc.log_triage_decision('ex-002', 'topic', _make_triage_result())
        svc.log_triage_decision('ex-003', 'topic', _make_triage_result())

        rows = conn.execute("SELECT id FROM triage_calibration_events").fetchall()
        assert len(rows) == 3
        ids = {r['id'] for r in rows}
        assert len(ids) == 3  # all distinct

    def test_log_triage_decision_empty_exchange_id_does_not_crash(self):
        """Empty exchange_id (stored as '') must not prevent insert."""
        db, conn = _make_db()
        svc = TriageCalibrationService(db)

        svc.log_triage_decision(None, 'topic', _make_triage_result())
        svc.log_triage_decision('', 'topic', _make_triage_result())

        rows = conn.execute("SELECT COUNT(*) FROM triage_calibration_events").fetchone()
        assert rows[0] == 2

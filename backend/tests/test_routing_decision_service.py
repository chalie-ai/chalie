"""
Unit tests for RoutingDecisionService — routing_time_ms persistence.
"""

import json
import sqlite3
import pytest

from services.routing_decision_service import RoutingDecisionService
from services.database_service import DatabaseService


@pytest.fixture
def db_service(tmp_path):
    """In-memory SQLite with the routing_decisions table."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE routing_decisions (
            id TEXT PRIMARY KEY,
            topic TEXT,
            exchange_id TEXT,
            selected_mode TEXT,
            router_confidence REAL,
            scores TEXT,
            tiebreaker_used INTEGER,
            tiebreaker_candidates TEXT,
            margin REAL,
            effective_margin REAL,
            signal_snapshot TEXT,
            weight_snapshot TEXT,
            routing_time_ms REAL,
            feedback TEXT,
            reflection TEXT,
            reasoning TEXT,
            previous_mode TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

    svc = DatabaseService.__new__(DatabaseService)
    svc.db_path = db_path
    from contextlib import contextmanager

    @contextmanager
    def _conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    svc.connection = _conn
    return svc


@pytest.mark.unit
class TestRoutingDecisionServiceTimePersistence:

    def test_routing_time_ms_is_persisted(self, db_service):
        """routing_time_ms from the routing result must be written to the DB (not NULL)."""
        svc = RoutingDecisionService(db_service)
        routing_result = {
            'mode': 'RESPOND',
            'router_confidence': 0.85,
            'scores': {'RESPOND': 0.85, 'ACT': 0.15},
            'tiebreaker_used': False,
            'tiebreaker_candidates': None,
            'margin': 0.70,
            'effective_margin': 0.70,
            'signal_snapshot': {},
            'weight_snapshot': {},
            'routing_time_ms': 123.4,
        }

        decision_id = svc.log_decision(
            topic='test-topic',
            exchange_id='exchange-001',
            routing_result=routing_result,
        )

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT routing_time_ms FROM routing_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(123.4), (
            f"routing_time_ms should be 123.4 but got {row[0]!r} — "
            "triage path was not passing routing_time_ms into routing result dicts"
        )

    def test_routing_time_ms_triage_source(self, db_service):
        """Triage-sourced pre_routing_result (no weight_snapshot) still persists routing_time_ms."""
        svc = RoutingDecisionService(db_service)
        # Mirrors the dict built inline in digest_worker for the triage fast-path
        routing_result = {
            'mode': 'ACT',
            'router_confidence': 0.9,
            'routing_source': 'triage',
            'routing_time_ms': 48.7,
        }

        decision_id = svc.log_decision(
            topic='test-topic',
            exchange_id='exchange-002',
            routing_result=routing_result,
        )

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT routing_time_ms FROM routing_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(48.7)

    def test_routing_time_ms_null_when_missing(self, db_service):
        """If routing_time_ms is absent from the result dict, column stays NULL (no crash)."""
        svc = RoutingDecisionService(db_service)
        routing_result = {
            'mode': 'RESPOND',
            'router_confidence': 0.7,
            # routing_time_ms intentionally omitted
        }

        decision_id = svc.log_decision(
            topic='test-topic',
            exchange_id='exchange-003',
            routing_result=routing_result,
        )

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT routing_time_ms FROM routing_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()

        assert row is not None
        assert row[0] is None  # graceful — no crash, just NULL

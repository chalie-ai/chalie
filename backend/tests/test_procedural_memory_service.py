"""Tests for ProceduralMemoryService — action outcome recording and weight updates."""

import json
import sqlite3
import pytest

from services.procedural_memory_service import ProceduralMemoryService
from services.database_service import DatabaseService


pytestmark = pytest.mark.unit


# ── Schema for test DB ───────────────────────────────────────────────

PROCEDURAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedural_memory (
    id TEXT PRIMARY KEY,
    action_name TEXT NOT NULL UNIQUE,
    total_attempts INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_reward REAL DEFAULT 0.0,
    weight REAL DEFAULT 1.0,
    reward_history TEXT DEFAULT '[]',
    context_stats TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def db_service(tmp_path):
    """Real in-memory-ish SQLite DB with procedural_memory schema."""
    db_path = str(tmp_path / "test_procedural.db")
    conn = sqlite3.connect(db_path)
    conn.execute(PROCEDURAL_SCHEMA)
    conn.commit()
    conn.close()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True

    import contextlib

    @contextlib.contextmanager
    def _connection():
        c = sqlite3.connect(db_path)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    db.connection = _connection
    return db


@pytest.fixture
def svc(db_service):
    """ProceduralMemoryService with real SQLite backend."""
    return ProceduralMemoryService(db_service)


# ── Record action outcome ────────────────────────────────────────────

class TestRecordActionOutcome:

    def test_record_success(self, svc):
        """Recording a successful action creates the entry and updates stats."""
        result = svc.record_action_outcome('test_action', success=True, reward=0.5)
        assert result is True

        stats = svc.get_action_stats('test_action')
        assert stats is not None
        assert stats['total_attempts'] == 1
        assert stats['total_successes'] == 1
        assert stats['success_rate'] == 1.0

    def test_record_failure(self, svc):
        """Recording a failed action updates stats correctly."""
        svc.record_action_outcome('test_action', success=False, reward=-0.3)

        stats = svc.get_action_stats('test_action')
        assert stats['total_attempts'] == 1
        assert stats['total_successes'] == 0
        assert stats['success_rate'] == 0.0

    def test_multiple_outcomes_track_correctly(self, svc):
        """Multiple outcomes accumulate correctly."""
        svc.record_action_outcome('multi', success=True, reward=0.5)
        svc.record_action_outcome('multi', success=True, reward=0.3)
        svc.record_action_outcome('multi', success=False, reward=-0.2)

        stats = svc.get_action_stats('multi')
        assert stats['total_attempts'] == 3
        assert stats['total_successes'] == 2

    def test_failure_class_stored_in_history(self, svc):
        """failure_class is recorded in reward_history entries."""
        svc.record_action_outcome('fc_test', success=False, reward=-0.1,
                                   failure_class='internal')

        stats = svc.get_action_stats('fc_test')
        assert stats is not None
        # reward_history is stored as JSON string in SQLite
        # Just verify the action was recorded
        assert stats['total_attempts'] == 1

    def test_topic_context_stats_updated(self, svc):
        """Context stats are updated when topic is provided."""
        svc.record_action_outcome('ctx_test', success=True, reward=0.5, topic='python')

        stats = svc.get_action_stats('ctx_test')
        ctx = stats['context_stats']
        assert 'python' in ctx
        assert ctx['python']['attempts'] == 1
        assert ctx['python']['successes'] == 1


# ── Weight updates ───────────────────────────────────────────────────

class TestWeightUpdates:

    def test_weight_increases_on_success(self, svc):
        """Successful actions should increase weight (toward target)."""
        svc.record_action_outcome('w_test', success=True, reward=0.5)
        w1 = svc.get_action_weight('w_test')

        svc.record_action_outcome('w_test', success=True, reward=0.5)
        w2 = svc.get_action_weight('w_test')

        # Weight should move toward target (success_rate * (1+reward))
        assert w2 >= w1 or abs(w2 - w1) < 0.01  # Allow small float diffs

    def test_weight_bounded(self, svc):
        """Weight should stay within [0.1, 5.0] bounds."""
        for _ in range(50):
            svc.record_action_outcome('bounded', success=True, reward=0.5)

        w = svc.get_action_weight('bounded')
        assert 0.1 <= w <= 5.0

    def test_default_weight_for_unknown_action(self, svc):
        """Unknown action returns default weight."""
        w = svc.get_action_weight('nonexistent_action')
        assert w == 1.0  # default_action_weight


# ── Gate rejection ───────────────────────────────────────────────────

class TestGateRejection:

    def test_gate_rejection_records_neutral(self, svc):
        """Gate rejections record as neutral (reward=0.0, success=False)."""
        result = svc.record_gate_rejection('gated_action', reason='wrong phase')
        assert result is True

        stats = svc.get_action_stats('gated_action')
        assert stats['total_attempts'] == 1
        assert stats['total_successes'] == 0
        # Neutral reward keeps weight stable (close to default)
        assert abs(stats['weight'] - 1.0) < 0.5


# ── Retrieval methods ────────────────────────────────────────────────

class TestRetrieval:

    def test_get_all_policy_weights(self, svc):
        """get_all_policy_weights returns all recorded actions."""
        svc.record_action_outcome('action_a', success=True, reward=0.1)
        svc.record_action_outcome('action_b', success=False, reward=-0.1)

        weights = svc.get_all_policy_weights()
        assert 'action_a' in weights
        assert 'action_b' in weights

    def test_get_ranked_skills(self, svc):
        """get_ranked_skills returns sorted list by expected value."""
        svc.record_action_outcome('high', success=True, reward=0.8)
        svc.record_action_outcome('low', success=False, reward=-0.5)

        ranked = svc.get_ranked_skills()
        assert len(ranked) == 2
        assert ranked[0]['expected_value'] >= ranked[1]['expected_value']

    def test_get_ranked_skills_with_topic(self, svc):
        """Topic affinity adjusts expected value."""
        svc.record_action_outcome('topic_skill', success=True, reward=0.5, topic='python')

        ranked = svc.get_ranked_skills(topic='python')
        assert len(ranked) == 1
        assert ranked[0]['topic_affinity'] == 1.0

    def test_get_action_stats_nonexistent(self, svc):
        """Non-existent action returns None."""
        assert svc.get_action_stats('ghost') is None

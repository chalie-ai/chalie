"""Tests for P1-a: Goal Evidence Pipeline Dead + Deduplication Broken."""
import json
import sqlite3
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from services.goal_ecology_service import GoalEcologyService
from services.goal_signal_service import (
    extract_and_route_signals, route_cognitive_signal,
    _try_graduate_candidate, CANDIDATE_MATCH_THRESHOLD,
    CANDIDATE_GRADUATION_EVIDENCE, GOAL_MATCH_THRESHOLD,
)
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'emergent',
    status TEXT NOT NULL DEFAULT 'candidate',
    description TEXT NOT NULL,
    parent_motives TEXT DEFAULT '[]',
    identity_links TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.1,
    salience REAL DEFAULT 0.0,
    commitment REAL DEFAULT 0.0,
    urgency REAL DEFAULT 0.0,
    timescale TEXT DEFAULT 'medium_term',
    strategy TEXT,
    strategy_hash INTEGER,
    lineage_parent_id TEXT,
    evidence_count INTEGER DEFAULT 0,
    is_muted INTEGER DEFAULT 0,
    last_reinforced_at TEXT,
    last_acted_at TEXT,
    outcome_feedback TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goal_evidence (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goals(id),
    signal_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    strength REAL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
"""


def _make_db():
    """Create an in-memory SQLite db with goal schema, return db_service mock."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(GOAL_SCHEMA)
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



class TestGoalDeduplication:
    def test_duplicate_emergent_goal_reinforces_instead_of_creating(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        g1 = eco.create_goal(description="Interface as capability surface", type="emergent")
        orig = g1["salience"]
        with patch.object(eco, "_find_similar_goal", return_value={
            "id": g1["id"], "description": g1["description"],
            "type": g1["type"], "status": g1["status"],
            "salience": g1["salience"], "confidence": g1["confidence"],
            "similarity": 0.92,
        }):
            g2 = eco.create_goal(description="Interface capability surface layer", type="emergent")
        assert g2["id"] == g1["id"]
        assert g2["salience"] == min(1.0, orig + 0.1)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM goals")
        assert cursor.fetchone()[0] == 1
        cursor.close()

    def test_dedup_caps_salience_at_1(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        g1 = eco.create_goal(description="High salience goal", type="stated")
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET salience = 0.95 WHERE id = ?", (g1["id"],))
        conn.commit()
        cursor.close()
        with patch.object(eco, "_find_similar_goal", return_value={
            "id": g1["id"], "description": g1["description"],
            "type": "stated", "status": "actionable",
            "salience": 0.95, "confidence": 0.8, "similarity": 0.90,
        }):
            g2 = eco.create_goal(description="High salience goal again", type="stated")
        assert g2["salience"] == 1.0

    def test_no_dedup_when_find_similar_returns_none(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        with patch.object(eco, "_find_similar_goal", return_value=None):
            g = eco.create_goal(description="Unique goal", type="emergent")
        assert g["status"] == "candidate"
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM goals")
        assert cursor.fetchone()[0] == 1
        cursor.close()


class TestCandidateGoalMatching:
    def test_signals_match_candidate_goals_with_lower_threshold(self):
        mock_eco = MagicMock()
        mock_eco.find_matching_goals.return_value = [{
            "id": "cg1", "description": "Emerging interest in cooking",
            "type": "emergent", "status": "candidate",
            "salience": 0.1, "confidence": 0.1, "similarity": 0.58,
        }]
        mock_eco.get_goal.return_value = {"id": "cg1", "status": "candidate", "evidence_count": 1}
        with patch("services.goal_ecology_service.GoalEcologyService", return_value=mock_eco):
            extract_and_route_signals("cooking", "I want to learn more about cooking techniques", {"intent_type": "statement"})
        mock_eco.find_matching_goals.assert_called_once_with("I want to learn more about cooking techniques", threshold=CANDIDATE_MATCH_THRESHOLD)
        mock_eco.add_evidence.assert_called_once()

    def test_non_candidate_goals_still_need_strict_threshold(self):
        mock_eco = MagicMock()
        mock_eco.find_matching_goals.return_value = [{
            "id": "ag1", "description": "Learn to cook", "type": "emergent",
            "status": "active", "salience": 0.5, "confidence": 0.6, "similarity": 0.58,
        }]
        with patch("services.goal_ecology_service.GoalEcologyService", return_value=mock_eco):
            extract_and_route_signals("cooking", "Something about cooking and recipes and techniques", {"intent_type": "statement"})
        mock_eco.add_evidence.assert_not_called()


class TestCandidateGoalGraduation:
    def test_candidate_graduates_with_3_evidence(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        g = eco.create_goal(description="Cooking interest", type="emergent")
        assert g["status"] == "candidate"
        for i in range(3):
            eco.add_evidence(goal_id=g["id"], signal_type="topic_recurrence", content="Evidence %d" % i, strength=0.7)
        _try_graduate_candidate(eco, g["id"])
        updated = eco.get_goal(g["id"])
        assert updated["status"] == "active"
        assert updated["confidence"] >= 0.5

    def test_candidate_does_not_graduate_with_fewer_than_3(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        g = eco.create_goal(description="Cooking interest", type="emergent")
        for i in range(2):
            eco.add_evidence(goal_id=g["id"], signal_type="topic_recurrence", content="Evidence %d" % i, strength=0.7)
        _try_graduate_candidate(eco, g["id"])
        updated = eco.get_goal(g["id"])
        assert updated["status"] == "candidate"

    def test_graduation_only_applies_to_candidate_status(self):
        db, conn = _make_db()
        eco = GoalEcologyService(db_service=db)
        g = eco.create_goal(description="Stated cooking goal", type="stated")
        assert g["status"] == "actionable"
        for i in range(3):
            eco.add_evidence(goal_id=g["id"], signal_type="topic_recurrence", content="Evidence %d" % i, strength=0.7)
        _try_graduate_candidate(eco, g["id"])
        updated = eco.get_goal(g["id"])
        assert updated["status"] == "actionable"

    def test_cognitive_signal_routes_to_candidate_goals(self):
        mock_eco = MagicMock()
        mock_eco.find_matching_goals.return_value = [{
            "id": "c1", "description": "Interest in data patterns", "type": "emergent",
            "status": "candidate", "salience": 0.1, "confidence": 0.1, "similarity": 0.60,
        }]
        mock_eco.get_goal.return_value = {"id": "c1", "status": "candidate", "evidence_count": 1}
        with patch("services.goal_ecology_service.GoalEcologyService", return_value=mock_eco):
            route_cognitive_signal({"signal_type": "new_knowledge", "payload": {"content": "Discovered patterns in user data analysis"}, "source_id": "semantic_consolidation"})
        mock_eco.add_evidence.assert_called_once()
        mock_eco.find_matching_goals.assert_called_once_with("Discovered patterns in user data analysis", threshold=CANDIDATE_MATCH_THRESHOLD)

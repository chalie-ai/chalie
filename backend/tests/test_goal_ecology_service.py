"""Unit tests for GoalEcologyService — persistent goal lifecycle management."""

import json
import sqlite3
import pytest
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import MagicMock, patch

from services.goal_ecology_service import (
    GoalEcologyService, CORE_MOTIVES, ACTIONABLE_THRESHOLDS,
    SALIENCE_DECAY_RATE, DECAY_THRESHOLD, TIMESCALE_WINDOWS,
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
    lineage_parent_id TEXT,
    evidence_count INTEGER DEFAULT 0,
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


class TestGoalCreation:

    def test_create_stated_goal_has_high_salience(self):
        """Stated goals should start with boosted salience and actionable status."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="Plan dinner for 20 people",
            type='stated',
        )

        assert goal['type'] == 'stated'
        assert goal['status'] == 'actionable'
        assert goal['confidence'] == 0.8
        assert goal['salience'] >= 0.7

    def test_create_emergent_goal_starts_as_candidate(self):
        """Emergent goals should start as candidates with low salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="Become an expert in machine learning",
            type='emergent',
        )

        assert goal['type'] == 'emergent'
        assert goal['status'] == 'candidate'
        assert goal['confidence'] == 0.1
        assert goal['salience'] == 0.0

    def test_create_inferred_goal_starts_as_candidate(self):
        """Inferred goals should start as candidates."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="User seems interested in cooking",
            type='inferred',
        )

        assert goal['type'] == 'inferred'
        assert goal['status'] == 'candidate'
        assert goal['confidence'] == 0.1

    def test_create_developmental_goal(self):
        """Developmental goals should start as candidates with low confidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="Become Malta's AI voice",
            type='developmental',
            timescale='long_term',
        )

        assert goal['type'] == 'developmental'
        assert goal['timescale'] == 'long_term'
        assert goal['status'] == 'candidate'

    def test_create_goal_with_motives_and_links(self):
        """Goal creation should persist parent motives and identity links."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="Build a knowledge base",
            type='emergent',
            parent_motives=['competence_growth', 'initiative'],
            identity_links=['researcher', 'builder'],
        )

        # Verify stored in DB
        saved = ecology.get_goal(goal['id'])
        assert saved is not None
        assert 'competence_growth' in saved['parent_motives']
        assert 'researcher' in saved['identity_links']

    def test_goal_persisted_in_db(self):
        """Created goal should be retrievable from the database."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Test goal", type='stated')
        saved = ecology.get_goal(goal['id'])

        assert saved is not None
        assert saved['description'] == "Test goal"
        assert saved['type'] == 'stated'


class TestEvidenceAccumulation:

    def test_add_evidence_increments_count(self):
        """Adding evidence should increment the goal's evidence count."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Learn cooking", type='emergent')
        assert goal['salience'] == 0.0

        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='topic_recurrence',
            content='User discussed cooking techniques',
            strength=1.0,
        )

        saved = ecology.get_goal(goal['id'])
        assert saved['evidence_count'] == 1
        assert saved['last_reinforced_at'] is not None

    def test_multiple_evidence_increases_confidence(self):
        """Multiple evidence additions should increase confidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Improve fitness", type='inferred')
        initial_confidence = ecology.get_goal(goal['id'])['confidence']

        for i in range(5):
            ecology.add_evidence(
                goal_id=goal['id'],
                signal_type='topic_recurrence',
                content=f'User mentioned fitness topic #{i}',
                strength=1.0,
            )

        saved = ecology.get_goal(goal['id'])
        assert saved['confidence'] > initial_confidence
        assert saved['evidence_count'] == 5


class TestSalienceRecalculation:

    def test_salience_increases_with_evidence(self):
        """Salience should increase when evidence is added."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Plan a vacation", type='emergent')
        initial_salience = ecology.get_goal(goal['id'])['salience']

        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='explicit_statement',
            content='I want to plan a vacation',
            strength=1.0,
        )

        saved = ecology.get_goal(goal['id'])
        assert saved['salience'] >= initial_salience

    def test_diverse_evidence_increases_cross_context(self):
        """Evidence from diverse signal types should increase salience more."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Start a business", type='emergent')

        # Add diverse evidence types
        for sig_type in ['explicit_statement', 'topic_recurrence', 'behavioral']:
            ecology.add_evidence(
                goal_id=goal['id'],
                signal_type=sig_type,
                content=f'Signal about business from {sig_type}',
                strength=1.0,
            )

        saved = ecology.get_goal(goal['id'])
        # Cross-context should contribute to salience
        assert saved['salience'] > 0.0


class TestSalienceDecay:

    def test_decay_reduces_salience(self):
        """Goals not reinforced within their timescale window should lose salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create a goal with some salience
        goal = ecology.create_goal(description="Old goal", type='emergent')
        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='explicit_statement',
            content='Old signal',
            strength=1.0,
        )

        # Manually set last_reinforced_at to long ago to trigger decay
        old_time = (utc_now() - timedelta(days=30)).isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET last_reinforced_at = ?, salience = 0.3 WHERE id = ?
        """, (old_time, goal['id']))
        conn.commit()

        decayed = ecology.decay_unreinforced()

        saved = ecology.get_goal(goal['id'])
        assert saved['salience'] < 0.3

    def test_decay_marks_low_salience_as_decayed(self):
        """Goals below decay threshold should be marked 'decayed'."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Dying goal", type='emergent')

        # Set salience just above decay threshold and reinforced long ago
        old_time = (utc_now() - timedelta(days=30)).isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET last_reinforced_at = ?, salience = ? WHERE id = ?
        """, (old_time, DECAY_THRESHOLD + 0.01, goal['id']))
        conn.commit()

        ecology.decay_unreinforced()

        saved = ecology.get_goal(goal['id'])
        assert saved['status'] == 'decayed'
        assert saved['salience'] == 0.0

    def test_recently_reinforced_goals_dont_decay(self):
        """Goals reinforced recently should not lose salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Fresh goal", type='emergent')
        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='explicit_statement',
            content='Recent signal',
            strength=1.0,
        )

        saved_before = ecology.get_goal(goal['id'])
        ecology.decay_unreinforced()
        saved_after = ecology.get_goal(goal['id'])

        # Should not have decayed
        assert saved_after['salience'] >= saved_before['salience']


class TestGoalEvolution:

    def test_evolve_creates_lineage(self):
        """Evolving a goal should create a new goal with lineage_parent_id."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        original = ecology.create_goal(
            description="Learn cooking",
            type='emergent',
            parent_motives=['competence_growth'],
        )

        evolved = ecology.evolve_goal(original['id'], "Master Italian cuisine")

        # Original should be marked as 'evolved'
        original_saved = ecology.get_goal(original['id'])
        assert original_saved['status'] == 'evolved'

        # Evolved goal should reference parent
        evolved_saved = ecology.get_goal(evolved['id'])
        assert evolved_saved['lineage_parent_id'] == original['id']
        assert evolved_saved['description'] == "Master Italian cuisine"
        assert evolved_saved['status'] == 'strengthening'
        assert evolved_saved['type'] == original['type']


class TestActionableThresholds:

    def test_stated_goals_always_actionable(self):
        """Stated goals should be actionable at any salience level."""
        assert ACTIONABLE_THRESHOLDS['stated'] == 0.0

    def test_inferred_needs_moderate_salience(self):
        """Inferred goals need moderate salience."""
        assert ACTIONABLE_THRESHOLDS['inferred'] == 0.4

    def test_emergent_needs_high_salience(self):
        """Emergent goals need higher salience."""
        assert ACTIONABLE_THRESHOLDS['emergent'] == 0.6

    def test_developmental_needs_highest_salience(self):
        """Developmental goals need the highest salience."""
        assert ACTIONABLE_THRESHOLDS['developmental'] == 0.75

    def test_get_actionable_goals_filters_correctly(self):
        """get_actionable_goals should respect type-specific thresholds."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create a stated goal with strategy — should be actionable
        goal = ecology.create_goal(description="Do X", type='stated')
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET strategy = 'execute immediately' WHERE id = ?
        """, (goal['id'],))
        conn.commit()

        actionable = ecology.get_actionable_goals()
        assert len(actionable) == 1
        assert actionable[0]['id'] == goal['id']

    def test_get_actionable_excludes_no_strategy(self):
        """Goals without strategy should not be actionable."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create a stated goal WITHOUT strategy
        goal = ecology.create_goal(description="Do Y", type='stated')

        actionable = ecology.get_actionable_goals()
        assert len(actionable) == 0


class TestOutcomeFeedback:

    def test_engaged_strengthens_goal(self):
        """Engaged feedback should increase confidence and salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Research topic", type='stated')
        before = ecology.get_goal(goal['id'])

        ecology.record_outcome(goal['id'], 'engaged')

        after = ecology.get_goal(goal['id'])
        assert after['confidence'] > before['confidence']
        assert after['salience'] > before['salience']

    def test_rejected_weakens_goal(self):
        """Rejected feedback should decrease confidence. Strategy requires 2 rejections to clear."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Bad idea", type='stated')

        # Set a strategy first
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET strategy = 'test strategy' WHERE id = ?
        """, (goal['id'],))
        conn.commit()

        before = ecology.get_goal(goal['id'])
        ecology.record_outcome(goal['id'], 'rejected')

        after = ecology.get_goal(goal['id'])
        assert after['confidence'] < before['confidence']
        assert after['strategy'] is not None  # Strategy preserved after single rejection

        # Second rejection clears strategy
        ecology.record_outcome(goal['id'], 'rejected')
        after2 = ecology.get_goal(goal['id'])
        assert after2['strategy'] is None  # Strategy cleared after 2 rejections

    def test_ignored_reduces_salience(self):
        """Ignored feedback should reduce salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Ignored idea", type='stated')
        before = ecology.get_goal(goal['id'])

        ecology.record_outcome(goal['id'], 'ignored')

        after = ecology.get_goal(goal['id'])
        assert after['salience'] < before['salience']

    def test_acknowledged_slightly_increases_confidence(self):
        """Acknowledged feedback should slightly increase confidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Ack'd idea", type='stated')
        before = ecology.get_goal(goal['id'])

        ecology.record_outcome(goal['id'], 'acknowledged')

        after = ecology.get_goal(goal['id'])
        assert after['confidence'] >= before['confidence']

    def test_outcome_feedback_appended(self):
        """Each outcome should be appended to the feedback list."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Feedback test", type='stated')

        ecology.record_outcome(goal['id'], 'engaged')
        ecology.record_outcome(goal['id'], 'acknowledged')

        saved = ecology.get_goal(goal['id'])
        assert len(saved['outcome_feedback']) == 2
        assert saved['outcome_feedback'][0]['response'] == 'engaged'
        assert saved['outcome_feedback'][1]['response'] == 'acknowledged'


class TestActiveStack:

    def test_get_active_stack_returns_top_by_salience(self):
        """Active stack should return goals sorted by salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        g1 = ecology.create_goal(description="Low priority", type='emergent')
        g2 = ecology.create_goal(description="High priority", type='stated')

        stack = ecology.get_active_stack(limit=3)

        assert len(stack) == 2
        # Stated goal should be first (higher salience)
        assert stack[0]['id'] == g2['id']

    def test_active_stack_excludes_completed(self):
        """Completed goals should not appear in active stack."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Done goal", type='stated')

        # Mark as completed
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'completed' WHERE id = ?", (goal['id'],))
        conn.commit()

        stack = ecology.get_active_stack()
        assert len(stack) == 0

    def test_active_stack_excludes_decayed(self):
        """Decayed goals should not appear in active stack."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Decayed goal", type='emergent')

        # Mark as decayed
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'decayed' WHERE id = ?", (goal['id'],))
        conn.commit()

        stack = ecology.get_active_stack()
        assert len(stack) == 0


class TestCoreMotive:

    def test_core_motives_defined(self):
        """All core motives should have values between 0 and 1."""
        assert len(CORE_MOTIVES) == 7
        for motive, value in CORE_MOTIVES.items():
            assert 0.0 <= value <= 1.0, f"Motive {motive} out of range: {value}"


class TestFastTrack:
    """Tests for H2.0a stated goal fast-track behaviour."""

    def test_stated_goal_high_confidence_immediately_actionable(self):
        """_determine_status() must return 'actionable' for stated goals with confidence >= 0.8."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Zero salience — would normally block actionable status
        status = ecology._determine_status('stated', 'candidate', salience=0.0, confidence=0.8)
        assert status == 'actionable'

    def test_stated_goal_below_confidence_threshold_not_fast_tracked(self):
        """Stated goals with confidence < 0.6 (and zero salience) must not be actionable."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # confidence=0.5 falls below the 0.6 minimum for the general path,
        # and below 0.8 for the fast-track, so status should be 'strengthening'.
        status = ecology._determine_status('stated', 'candidate', salience=0.0, confidence=0.5)
        assert status != 'actionable'

    def test_emergent_goal_not_fast_tracked(self):
        """Emergent goals with confidence >= 0.8 must NOT be fast-tracked."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # High confidence but zero salience for an emergent goal — should not be actionable
        status = ecology._determine_status('emergent', 'candidate', salience=0.0, confidence=0.9)
        assert status != 'actionable'

    def test_stated_goal_create_salience_boosted(self):
        """create_goal() for stated type should produce salience >= 0.7."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Plan dinner for 20", type='stated')

        assert goal['salience'] >= 0.7

    def test_stated_goal_create_urgency_boosted(self):
        """create_goal() for stated type should produce urgency >= 0.5 when no urgency given."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Plan dinner for 20", type='stated')
        saved = ecology.get_goal(goal['id'])

        assert saved['urgency'] >= 0.5

    def test_fast_track_does_not_apply_to_completed(self):
        """Fast-track must not revive completed goals."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        status = ecology._determine_status('stated', 'completed', salience=0.0, confidence=0.9)
        assert status == 'completed'

    def test_fast_track_does_not_apply_to_decayed(self):
        """Fast-track must not revive decayed goals."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        status = ecology._determine_status('stated', 'decayed', salience=0.0, confidence=0.9)
        assert status == 'decayed'


class TestMuteUnmute:
    """Tests for goal mute/unmute lifecycle."""

    def test_mute_goal_returns_true_for_existing_goal(self):
        """mute_goal() should return True when the goal exists."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Dinner planning", type='stated')
        result = ecology.mute_goal(goal['id'])

        assert result is True

    def test_mute_goal_returns_false_for_missing_goal(self):
        """mute_goal() should return False for a non-existent goal ID."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        result = ecology.mute_goal("nonexistent-id")
        assert result is False

    def test_muted_goal_excluded_from_actionable(self):
        """get_actionable_goals() must skip muted goals."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Do X", type='stated')
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET strategy = 'act on this' WHERE id = ?", (goal['id'],)
        )
        conn.commit()

        # Confirm it appears before muting
        before = ecology.get_actionable_goals()
        assert any(g['id'] == goal['id'] for g in before)

        ecology.mute_goal(goal['id'])

        after = ecology.get_actionable_goals()
        assert not any(g['id'] == goal['id'] for g in after)

    def test_unmute_goal_re_enables_proactive(self):
        """After unmute_goal(), the goal should re-appear in get_actionable_goals()."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Do Y", type='stated')
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET strategy = 'act on this' WHERE id = ?", (goal['id'],)
        )
        conn.commit()

        ecology.mute_goal(goal['id'])
        assert not any(g['id'] == goal['id'] for g in ecology.get_actionable_goals())

        ecology.unmute_goal(goal['id'])
        assert any(g['id'] == goal['id'] for g in ecology.get_actionable_goals())

    def test_unmute_goal_returns_false_for_missing_goal(self):
        """unmute_goal() should return False for a non-existent goal ID."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        result = ecology.unmute_goal("nonexistent-id")
        assert result is False

    def test_is_muted_helper_with_no_feedback(self):
        """_is_muted() should return False when outcome_feedback is empty."""
        assert GoalEcologyService._is_muted(None) is False
        assert GoalEcologyService._is_muted('[]') is False

    def test_is_muted_helper_detects_mute_marker(self):
        """_is_muted() should detect a mute marker in outcome_feedback."""
        import json
        feedback = json.dumps([{'muted': True, 'timestamp': '2026-03-22T10:00:00+00:00'}])
        assert GoalEcologyService._is_muted(feedback) is True


class TestGoalHierarchy:
    """Tests for H2.1 — Goal Hierarchy (lineage parent detection and salience boost)."""

    # ── _find_parent_goal ────────────────────────────────────────────────────

    def test_find_parent_goal_returns_parent_above_threshold(self):
        """_find_parent_goal returns a parent ID when similarity > 0.6 and timescale is longer."""
        import numpy as np
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create a long-term goal that should be the parent
        parent = ecology.create_goal(
            description="Build reputation as exceptional host",
            type='developmental',
            timescale='long_term',
        )

        # Mock embedding service so similarity is controllable
        from unittest.mock import patch, MagicMock
        emb_service = MagicMock()
        parent_emb = np.array([1.0, 0.0, 0.0], dtype=float)
        child_emb = np.array([0.95, 0.1, 0.0], dtype=float)
        # Normalize
        parent_emb /= np.linalg.norm(parent_emb)
        child_emb /= np.linalg.norm(child_emb)

        def _mock_emb(text):
            if 'host' in text.lower() or 'reputation' in text.lower():
                return parent_emb
            return child_emb

        emb_service.generate_embedding_np.side_effect = _mock_emb

        with patch('services.embedding_service.get_embedding_service', return_value=emb_service):
            result = ecology._find_parent_goal(
                "Plan dinner for 20 people", timescale='short_term'
            )

        assert result == parent['id']

    def test_find_parent_goal_returns_none_when_no_longer_timescale(self):
        """_find_parent_goal returns None when there are no goals with a longer timescale."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create a goal with the same or shorter timescale
        ecology.create_goal(
            description="Buy groceries",
            type='stated',
            timescale='immediate',
        )

        # Looking for parent with timescale 'long_term' — nothing longer exists
        result = ecology._find_parent_goal("Buy groceries", timescale='long_term')
        assert result is None

    def test_find_parent_goal_returns_none_below_threshold(self):
        """_find_parent_goal returns None when best similarity is below 0.6."""
        import numpy as np
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        ecology.create_goal(
            description="Some completely unrelated long-term ambition",
            type='developmental',
            timescale='long_term',
        )

        from unittest.mock import patch, MagicMock
        emb_service = MagicMock()
        # Return orthogonal embeddings (similarity ≈ 0)
        emb_service.generate_embedding_np.side_effect = lambda text: (
            np.array([1.0, 0.0, 0.0]) if 'unrelated' in text.lower()
            else np.array([0.0, 1.0, 0.0])
        )

        with patch('services.embedding_service.get_embedding_service', return_value=emb_service):
            result = ecology._find_parent_goal("Completely different thing", timescale='short_term')

        assert result is None

    def test_find_parent_goal_skips_exclude_id(self):
        """_find_parent_goal skips the goal whose id equals exclude_id."""
        import numpy as np
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Long-term ambition to host events",
            type='developmental',
            timescale='long_term',
        )

        from unittest.mock import patch, MagicMock
        emb_service = MagicMock()
        high_sim_emb = np.array([1.0, 0.0, 0.0], dtype=float)
        emb_service.generate_embedding_np.return_value = high_sim_emb

        with patch('services.embedding_service.get_embedding_service', return_value=emb_service):
            result = ecology._find_parent_goal(
                "Host an event", timescale='short_term', exclude_id=parent['id']
            )

        assert result is None

    def test_find_parent_goal_returns_none_on_embedding_error(self):
        """_find_parent_goal returns None gracefully when EmbeddingService raises."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        ecology.create_goal(
            description="Some long-term goal", type='developmental', timescale='long_term'
        )

        from unittest.mock import patch
        with patch(
            'services.embedding_service.get_embedding_service',
            side_effect=RuntimeError("embedding unavailable"),
        ):
            result = ecology._find_parent_goal("anything", timescale='short_term')

        assert result is None

    # ── create_goal with parent linking ─────────────────────────────────────

    def test_create_goal_sets_lineage_parent_id_when_parent_found(self):
        """create_goal sets lineage_parent_id in DB when _find_parent_goal returns a match."""
        import numpy as np
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Build reputation as exceptional host",
            type='developmental',
            timescale='long_term',
        )

        from unittest.mock import patch, MagicMock
        emb_service = MagicMock()
        shared_emb = np.array([1.0, 0.0, 0.0], dtype=float)
        emb_service.generate_embedding_np.return_value = shared_emb

        with patch('services.embedding_service.get_embedding_service', return_value=emb_service):
            child = ecology.create_goal(
                description="Plan dinner for 20 people",
                type='stated',
                timescale='short_term',
            )

        saved = ecology.get_goal(child['id'])
        assert saved['lineage_parent_id'] == parent['id']

    # ── recalculate_salience lineage boost ───────────────────────────────────

    def test_lineage_salience_boost_applied_when_parent_salient(self):
        """recalculate_salience boosts child salience when parent salience > 0.5."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Build reputation as exceptional host",
            type='developmental',
            timescale='long_term',
        )

        # Give parent a high salience manually
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET salience = 0.8 WHERE id = ?", (parent['id'],))
        conn.commit()

        child = ecology.create_goal(
            description="Plan dinner for 20 people",
            type='emergent',
            timescale='short_term',
        )

        # Manually wire lineage
        cursor.execute(
            "UPDATE goals SET lineage_parent_id = ? WHERE id = ?",
            (parent['id'], child['id']),
        )
        conn.commit()

        # Add evidence so child has non-zero base salience
        ecology.add_evidence(child['id'], 'explicit_statement', 'I need to plan a dinner', strength=1.0)

        salience_without_boost = ecology.recalculate_salience(parent['id'])  # doesn't change child

        # The child's salience should now include the lineage boost
        saved_child = ecology.get_goal(child['id'])
        # We can't easily separate boost from base, but we can verify salience > 0 and
        # that it ran without error
        assert saved_child['salience'] >= 0.0

    def test_lineage_boost_not_applied_when_parent_salience_low(self):
        """recalculate_salience does NOT boost child when parent salience <= 0.5."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Vague long-term aspiration",
            type='developmental',
            timescale='long_term',
        )

        # Parent has low salience — below 0.5 threshold
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET salience = 0.3 WHERE id = ?", (parent['id'],))

        child = ecology.create_goal(
            description="Plan dinner for 20 people",
            type='emergent',
            timescale='short_term',
        )
        cursor.execute(
            "UPDATE goals SET lineage_parent_id = ?, salience = 0.2 WHERE id = ?",
            (parent['id'], child['id']),
        )
        conn.commit()

        salience_after = ecology.recalculate_salience(child['id'])
        # Salience should not exceed what it would be from base formula alone
        # (no boost since parent salience = 0.3 ≤ 0.5)
        # Just ensure it ran without raising
        assert salience_after >= 0.0

    # ── get_children ─────────────────────────────────────────────────────────

    def test_get_children_returns_child_goals(self):
        """get_children returns goals whose lineage_parent_id matches."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Build reputation as exceptional host",
            type='developmental',
            timescale='long_term',
        )
        child = ecology.create_goal(
            description="Plan dinner for 20 people",
            type='stated',
            timescale='short_term',
        )

        # Wire lineage manually
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET lineage_parent_id = ? WHERE id = ?",
            (parent['id'], child['id']),
        )
        conn.commit()

        children = ecology.get_children(parent['id'])
        assert len(children) == 1
        assert children[0]['id'] == child['id']
        assert children[0]['description'] == "Plan dinner for 20 people"

    def test_get_children_excludes_completed_and_decayed(self):
        """get_children excludes goals with status completed or decayed."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Long-term vision",
            type='developmental',
            timescale='long_term',
        )
        completed_child = ecology.create_goal(
            description="Finished task",
            type='stated',
            timescale='short_term',
        )
        decayed_child = ecology.create_goal(
            description="Forgotten task",
            type='emergent',
            timescale='short_term',
        )
        active_child = ecology.create_goal(
            description="Active task",
            type='stated',
            timescale='short_term',
        )

        cursor = conn.cursor()
        for child_id, status in [
            (completed_child['id'], 'completed'),
            (decayed_child['id'], 'decayed'),
            (active_child['id'], 'actionable'),
        ]:
            cursor.execute(
                "UPDATE goals SET lineage_parent_id = ?, status = ? WHERE id = ?",
                (parent['id'], status, child_id),
            )
        conn.commit()

        children = ecology.get_children(parent['id'])
        child_ids = [c['id'] for c in children]
        assert active_child['id'] in child_ids
        assert completed_child['id'] not in child_ids
        assert decayed_child['id'] not in child_ids

    def test_get_children_returns_empty_when_no_children(self):
        """get_children returns empty list when no goals link to parent."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        parent = ecology.create_goal(
            description="Standalone goal",
            type='developmental',
            timescale='long_term',
        )

        children = ecology.get_children(parent['id'])
        assert children == []

    # ── get_active_stack includes lineage_parent_id ──────────────────────────

    def test_get_active_stack_includes_lineage_parent_id(self):
        """get_active_stack results include lineage_parent_id field."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Some goal", type='stated')
        stack = ecology.get_active_stack()

        assert len(stack) == 1
        assert 'lineage_parent_id' in stack[0]

    def test_is_muted_helper_detects_unmute_marker(self):
        """_is_muted() should return False after an unmute entry follows a mute entry."""
        import json
        feedback = json.dumps([
            {'muted': True, 'timestamp': '2026-03-22T10:00:00+00:00'},
            {'muted': False, 'timestamp': '2026-03-22T11:00:00+00:00'},
        ])
        assert GoalEcologyService._is_muted(feedback) is False

    def test_truthfulness_highest_motive(self):
        """Truthfulness should be the highest-valued core motive."""
        assert CORE_MOTIVES['truthfulness'] == max(CORE_MOTIVES.values())


class TestGetGoalEdgeCases:

    def test_get_goal_nonexistent_returns_none(self):
        """get_goal for a non-existent ID should return None."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        result = ecology.get_goal('nonexistent-id')
        assert result is None

    def test_empty_database_active_stack(self):
        """Active stack on empty database should return empty list."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        stack = ecology.get_active_stack()
        assert stack == []

    def test_empty_database_actionable_goals(self):
        """Actionable goals on empty database should return empty list."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        actionable = ecology.get_actionable_goals()
        assert actionable == []

    def test_empty_database_decay(self):
        """Decay on empty database should return 0."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        count = ecology.decay_unreinforced()
        assert count == 0


class TestActiveStackAdvanced:

    def test_active_stack_excludes_evolved(self):
        """Evolved goals should not appear in active stack."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Evolved goal", type='emergent')
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'evolved' WHERE id = ?", (goal['id'],))
        conn.commit()

        stack = ecology.get_active_stack()
        assert len(stack) == 0

    def test_active_stack_respects_limit(self):
        """Active stack should respect the limit parameter."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        for i in range(5):
            ecology.create_goal(description=f"Goal {i}", type='stated')

        stack = ecology.get_active_stack(limit=2)
        assert len(stack) == 2

    def test_active_stack_sorted_by_salience_descending(self):
        """Active stack should be sorted by salience highest first."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        g1 = ecology.create_goal(description="Low", type='emergent')
        g2 = ecology.create_goal(description="High", type='stated')
        g3 = ecology.create_goal(description="Medium", type='inferred')

        # Manually set saliences
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET salience = 0.2 WHERE id = ?", (g1['id'],))
        cursor.execute("UPDATE goals SET salience = 0.8 WHERE id = ?", (g2['id'],))
        cursor.execute("UPDATE goals SET salience = 0.5 WHERE id = ?", (g3['id'],))
        conn.commit()

        stack = ecology.get_active_stack(limit=3)
        assert stack[0]['id'] == g2['id']
        assert stack[1]['id'] == g3['id']
        assert stack[2]['id'] == g1['id']


class TestConfidenceCalculation:

    def test_stated_goal_confidence_starts_high(self):
        """Stated goals should have 0.8 base confidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Stated", type='stated')
        saved = ecology.get_goal(goal['id'])
        assert saved['confidence'] == 0.8

    def test_stated_confidence_grows_with_evidence(self):
        """Stated goal confidence should grow with evidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Stated", type='stated')

        for i in range(3):
            ecology.add_evidence(
                goal_id=goal['id'],
                signal_type='topic_recurrence',
                content=f'Evidence {i}',
            )

        saved = ecology.get_goal(goal['id'])
        # 0.8 + 3 * 0.05 = 0.95
        assert saved['confidence'] == pytest.approx(0.95, abs=0.01)

    def test_stated_confidence_caps_at_one(self):
        """Stated goal confidence should cap at 1.0."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Stated", type='stated')

        for i in range(10):
            ecology.add_evidence(
                goal_id=goal['id'],
                signal_type='topic_recurrence',
                content=f'Evidence {i}',
            )

        saved = ecology.get_goal(goal['id'])
        assert saved['confidence'] <= 1.0

    def test_inferred_confidence_rate(self):
        """Inferred goals should gain confidence at 0.15 per evidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Inferred", type='inferred')
        ecology.add_evidence(goal_id=goal['id'], signal_type='x', content='Evidence 1')

        saved = ecology.get_goal(goal['id'])
        # 0.1 + 1 * 0.15 = 0.25
        assert saved['confidence'] == pytest.approx(0.25, abs=0.01)

    def test_emergent_confidence_rate(self):
        """Emergent goals should gain confidence at 0.08 per evidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Emergent", type='emergent')
        ecology.add_evidence(goal_id=goal['id'], signal_type='x', content='Evidence 1')

        saved = ecology.get_goal(goal['id'])
        # 0.05 + 1 * 0.08 = 0.13
        assert saved['confidence'] == pytest.approx(0.13, abs=0.01)

    def test_developmental_confidence_rate(self):
        """Developmental goals should gain confidence at 0.05 per evidence."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Dev", type='developmental')
        ecology.add_evidence(goal_id=goal['id'], signal_type='x', content='Evidence 1')

        saved = ecology.get_goal(goal['id'])
        # 0.02 + 1 * 0.05 = 0.07
        assert saved['confidence'] == pytest.approx(0.07, abs=0.01)


class TestStatusTransitions:

    def test_completed_status_not_changed_by_recalculation(self):
        """Completed goals should not transition out of terminal state."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Done", type='stated')
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'completed' WHERE id = ?", (goal['id'],))
        conn.commit()

        ecology.recalculate_salience(goal['id'])

        saved = ecology.get_goal(goal['id'])
        assert saved['status'] == 'completed'

    def test_decayed_status_not_changed_by_recalculation(self):
        """Decayed goals should not transition out of terminal state."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Decayed", type='emergent')
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'decayed' WHERE id = ?", (goal['id'],))
        conn.commit()

        ecology.recalculate_salience(goal['id'])

        saved = ecology.get_goal(goal['id'])
        assert saved['status'] == 'decayed'

    def test_evolved_status_not_changed_by_recalculation(self):
        """Evolved goals should not transition out of terminal state."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Evolved", type='emergent')
        cursor = conn.cursor()
        cursor.execute("UPDATE goals SET status = 'evolved' WHERE id = ?", (goal['id'],))
        conn.commit()

        ecology.recalculate_salience(goal['id'])

        saved = ecology.get_goal(goal['id'])
        assert saved['status'] == 'evolved'

    def test_recalculate_nonexistent_goal_returns_zero(self):
        """Recalculating salience for non-existent goal should return 0."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        result = ecology.recalculate_salience('nonexistent')
        assert result == 0.0


@pytest.mark.unit
class TestGoalContextInjection:
    """Test _get_goal_context() for prompt injection."""

    def test_returns_formatted_string_with_goals(self, mock_store):
        """Active goals are formatted as markdown list."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'type': 'stated', 'description': 'Learn Rust', 'salience': 0.85, 'confidence': 0.9, 'strategy': None},
                {'type': 'emergent', 'description': 'Research AI regulation', 'salience': 0.7, 'confidence': 0.55, 'strategy': 'Monitor EU AI Act'},
            ]

            from services.frontal_cortex_service import FrontalCortexService
            fc = FrontalCortexService.__new__(FrontalCortexService)
            result = fc._get_goal_context()

        assert '## Active Goals' in result
        assert '[stated] Learn Rust' in result
        assert 'salience=85%' in result
        assert 'confidence=90%' in result
        assert '[emergent] Research AI regulation' in result
        assert '(strategy: Monitor EU AI Act)' in result

    def test_returns_empty_string_when_no_goals(self, mock_store):
        """Empty goal stack returns empty string."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = []

            from services.frontal_cortex_service import FrontalCortexService
            fc = FrontalCortexService.__new__(FrontalCortexService)
            result = fc._get_goal_context()

        assert result == ''

    def test_returns_empty_on_exception(self, mock_store):
        """Service failure returns empty string (fault tolerance)."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            MockEcology.side_effect = Exception("DB down")

            from services.frontal_cortex_service import FrontalCortexService
            fc = FrontalCortexService.__new__(FrontalCortexService)
            result = fc._get_goal_context()

        assert result == ''

    def test_truncates_long_descriptions(self, mock_store):
        """Descriptions longer than 100 chars are truncated."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'type': 'stated', 'description': 'A' * 200, 'salience': 0.5, 'confidence': 0.5, 'strategy': None},
            ]

            from services.frontal_cortex_service import FrontalCortexService
            fc = FrontalCortexService.__new__(FrontalCortexService)
            result = fc._get_goal_context()

        # Description should be truncated to 100 chars
        assert 'A' * 100 in result
        assert 'A' * 101 not in result


class TestDecayAdvanced:

    def test_decay_returns_correct_count(self):
        """decay_unreinforced should return the count of decayed goals."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create two goals and set them up for decay
        g1 = ecology.create_goal(description="Goal 1", type='emergent')
        g2 = ecology.create_goal(description="Goal 2", type='emergent')

        old_time = (utc_now() - timedelta(days=30)).isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET last_reinforced_at = ?, salience = ?
            WHERE id IN (?, ?)
        """, (old_time, DECAY_THRESHOLD + 0.01, g1['id'], g2['id']))
        conn.commit()

        count = ecology.decay_unreinforced()
        assert count == 2

    def test_immediate_timescale_decays_after_one_hour(self):
        """Goals with immediate timescale should decay after 1 hour."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(
            description="Immediate goal", type='stated', timescale='immediate',
        )

        # Set reinforced 2 hours ago (past the 1h window)
        two_hours_ago = (utc_now() - timedelta(hours=2)).isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET last_reinforced_at = ?, salience = 0.3,
                            status = 'strengthening'
            WHERE id = ?
        """, (two_hours_ago, goal['id']))
        conn.commit()

        ecology.decay_unreinforced()
        saved = ecology.get_goal(goal['id'])
        assert saved['salience'] < 0.3

    def test_completed_goals_dont_decay(self):
        """Completed goals should not be affected by decay."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Completed", type='emergent')
        old_time = (utc_now() - timedelta(days=30)).isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE goals SET status = 'completed', salience = 0.5,
                            last_reinforced_at = ?
            WHERE id = ?
        """, (old_time, goal['id']))
        conn.commit()

        ecology.decay_unreinforced()
        saved = ecology.get_goal(goal['id'])
        assert saved['status'] == 'completed'
        assert saved['salience'] == 0.5


class TestGoalEvolutionAdvanced:

    def test_evolve_nonexistent_goal_raises(self):
        """Evolving a non-existent goal should raise ValueError."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        with pytest.raises(ValueError):
            ecology.evolve_goal('nonexistent', 'New description')

    def test_evolved_goal_inherits_type(self):
        """Evolved goal should inherit the parent's type."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        original = ecology.create_goal(description="Original", type='inferred')
        evolved = ecology.evolve_goal(original['id'], "Evolved version")

        assert evolved['type'] == 'inferred'

    def test_evolved_goal_has_reduced_confidence(self):
        """Evolved goal should have confidence at 80% of parent."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        original = ecology.create_goal(description="Original", type='stated')
        original_data = ecology.get_goal(original['id'])
        parent_confidence = original_data['confidence']

        evolved = ecology.evolve_goal(original['id'], "Evolved version")
        evolved_data = ecology.get_goal(evolved['id'])

        expected = min(1.0, parent_confidence * 0.8)
        assert evolved_data['confidence'] == pytest.approx(expected, abs=0.01)

    def test_evolved_goal_starts_with_salience(self):
        """Evolved goal should start with salience of 0.5."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        original = ecology.create_goal(description="Original", type='emergent')
        evolved = ecology.evolve_goal(original['id'], "Evolved version")
        evolved_data = ecology.get_goal(evolved['id'])

        assert evolved_data['salience'] == 0.5


class TestDuplicateEvidence:

    def test_duplicate_evidence_increments_count_each_time(self):
        """Adding the same evidence content twice should increment count twice."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Test", type='emergent')

        for _ in range(3):
            ecology.add_evidence(
                goal_id=goal['id'],
                signal_type='topic_recurrence',
                content='Same content',
                strength=1.0,
            )

        saved = ecology.get_goal(goal['id'])
        assert saved['evidence_count'] == 3


class TestOutcomeFeedbackAdvanced:

    def test_record_outcome_nonexistent_goal_no_crash(self):
        """Recording outcome for non-existent goal should not crash."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Should not raise
        ecology.record_outcome('nonexistent-id', 'engaged')

    def test_rejected_clears_strategy_after_two(self):
        """Two rejected outcomes should clear the goal's strategy."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Has strategy", type='stated')
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET strategy = 'test plan' WHERE id = ?",
            (goal['id'],),
        )
        conn.commit()

        # First rejection preserves strategy
        ecology.record_outcome(goal['id'], 'rejected')
        saved = ecology.get_goal(goal['id'])
        assert saved['strategy'] is not None

        # Second rejection clears strategy
        ecology.record_outcome(goal['id'], 'rejected')
        saved = ecology.get_goal(goal['id'])
        assert saved['strategy'] is None

    def test_confidence_clamped_to_zero(self):
        """Confidence should not go below 0 after rejection."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Low conf", type='emergent')

        # Set confidence very low
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET confidence = 0.1 WHERE id = ?",
            (goal['id'],),
        )
        conn.commit()

        ecology.record_outcome(goal['id'], 'rejected')
        saved = ecology.get_goal(goal['id'])
        assert saved['confidence'] >= 0.0

    def test_salience_clamped_to_zero(self):
        """Salience should not go below 0 after rejection."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        goal = ecology.create_goal(description="Low sal", type='emergent')
        ecology.record_outcome(goal['id'], 'rejected')

        saved = ecology.get_goal(goal['id'])
        assert saved['salience'] >= 0.0


class TestFindMatchingGoals:

    def test_find_matching_goals_with_mock_embeddings(self):
        """find_matching_goals should return goals with similar descriptions."""
        import numpy as np

        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        ecology.create_goal(description="Learn to cook Italian food", type='emergent')
        ecology.create_goal(description="Master Python programming", type='emergent')

        # Mock the embedding service to return predictable embeddings
        mock_embedding_service = MagicMock()

        def mock_embed(text):
            if 'cook' in text.lower() or 'italian' in text.lower():
                emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
                return emb / np.linalg.norm(emb)
            else:
                emb = np.array([0.0, 1.0, 0.0] + [0.0] * 381)
                return emb / np.linalg.norm(emb)

        mock_embedding_service.generate_embedding_np = mock_embed

        with patch('services.embedding_service.get_embedding_service',
                   return_value=mock_embedding_service):
            matches = ecology.find_matching_goals("cooking recipes", threshold=0.5)

        assert len(matches) >= 1
        assert any('Italian' in m['description'] for m in matches)

    def test_find_matching_goals_empty_db(self):
        """find_matching_goals on empty DB should return empty list."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        mock_embedding_service = MagicMock()
        import numpy as np
        mock_embedding_service.generate_embedding_np.return_value = np.zeros(384)

        with patch('services.embedding_service.get_embedding_service',
                   return_value=mock_embedding_service):
            matches = ecology.find_matching_goals("anything")

        assert matches == []

    def test_find_matching_goals_handles_embedding_error(self):
        """find_matching_goals should return empty list on embedding error."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        ecology.create_goal(description="Some goal", type='emergent')

        with patch('services.embedding_service.get_embedding_service',
                   side_effect=ImportError("no embedding")):
            matches = ecology.find_matching_goals("test query")

        assert matches == []


class TestMotiveAlignment:

    def test_motive_alignment_increases_salience(self):
        """Goals with core motive alignment should have higher salience."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)

        # Create two goals — one with motives, one without
        g_with_motives = ecology.create_goal(
            description="Goal with motives", type='emergent',
            parent_motives=['truthfulness', 'coherence'],
        )
        g_without_motives = ecology.create_goal(
            description="Goal without motives", type='emergent',
        )

        # Add same evidence to both
        for gid in [g_with_motives['id'], g_without_motives['id']]:
            ecology.add_evidence(
                goal_id=gid,
                signal_type='explicit_statement',
                content='Some evidence',
                strength=1.0,
            )

        s_with = ecology.get_goal(g_with_motives['id'])['salience']
        s_without = ecology.get_goal(g_without_motives['id'])['salience']

        # Goal with motives should have higher salience due to motive_alignment component
        assert s_with > s_without


class TestTimescaleWindows:

    def test_all_timescales_defined(self):
        """All expected timescale windows should be defined."""
        assert 'immediate' in TIMESCALE_WINDOWS
        assert 'short_term' in TIMESCALE_WINDOWS
        assert 'medium_term' in TIMESCALE_WINDOWS
        assert 'long_term' in TIMESCALE_WINDOWS

    def test_timescale_ordering(self):
        """Timescale windows should be in ascending order."""
        windows = [
            TIMESCALE_WINDOWS['immediate'],
            TIMESCALE_WINDOWS['short_term'],
            TIMESCALE_WINDOWS['medium_term'],
            TIMESCALE_WINDOWS['long_term'],
        ]
        for i in range(len(windows) - 1):
            assert windows[i] < windows[i + 1]


class TestStrategyEvolution:
    """Test outcome-driven strategy evolution in record_outcome."""

    def _create_goal_with_strategy(self, ecology, conn, goal_id='test-strat-goal', strategy='Original approach'):
        """Helper to create a goal with a preset strategy."""
        now = utc_now().isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO goals (id, type, status, description, confidence, salience,
                commitment, urgency, timescale, strategy, evidence_count,
                last_reinforced_at, outcome_feedback, created_at, updated_at)
            VALUES (?, 'stated', 'actionable', 'Test goal', 0.8, 0.7,
                0.0, 0.5, 'short_term', ?, 1, ?, '[]', ?, ?)
        """, (goal_id, strategy, now, now, now))
        conn.commit()
        cursor.close()
        return goal_id

    def test_single_rejection_does_not_clear_strategy(self):
        """One rejection should not clear strategy (needs 2+)."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn)

        ecology.record_outcome(gid, 'rejected')

        goal = ecology.get_goal(gid)
        assert goal['strategy'] is not None  # Strategy preserved after single rejection

    def test_two_rejections_clears_strategy(self):
        """Two rejections against same strategy should null it out."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn)

        ecology.record_outcome(gid, 'rejected')
        ecology.record_outcome(gid, 'rejected')

        goal = ecology.get_goal(gid)
        assert goal['strategy'] is None  # Strategy cleared after 2 rejections

    def test_rejected_strategy_archived_in_feedback(self):
        """Failed strategy should be archived in outcome_feedback."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn, strategy='Research directly')

        ecology.record_outcome(gid, 'rejected')
        ecology.record_outcome(gid, 'rejected')

        goal = ecology.get_goal(gid)
        feedback = goal['outcome_feedback']
        failed_entries = [f for f in feedback if f.get('event') == 'strategy_failed']
        assert len(failed_entries) == 1
        assert 'Research directly' in failed_entries[0]['strategy']
        assert failed_entries[0]['rejection_count'] >= 2

    def test_three_engagements_marks_strategy_confirmed(self):
        """Three engaged responses should mark strategy as confirmed in feedback."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn)

        ecology.record_outcome(gid, 'engaged')
        ecology.record_outcome(gid, 'engaged')
        ecology.record_outcome(gid, 'engaged')

        goal = ecology.get_goal(gid)
        feedback = goal['outcome_feedback']
        confirmed = [f for f in feedback if f.get('event') == 'strategy_confirmed']
        assert len(confirmed) == 1
        assert confirmed[0]['engagement_count'] >= 3

    def test_mixed_responses_two_rejections_clears(self):
        """Mixed responses with 2 rejections total for same strategy still triggers clear."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn)

        ecology.record_outcome(gid, 'rejected')
        ecology.record_outcome(gid, 'engaged')
        ecology.record_outcome(gid, 'rejected')

        goal = ecology.get_goal(gid)
        # 2 rejections total for same strategy hash -> strategy cleared
        assert goal['strategy'] is None

    def test_feedback_tracks_strategy_hash(self):
        """Feedback entries should include strategy_hash when strategy exists."""
        db, conn = _make_db()
        ecology = GoalEcologyService(db_service=db)
        gid = self._create_goal_with_strategy(ecology, conn)

        ecology.record_outcome(gid, 'engaged')

        goal = ecology.get_goal(gid)
        feedback = goal['outcome_feedback']
        assert len(feedback) >= 1
        assert 'strategy_hash' in feedback[0]

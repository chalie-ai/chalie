"""
Tests for PlanAction outcome learning and adaptive backoff.

Covers: on_outcome() backoff adjustments, backoff cap/floor, cooldown gate
integration with backoff, time-based decay, and procedural memory recording.
"""

import time
import pytest
from unittest.mock import patch, MagicMock, call

from services.autonomous_actions.plan_action import (
    PlanAction,
    BACKOFF_MULTIPLIER_KEY,
    LAST_CANCELLATION_KEY,
    COOLDOWN_KEY,
    COOLDOWN_TTL,
    BACKOFF_MAX,
    BACKOFF_MIN,
    BACKOFF_DECAY_DAYS,
    BACKOFF_INCREASE_FACTOR,
    BACKOFF_DECREASE_FACTOR,
)
from services.autonomous_actions.base import ThoughtContext


def _make_thought(**overrides):
    """Create a ThoughtContext suitable for PlanAction gate evaluation."""
    defaults = {
        'thought_type': 'hypothesis',
        'thought_content': 'We should research quantum computing advances',
        'activation_energy': 0.85,
        'seed_concept': 'quantum_computing',
        'seed_topic': 'technology',
    }
    defaults.update(overrides)
    return ThoughtContext(**defaults)


@pytest.mark.unit
class TestPlanActionOnOutcome:
    """Tests for PlanAction.on_outcome() backoff learning."""

    @pytest.fixture
    def action(self, mock_store):
        """PlanAction with isolated MemoryStore."""
        return PlanAction()

    @pytest.fixture
    def action_with_mocked_procedural(self, mock_store):
        """PlanAction with procedural memory mocked out."""
        action = PlanAction()
        with patch.object(action, '_record_procedural_outcome') as mock_proc, \
             patch.object(action, '_log_outcome_event'):
            action._mock_proc = mock_proc
            yield action

    # -- Cancellation backoff --------------------------------------------------

    def test_cancelled_increases_backoff(self, action, mock_store):
        """on_outcome('cancelled') should increase the backoff multiplier."""
        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('cancelled', task_id=1)

        raw = mock_store.get(BACKOFF_MULTIPLIER_KEY)
        assert raw is not None
        assert float(raw) == pytest.approx(BACKOFF_MIN * BACKOFF_INCREASE_FACTOR)

    def test_cancelled_twice_compounds(self, action, mock_store):
        """Two cancellations should compound the backoff multiplier."""
        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('cancelled', task_id=1)
            action.on_outcome('cancelled', task_id=2)

        expected = BACKOFF_MIN * BACKOFF_INCREASE_FACTOR * BACKOFF_INCREASE_FACTOR
        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(expected)

    def test_backoff_capped_at_max(self, action, mock_store):
        """Backoff multiplier should never exceed BACKOFF_MAX."""
        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            # Drive backoff past the cap
            for _ in range(20):
                action.on_outcome('cancelled', task_id=1)

        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(BACKOFF_MAX)

    def test_cancelled_stores_timestamp(self, action, mock_store):
        """Cancellation should store a last_cancellation_at timestamp."""
        before = time.time()
        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('cancelled', task_id=1)
        after = time.time()

        ts = float(mock_store.get(LAST_CANCELLATION_KEY))
        assert before <= ts <= after

    # -- Completion backoff ----------------------------------------------------

    def test_completed_decreases_backoff(self, action, mock_store):
        """on_outcome('completed') should decrease the backoff multiplier."""
        # Set up elevated backoff
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')

        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('completed', task_id=1)

        expected = 2.0 * BACKOFF_DECREASE_FACTOR
        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(expected)

    def test_backoff_floors_at_min(self, action, mock_store):
        """Backoff multiplier should never go below BACKOFF_MIN (1.0)."""
        # Start at minimum
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '1.0')

        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('completed', task_id=1)

        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(BACKOFF_MIN)

    def test_completed_from_no_backoff_stays_at_min(self, action, mock_store):
        """Completion with no prior backoff should stay at 1.0."""
        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('completed', task_id=1)

        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(BACKOFF_MIN)

    # -- Expired ---------------------------------------------------------------

    def test_expired_does_not_change_backoff(self, action, mock_store):
        """on_outcome('expired') should not change the backoff multiplier."""
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')

        with patch.object(action, '_record_procedural_outcome'), \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('expired', task_id=1)

        # Backoff unchanged
        assert float(mock_store.get(BACKOFF_MULTIPLIER_KEY)) == pytest.approx(2.0)


@pytest.mark.unit
class TestPlanActionCooldownWithBackoff:
    """Tests for cooldown gate integration with backoff multiplier."""

    @pytest.fixture
    def action(self, mock_store):
        return PlanAction()

    def test_cooldown_gate_passes_without_cooldown(self, action, mock_store):
        """Cooldown gate should pass when no cooldown key exists."""
        passes, _ = action._cooldown_gate()
        assert passes is True

    def test_cooldown_gate_blocked_during_base_cooldown(self, action, mock_store):
        """Cooldown gate should block during base cooldown period."""
        mock_store.setex(COOLDOWN_KEY, COOLDOWN_TTL, str(time.time()))
        passes, _ = action._cooldown_gate()
        assert passes is False

    def test_cooldown_uses_backoff_multiplier(self, action, mock_store):
        """Cooldown gate should extend effective cooldown by backoff multiplier."""
        # Set cooldown started 50 hours ago (past base 48h)
        cooldown_start = time.time() - (50 * 3600)
        mock_store.set(COOLDOWN_KEY, str(cooldown_start))
        # Set backoff to 2.0x (effective cooldown = 96h)
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')

        passes, effective_hours = action._cooldown_gate()
        assert passes is False
        assert effective_hours == pytest.approx(96.0, rel=0.01)

    def test_cooldown_passes_after_backoff_extended_period(self, action, mock_store):
        """Cooldown should pass once backoff-extended period has elapsed."""
        # Set cooldown started 100 hours ago, backoff 2.0x (effective = 96h)
        cooldown_start = time.time() - (100 * 3600)
        mock_store.set(COOLDOWN_KEY, str(cooldown_start))
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')

        passes, _ = action._cooldown_gate()
        assert passes is True

    def test_should_execute_uses_backoff_in_cooldown(self, action, mock_store):
        """Full should_execute() should respect backoff in cooldown gate."""
        thought = _make_thought()

        # Set cooldown started 50 hours ago (past base 48h)
        cooldown_start = time.time() - (50 * 3600)
        mock_store.set(COOLDOWN_KEY, str(cooldown_start))
        # Backoff 2.0x -> effective 96h -> 50h is still within cooldown
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')

        # Mock gates 1-6 to pass
        with patch.object(action, '_thought_type_gate', return_value=True), \
             patch.object(action, '_activation_gate', return_value=True), \
             patch.object(action, '_signal_persistence_gate', return_value=(True, 3)), \
             patch.object(action, '_actionability_gate', return_value=True), \
             patch.object(action, '_duplicate_gate', return_value=True), \
             patch.object(action, '_active_count_gate', return_value=True):

            score, eligible = action.should_execute(thought)

        assert eligible is False
        assert action.last_gate_result['gate'] == 'cooldown'


@pytest.mark.unit
class TestPlanActionBackoffDecay:
    """Tests for time-based backoff decay."""

    @pytest.fixture
    def action(self, mock_store):
        return PlanAction()

    def test_backoff_decays_after_full_period(self, action, mock_store):
        """Backoff should reset to 1.0 after BACKOFF_DECAY_DAYS with no cancellations."""
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '3.0')
        # Last cancellation 8 days ago (> 7 day decay window)
        mock_store.set(LAST_CANCELLATION_KEY, str(time.time() - 8 * 86400))

        effective = action._get_effective_backoff()
        assert effective == pytest.approx(BACKOFF_MIN)

    def test_backoff_partially_decays(self, action, mock_store):
        """Backoff should decay linearly between cancellation and full decay."""
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '3.0')
        # Last cancellation 3.5 days ago (half of 7-day window)
        mock_store.set(LAST_CANCELLATION_KEY, str(time.time() - 3.5 * 86400))

        effective = action._get_effective_backoff()
        # At 50% through decay: 3.0 - (3.0 - 1.0) * 0.5 = 2.0
        assert effective == pytest.approx(2.0, abs=0.1)

    def test_no_decay_immediately_after_cancellation(self, action, mock_store):
        """Backoff should be at full stored value immediately after cancellation."""
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '3.0')
        mock_store.set(LAST_CANCELLATION_KEY, str(time.time()))

        effective = action._get_effective_backoff()
        assert effective == pytest.approx(3.0, abs=0.05)

    def test_no_backoff_key_returns_min(self, action, mock_store):
        """No backoff key should return BACKOFF_MIN."""
        effective = action._get_effective_backoff()
        assert effective == BACKOFF_MIN

    def test_decay_cleans_up_keys_after_full_period(self, action, mock_store):
        """After full decay, MemoryStore keys should be cleaned up."""
        mock_store.set(BACKOFF_MULTIPLIER_KEY, '2.0')
        mock_store.set(LAST_CANCELLATION_KEY, str(time.time() - 8 * 86400))

        action._get_effective_backoff()

        assert mock_store.get(BACKOFF_MULTIPLIER_KEY) is None
        assert mock_store.get(LAST_CANCELLATION_KEY) is None


@pytest.mark.unit
class TestPlanActionProceduralMemory:
    """Tests for procedural memory integration."""

    @pytest.fixture
    def action(self, mock_store):
        return PlanAction()

    def test_cancellation_records_negative_reward(self, action):
        """Cancellation should record reward=-0.5 in procedural memory."""
        with patch.object(action, '_record_procedural_outcome') as mock_record, \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('cancelled', task_id=42)

        mock_record.assert_called_once_with(success=False, reward=-0.5, task_id=42)

    def test_completion_records_positive_reward(self, action):
        """Completion should record reward=0.5 in procedural memory."""
        with patch.object(action, '_record_procedural_outcome') as mock_record, \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('completed', task_id=42)

        mock_record.assert_called_once_with(success=True, reward=0.5, task_id=42)

    def test_expiry_records_neutral_reward(self, action):
        """Expiry should record reward=0.0 in procedural memory."""
        with patch.object(action, '_record_procedural_outcome') as mock_record, \
             patch.object(action, '_log_outcome_event'):
            action.on_outcome('expired', task_id=42)

        mock_record.assert_called_once_with(success=False, reward=0.0, task_id=42)

    def test_record_procedural_outcome_calls_service(self, action):
        """_record_procedural_outcome should call ProceduralMemoryService."""
        mock_proc = MagicMock()
        mock_db = MagicMock()

        with patch(
            'services.database_service.get_shared_db_service', return_value=mock_db
        ), patch(
            'services.procedural_memory_service.ProceduralMemoryService',
            return_value=mock_proc,
        ):
            action._record_procedural_outcome(success=False, reward=-0.5, task_id=10)

        mock_proc.record_action_outcome.assert_called_once_with(
            action_name='PLAN',
            success=False,
            reward=-0.5,
        )

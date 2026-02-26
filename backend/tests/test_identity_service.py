"""Tests for IdentityService — dual-channel reinforcement, inertia, drift gates, coherence."""

import json
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from services.identity_service import IdentityService


pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────

def _make_identity_vector(
    vector_name="warmth",
    baseline_weight=0.5,
    current_activation=0.5,
    plasticity_rate=0.1,
    inertia_rate=0.05,
    min_cap=0.0,
    max_cap=1.0,
    reinforcement_count=0,
    signal_history=None,
    baseline_drift_today=0.0,
    drift_window_start=None,
):
    """Return an 11-element tuple matching identity_vectors SELECT order."""
    return (
        vector_name,
        baseline_weight,
        current_activation,
        plasticity_rate,
        inertia_rate,
        min_cap,
        max_cap,
        reinforcement_count,
        signal_history or [],
        baseline_drift_today,
        drift_window_start,
    )


@pytest.fixture
def identity_service(mock_db_rows):
    db, cursor = mock_db_rows
    with patch('services.identity_service.ConfigService') as mock_config:
        mock_config.get_agent_config.return_value = {
            'reinforcement': {
                'signal_history_size': 20,
                'emotion_weight': 0.6,
                'reward_weight': 0.4,
            },
            'baseline_drift': {
                'rate': 0.005,
                'reinforcement_threshold': 10,
                'max_drift_per_day': 0.02,
                'direction_consistency_min': 0.7,
                'variance_max': 0.15,
            },
            'coherence': {
                'relational_constraints': [
                    {"a": "assertiveness", "b": "warmth", "type": "floor_ratio",
                     "a_threshold": 0.75, "b_floor": 0.35, "nudge": 0.05},
                    {"a": "assertiveness", "b": "skepticism", "type": "ceiling_pair",
                     "threshold": 0.75, "target": 0.7},
                ],
            },
        }
        svc = IdentityService(db)
    return svc, db, cursor


# ── Config defaults ──────────────────────────────────────────────────

class TestConfigDefaults:

    def test_emotion_weight_default(self, identity_service):
        svc, _, _ = identity_service
        assert svc.emotion_weight == 0.6

    def test_reward_weight_default(self, identity_service):
        svc, _, _ = identity_service
        assert svc.reward_weight == 0.4

    def test_drift_rate_default(self, identity_service):
        svc, _, _ = identity_service
        assert svc.drift_rate == 0.005

    def test_max_drift_per_day_default(self, identity_service):
        svc, _, _ = identity_service
        assert svc.max_drift_per_day == 0.02


# ── update_activation ────────────────────────────────────────────────

class TestUpdateActivation:

    def test_dual_channel_signal_math(self, identity_service):
        """total = emotion*0.6 + reward*0.4, delta = total * plasticity."""
        svc, db, cursor = identity_service
        # current=0.5, plasticity=0.1
        cursor.fetchone.return_value = (0.5, 0.1, 0.0, 1.0, 0, '[]')

        svc.update_activation('warmth', emotion_signal=1.0, reward_signal=0.5)

        # total = 1.0*0.6 + 0.5*0.4 = 0.8
        # delta = 0.8 * 0.1 = 0.08
        # new = 0.5 + 0.08 = 0.58
        call_args = cursor.execute.call_args_list
        # Find the UPDATE call (second execute)
        update_call = [c for c in call_args if 'UPDATE identity_vectors' in str(c)]
        assert len(update_call) >= 1
        # The new activation should be 0.58 (first positional arg after SET)
        update_params = update_call[0][0][1]  # tuple of params
        assert update_params[0] == pytest.approx(0.58, abs=0.001)

    def test_clamps_to_max_cap(self, identity_service):
        """Activation clamped to max_cap when signal pushes above."""
        svc, db, cursor = identity_service
        # current=0.95, plasticity=0.5, max_cap=1.0
        cursor.fetchone.return_value = (0.95, 0.5, 0.0, 1.0, 0, '[]')

        svc.update_activation('warmth', emotion_signal=1.0, reward_signal=1.0)

        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_activation = update_call[0][0][1][0]
        assert new_activation <= 1.0

    def test_clamps_to_min_cap(self, identity_service):
        """Activation clamped to min_cap when signal pushes below."""
        svc, db, cursor = identity_service
        # current=0.05, plasticity=0.5, min_cap=0.0
        cursor.fetchone.return_value = (0.05, 0.5, 0.0, 1.0, 0, '[]')

        svc.update_activation('warmth', emotion_signal=-1.0, reward_signal=-1.0)

        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_activation = update_call[0][0][1][0]
        assert new_activation >= 0.0

    def test_signal_history_appended(self, identity_service):
        """Signal history ring buffer grows by 1 entry."""
        svc, db, cursor = identity_service
        existing_history = [0.1, 0.2, 0.3]
        cursor.fetchone.return_value = (0.5, 0.1, 0.0, 1.0, 5, json.dumps(existing_history))

        svc.update_activation('warmth', emotion_signal=0.5, reward_signal=0.0)

        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_history = json.loads(update_call[0][0][1][2])
        assert len(new_history) == 4  # was 3, now 4

    def test_reinforcement_count_incremented(self, identity_service):
        """reinforcement_count should be incremented by 1."""
        svc, db, cursor = identity_service
        cursor.fetchone.return_value = (0.5, 0.1, 0.0, 1.0, 7, '[]')

        svc.update_activation('warmth', emotion_signal=0.1, reward_signal=0.0)

        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_count = update_call[0][0][1][1]
        assert new_count == 8

    def test_unknown_vector_skipped(self, identity_service):
        """Unknown vector name → no update executed."""
        svc, db, cursor = identity_service
        cursor.fetchone.return_value = None  # vector not found

        svc.update_activation('nonexistent', emotion_signal=1.0, reward_signal=1.0)

        # No UPDATE should have been issued
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'UPDATE' in str(c)]
        assert len(update_calls) == 0


# ── apply_inertia ────────────────────────────────────────────────────

class TestApplyInertia:

    def test_pulls_activation_toward_baseline(self, identity_service):
        """Inertia nudges current_activation toward baseline_weight."""
        svc, db, cursor = identity_service
        # activation=0.7, baseline=0.5, inertia_rate=0.1
        # diff = 0.5 - 0.7 = -0.2, new = 0.7 + (-0.2)*0.1 = 0.68
        cursor.fetchall.return_value = [
            ('warmth', 0.7, 0.5, 0.1, 0.0, 1.0),
        ]

        count = svc.apply_inertia()

        assert count == 1
        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_val = update_call[0][0][1][0]
        assert new_val == pytest.approx(0.68, abs=0.001)

    def test_skips_small_diff(self, identity_service):
        """diff < 0.005 → no adjustment."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('warmth', 0.501, 0.5, 0.1, 0.0, 1.0),  # diff=0.001 < 0.005
        ]

        count = svc.apply_inertia()
        assert count == 0

    def test_clamps_result_to_caps(self, identity_service):
        """Inertia result clamped within [min_cap, max_cap]."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('warmth', 0.02, 0.5, 0.9, 0.0, 0.8),  # massive pull, max_cap=0.8
        ]

        count = svc.apply_inertia()
        assert count == 1
        update_call = [c for c in cursor.execute.call_args_list
                       if 'UPDATE identity_vectors' in str(c)]
        new_val = update_call[0][0][1][0]
        assert 0.0 <= new_val <= 0.8

    def test_returns_count_of_adjusted_vectors(self, identity_service):
        """Returns number of vectors actually adjusted."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('warmth', 0.7, 0.5, 0.1, 0.0, 1.0),      # adjusted
            ('curiosity', 0.501, 0.5, 0.1, 0.0, 1.0),  # too small
        ]

        count = svc.apply_inertia()
        assert count == 1


# ── evaluate_baseline_drift — gate tests ─────────────────────────────

class TestEvaluateBaselineDrift:

    def test_gate_rejects_low_reinforcement_count(self, identity_service):
        """Gate 1: reinforcement_count < threshold → no drift."""
        svc, db, cursor = identity_service
        cursor.fetchone.return_value = (
            0.5,   # baseline
            json.dumps([0.1] * 20),  # signal_history
            5,     # reinforcement_count < 10
            0.0,   # drift_today
            None,  # drift_window
            0.0, 1.0,  # caps
        )

        svc.evaluate_baseline_drift('warmth')

        # No UPDATE that SETs baseline_weight (SELECT also contains the column name)
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 0

    def test_gate_rejects_inconsistent_direction(self, identity_service):
        """Gate 2: signals lack consistent direction → no drift."""
        svc, db, cursor = identity_service
        # Mixed signals: 5 positive, 5 negative → consistency < 0.7
        mixed = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        cursor.fetchone.return_value = (
            0.5, json.dumps(mixed), 15, 0.0, None, 0.0, 1.0,
        )

        svc.evaluate_baseline_drift('warmth')

        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 0

    def test_gate_rejects_high_variance(self, identity_service):
        """Gate 3: high signal variance → no drift."""
        svc, db, cursor = identity_service
        # All positive but high spread
        high_var = [0.01, 0.9, 0.02, 0.85, 0.03, 0.88, 0.01, 0.9, 0.02, 0.85]
        cursor.fetchone.return_value = (
            0.5, json.dumps(high_var), 15, 0.0, None, 0.0, 1.0,
        )

        svc.evaluate_baseline_drift('warmth')

        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 0

    def test_gate_rejects_exceeded_daily_budget(self, identity_service):
        """Gate 4: daily drift budget exceeded → no drift."""
        svc, db, cursor = identity_service
        consistent = [0.1] * 15
        now = datetime.now()
        cursor.fetchone.return_value = (
            0.5, json.dumps(consistent), 15,
            0.02,  # drift_today = max budget
            now - timedelta(hours=1),  # within 24h window
            0.0, 1.0,
        )

        svc.evaluate_baseline_drift('warmth')

        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 0

    def test_drift_applied_when_all_gates_pass(self, identity_service):
        """All 4 gates pass → baseline drifts by ±drift_rate."""
        svc, db, cursor = identity_service
        consistent_positive = [0.1] * 15
        cursor.fetchone.return_value = (
            0.5,                              # baseline
            json.dumps(consistent_positive),  # consistent direction
            15,                               # above threshold
            0.0,                              # no drift yet today
            None,                             # no window
            0.0, 1.0,                         # caps
        )

        svc.evaluate_baseline_drift('warmth')

        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 1
        new_baseline = update_calls[0][0][1][0]
        # Positive signals → drift_sign = +1, new = 0.5 + 0.005 = 0.505
        assert new_baseline == pytest.approx(0.505, abs=0.001)

    def test_24h_window_reset(self, identity_service):
        """Drift window >24h old → baseline_drift_today resets to 0."""
        svc, db, cursor = identity_service
        consistent = [0.1] * 15
        old_window = datetime.now() - timedelta(hours=25)
        cursor.fetchone.return_value = (
            0.5, json.dumps(consistent), 15,
            0.01,      # had some drift
            old_window,  # >24h ago → resets
            0.0, 1.0,
        )

        svc.evaluate_baseline_drift('warmth')

        # Should still apply drift (budget was reset)
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 1

    def test_drift_sign_matches_signal_average(self, identity_service):
        """Negative signal average → negative drift direction."""
        svc, db, cursor = identity_service
        consistent_negative = [-0.1] * 15
        cursor.fetchone.return_value = (
            0.5, json.dumps(consistent_negative), 15, 0.0, None, 0.0, 1.0,
        )

        svc.evaluate_baseline_drift('warmth')

        update_calls = [c for c in cursor.execute.call_args_list
                        if 'SET baseline_weight' in str(c)]
        assert len(update_calls) == 1
        new_baseline = update_calls[0][0][1][0]
        assert new_baseline == pytest.approx(0.495, abs=0.001)


# ── check_coherence ──────────────────────────────────────────────────

class TestCheckCoherence:

    def test_cap_enforcement_clamps_over_max(self, identity_service):
        """Activation above max_cap gets clamped down."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('warmth', 1.2, 0.0, 1.0),  # 1.2 > max_cap 1.0
        ]

        result = svc.check_coherence()

        assert result is False  # was incoherent
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'UPDATE identity_vectors' in str(c)]
        assert len(update_calls) >= 1

    def test_floor_ratio_nudges_b_up(self, identity_service):
        """a > threshold and b < floor → b gets nudged up."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('assertiveness', 0.8, 0.0, 1.0),  # a_val=0.8 > 0.75
            ('warmth', 0.2, 0.0, 1.0),          # b_val=0.2 < 0.35
        ]

        result = svc.check_coherence()

        assert result is False
        # warmth should have been nudged to 0.25 (0.2 + 0.05)
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'UPDATE identity_vectors' in str(c)]
        assert len(update_calls) >= 1

    def test_ceiling_pair_pulls_both_down(self, identity_service):
        """Both vectors above ceiling threshold → both pulled toward target."""
        svc, db, cursor = identity_service
        cursor.fetchall.return_value = [
            ('assertiveness', 0.85, 0.0, 1.0),  # > 0.75
            ('skepticism', 0.80, 0.0, 1.0),      # > 0.75
            ('warmth', 0.5, 0.0, 1.0),           # control — no constraint
        ]

        result = svc.check_coherence()

        assert result is False
        # Both assertiveness and skepticism should have UPDATE calls
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'UPDATE identity_vectors' in str(c)]
        assert len(update_calls) >= 2

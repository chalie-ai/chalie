"""Tests for IdentityService — dual-channel reinforcement, inertia, drift gates, coherence."""

import json
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from services.identity_service import IdentityService
from services.database_service import get_shared_db_service


pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────

def _seed_vector(db, vector_name="warmth", baseline_weight=0.5,
                 current_activation=0.5, plasticity_rate=0.1,
                 inertia_rate=0.05, min_cap=0.0, max_cap=1.0,
                 reinforcement_count=0, signal_history=None,
                 baseline_drift_today=0.0, drift_window_start=None):
    """Seed or update a single identity_vector row for test setup."""
    history_json = json.dumps(signal_history or [])
    db.execute("""
        UPDATE identity_vectors
        SET baseline_weight = ?,
            current_activation = ?,
            plasticity_rate = ?,
            inertia_rate = ?,
            min_cap = ?,
            max_cap = ?,
            reinforcement_count = ?,
            signal_history = ?,
            baseline_drift_today = ?,
            drift_window_start = ?
        WHERE vector_name = ?
    """, (baseline_weight, current_activation, plasticity_rate, inertia_rate,
          min_cap, max_cap, reinforcement_count, history_json,
          baseline_drift_today, drift_window_start, vector_name))
    db.commit()


def _read_vector(db, vector_name):
    """Read back a vector row as a dict."""
    row = db.execute(
        "SELECT * FROM identity_vectors WHERE vector_name = ?",
        (vector_name,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


@pytest.fixture
def identity_service(db):
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
        svc = IdentityService(get_shared_db_service())
    return svc


# ── Config defaults ──────────────────────────────────────────────────

class TestConfigDefaults:

    def test_emotion_weight_default(self, db, identity_service):
        assert identity_service.emotion_weight == 0.6

    def test_reward_weight_default(self, db, identity_service):
        assert identity_service.reward_weight == 0.4

    def test_drift_rate_default(self, db, identity_service):
        assert identity_service.drift_rate == 0.005

    def test_max_drift_per_day_default(self, db, identity_service):
        assert identity_service.max_drift_per_day == 0.02


# ── update_activation ────────────────────────────────────────────────

class TestUpdateActivation:

    def test_dual_channel_signal_math(self, db, identity_service):
        """total = emotion*0.6 + reward*0.4, delta = total * plasticity."""
        # current=0.5, plasticity=0.1
        _seed_vector(db, 'warmth', current_activation=0.5, plasticity_rate=0.1,
                     min_cap=0.0, max_cap=1.0, reinforcement_count=0,
                     signal_history=[])

        identity_service.update_activation('warmth', emotion_signal=1.0, reward_signal=0.5)

        # total = 1.0*0.6 + 0.5*0.4 = 0.8
        # delta = 0.8 * 0.1 = 0.08
        # new = 0.5 + 0.08 = 0.58
        row = _read_vector(db, 'warmth')
        assert row['current_activation'] == pytest.approx(0.58, abs=0.001)

    def test_clamps_to_max_cap(self, db, identity_service):
        """Activation clamped to max_cap when signal pushes above."""
        # current=0.95, plasticity=0.5, max_cap=1.0
        _seed_vector(db, 'warmth', current_activation=0.95, plasticity_rate=0.5,
                     min_cap=0.0, max_cap=1.0, reinforcement_count=0,
                     signal_history=[])

        identity_service.update_activation('warmth', emotion_signal=1.0, reward_signal=1.0)

        row = _read_vector(db, 'warmth')
        assert row['current_activation'] <= 1.0

    def test_clamps_to_min_cap(self, db, identity_service):
        """Activation clamped to min_cap when signal pushes below."""
        # current=0.05, plasticity=0.5, min_cap=0.0
        _seed_vector(db, 'warmth', current_activation=0.05, plasticity_rate=0.5,
                     min_cap=0.0, max_cap=1.0, reinforcement_count=0,
                     signal_history=[])

        identity_service.update_activation('warmth', emotion_signal=-1.0, reward_signal=-1.0)

        row = _read_vector(db, 'warmth')
        assert row['current_activation'] >= 0.0

    def test_signal_history_appended(self, db, identity_service):
        """Signal history ring buffer grows by 1 entry."""
        existing_history = [0.1, 0.2, 0.3]
        _seed_vector(db, 'warmth', current_activation=0.5, plasticity_rate=0.1,
                     min_cap=0.0, max_cap=1.0, reinforcement_count=5,
                     signal_history=existing_history)

        identity_service.update_activation('warmth', emotion_signal=0.5, reward_signal=0.0)

        row = _read_vector(db, 'warmth')
        new_history = json.loads(row['signal_history'])
        assert len(new_history) == 4  # was 3, now 4

    def test_reinforcement_count_incremented(self, db, identity_service):
        """reinforcement_count should be incremented by 1."""
        _seed_vector(db, 'warmth', current_activation=0.5, plasticity_rate=0.1,
                     min_cap=0.0, max_cap=1.0, reinforcement_count=7,
                     signal_history=[])

        identity_service.update_activation('warmth', emotion_signal=0.1, reward_signal=0.0)

        row = _read_vector(db, 'warmth')
        assert row['reinforcement_count'] == 8

    def test_unknown_vector_skipped(self, db, identity_service):
        """Unknown vector name -> no update executed."""
        identity_service.update_activation('nonexistent', emotion_signal=1.0, reward_signal=1.0)

        # Verify no crash and warmth is untouched
        row = _read_vector(db, 'warmth')
        assert row is not None


# ── apply_inertia ────────────────────────────────────────────────────

class TestApplyInertia:

    def test_pulls_activation_toward_baseline(self, db, identity_service):
        """Inertia nudges current_activation toward baseline_weight."""
        # activation=0.7, baseline=0.5, inertia_rate=0.1
        # diff = 0.5 - 0.7 = -0.2, new = 0.7 + (-0.2)*0.1 = 0.68
        _seed_vector(db, 'warmth', baseline_weight=0.5, current_activation=0.7,
                     inertia_rate=0.1, min_cap=0.0, max_cap=1.0)

        count = identity_service.apply_inertia()

        assert count >= 1
        row = _read_vector(db, 'warmth')
        assert row['current_activation'] == pytest.approx(0.68, abs=0.001)

    def test_skips_small_diff(self, db, identity_service):
        """diff < 0.005 -> no adjustment."""
        _seed_vector(db, 'warmth', baseline_weight=0.5, current_activation=0.501,
                     inertia_rate=0.1, min_cap=0.0, max_cap=1.0)
        # Reset all other vectors to their baselines so they don't count
        for name in ('curiosity', 'assertiveness', 'playfulness', 'skepticism',
                     'emotional_intensity', 'uncertainty_tolerance'):
            _seed_vector(db, name, baseline_weight=0.5, current_activation=0.5,
                         inertia_rate=0.1, min_cap=0.0, max_cap=1.0)

        count = identity_service.apply_inertia()
        # warmth diff is 0.001, below 0.005 threshold
        # Check warmth was NOT adjusted
        row = _read_vector(db, 'warmth')
        assert row['current_activation'] == pytest.approx(0.501, abs=0.0001)

    def test_clamps_result_to_caps(self, db, identity_service):
        """Inertia result clamped within [min_cap, max_cap]."""
        _seed_vector(db, 'warmth', baseline_weight=0.5, current_activation=0.02,
                     inertia_rate=0.9, min_cap=0.0, max_cap=0.8)

        count = identity_service.apply_inertia()
        assert count >= 1
        row = _read_vector(db, 'warmth')
        assert 0.0 <= row['current_activation'] <= 0.8

    def test_returns_count_of_adjusted_vectors(self, db, identity_service):
        """Returns number of vectors actually adjusted."""
        _seed_vector(db, 'warmth', baseline_weight=0.5, current_activation=0.7,
                     inertia_rate=0.1, min_cap=0.0, max_cap=1.0)
        _seed_vector(db, 'curiosity', baseline_weight=0.5, current_activation=0.501,
                     inertia_rate=0.1, min_cap=0.0, max_cap=1.0)
        # Set all other vectors to baseline so they don't count
        for name in ('assertiveness', 'playfulness', 'skepticism',
                     'emotional_intensity', 'uncertainty_tolerance'):
            _seed_vector(db, name, baseline_weight=0.5, current_activation=0.5,
                         inertia_rate=0.1, min_cap=0.0, max_cap=1.0)

        count = identity_service.apply_inertia()
        # warmth adjusted (diff=0.2), curiosity not (diff=0.001)
        assert count == 1


# ── evaluate_baseline_drift — gate tests ─────────────────────────────

class TestEvaluateBaselineDrift:

    def test_gate_rejects_low_reinforcement_count(self, db, identity_service):
        """Gate 1: reinforcement_count < threshold -> no drift."""
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=[0.1] * 20,
                     reinforcement_count=5,  # < 10 threshold
                     baseline_drift_today=0.0,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] == pytest.approx(0.5, abs=0.001)

    def test_gate_rejects_inconsistent_direction(self, db, identity_service):
        """Gate 2: signals lack consistent direction -> no drift."""
        # Mixed signals: 5 positive, 5 negative -> consistency < 0.7
        mixed = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=mixed,
                     reinforcement_count=15,
                     baseline_drift_today=0.0,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] == pytest.approx(0.5, abs=0.001)

    def test_gate_rejects_high_variance(self, db, identity_service):
        """Gate 3: high signal variance -> no drift."""
        # All positive but high spread
        high_var = [0.01, 0.9, 0.02, 0.85, 0.03, 0.88, 0.01, 0.9, 0.02, 0.85]
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=high_var,
                     reinforcement_count=15,
                     baseline_drift_today=0.0,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] == pytest.approx(0.5, abs=0.001)

    def test_gate_rejects_exceeded_daily_budget(self, db, identity_service):
        """Gate 4: daily drift budget exceeded -> no drift."""
        consistent = [0.1] * 15
        from services.time_utils import utc_now
        now = utc_now()
        recent_window = (now - timedelta(hours=1)).isoformat()
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=consistent,
                     reinforcement_count=15,
                     baseline_drift_today=0.02,  # drift_today = max budget
                     drift_window_start=recent_window,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] == pytest.approx(0.5, abs=0.001)

    def test_drift_applied_when_all_gates_pass(self, db, identity_service):
        """All 4 gates pass -> baseline drifts by +drift_rate."""
        consistent_positive = [0.1] * 15
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=consistent_positive,
                     reinforcement_count=15,
                     baseline_drift_today=0.0,
                     drift_window_start=None,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        # Positive signals -> drift_sign = +1, new = 0.5 + 0.005 = 0.505
        assert row['baseline_weight'] == pytest.approx(0.505, abs=0.001)

    def test_24h_window_reset(self, db, identity_service):
        """Drift window >24h old -> baseline_drift_today resets to 0."""
        consistent = [0.1] * 15
        from services.time_utils import utc_now
        old_window = (utc_now() - timedelta(hours=25)).isoformat()
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=consistent,
                     reinforcement_count=15,
                     baseline_drift_today=0.01,  # had some drift
                     drift_window_start=old_window,  # >24h ago -> resets
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        # Should still apply drift (budget was reset)
        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] != pytest.approx(0.5, abs=0.001)

    def test_drift_sign_matches_signal_average(self, db, identity_service):
        """Negative signal average -> negative drift direction."""
        consistent_negative = [-0.1] * 15
        _seed_vector(db, 'warmth', baseline_weight=0.5,
                     signal_history=consistent_negative,
                     reinforcement_count=15,
                     baseline_drift_today=0.0,
                     drift_window_start=None,
                     min_cap=0.0, max_cap=1.0)

        identity_service.evaluate_baseline_drift('warmth')

        row = _read_vector(db, 'warmth')
        assert row['baseline_weight'] == pytest.approx(0.495, abs=0.001)


# ── check_coherence ──────────────────────────────────────────────────

class TestCheckCoherence:

    def test_cap_enforcement_clamps_over_max(self, db, identity_service):
        """Activation above max_cap gets clamped down."""
        _seed_vector(db, 'warmth', current_activation=1.2, min_cap=0.0, max_cap=1.0)

        result = identity_service.check_coherence()

        assert result is False  # was incoherent
        row = _read_vector(db, 'warmth')
        assert row['current_activation'] <= 1.0

    def test_floor_ratio_nudges_b_up(self, db, identity_service):
        """a > threshold and b < floor -> b gets nudged up."""
        _seed_vector(db, 'assertiveness', current_activation=0.8, min_cap=0.0, max_cap=1.0)
        _seed_vector(db, 'warmth', current_activation=0.2, min_cap=0.0, max_cap=1.0)

        result = identity_service.check_coherence()

        assert result is False
        # warmth should have been nudged to 0.25 (0.2 + 0.05)
        row = _read_vector(db, 'warmth')
        assert row['current_activation'] == pytest.approx(0.25, abs=0.01)

    def test_ceiling_pair_pulls_both_down(self, db, identity_service):
        """Both vectors above ceiling threshold -> both pulled toward target."""
        _seed_vector(db, 'assertiveness', current_activation=0.85, min_cap=0.0, max_cap=1.0)
        _seed_vector(db, 'skepticism', current_activation=0.80, min_cap=0.0, max_cap=1.0)
        _seed_vector(db, 'warmth', current_activation=0.5, min_cap=0.0, max_cap=1.0)

        result = identity_service.check_coherence()

        assert result is False
        # Both assertiveness and skepticism should have been pulled toward 0.7
        a_row = _read_vector(db, 'assertiveness')
        s_row = _read_vector(db, 'skepticism')
        assert a_row['current_activation'] < 0.85
        assert s_row['current_activation'] < 0.80

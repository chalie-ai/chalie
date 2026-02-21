"""Unit tests for AdaptiveBoundaryDetector — 3-layer self-calibrating topic boundary detector."""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit_vector(dim: int = 768, seed: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


def _perturb(base: np.ndarray, noise: float = 0.02, seed: int = None) -> np.ndarray:
    """Small perturbation around a base vector — stays in same topic."""
    rng = np.random.RandomState(seed)
    v = base + rng.randn(len(base)) * noise
    return v / np.linalg.norm(v)


def _orthogonal(base: np.ndarray, seed: int = None) -> np.ndarray:
    """Generate a vector nearly orthogonal to base — new topic."""
    rng = np.random.RandomState(seed)
    v = rng.randn(len(base))
    v = v - np.dot(v, base) * base      # Gram-Schmidt
    v = v / np.linalg.norm(v)
    return v


def _make_detector(thread_id: str = "test-thread", regulator_params: dict = None,
                   initial_state: dict = None):
    """Create an AdaptiveBoundaryDetector with mocked Redis."""
    with patch('services.adaptive_boundary_detector.RedisClientService') as mock_redis_cls:
        mock_r = MagicMock()
        mock_redis_cls.create_connection.return_value = mock_r

        if initial_state is not None:
            mock_r.get.return_value = json.dumps(initial_state)
        else:
            mock_r.get.return_value = None

        from services.adaptive_boundary_detector import AdaptiveBoundaryDetector
        detector = AdaptiveBoundaryDetector(
            thread_id=thread_id,
            regulator_params=regulator_params or {}
        )
        return detector, mock_r


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveBoundaryDetector:

    def test_cold_start_uses_conservative_threshold(self):
        """< 5 messages → static 0.55 threshold, no NEWMA/surprise signals."""
        from services.adaptive_boundary_detector import COLD_START_THRESHOLD, COLD_START_MSGS

        detector, _ = _make_detector()
        base = _unit_vector(seed=1)

        # High-similarity message — should NOT fire
        result = detector.update(base, best_similarity=0.80)
        assert result.is_boundary is False
        assert result.newma_signal == 0.0
        assert result.surprise_signal == 0.0

        # Low-similarity message below cold-start threshold — SHOULD fire
        result = detector.update(base, best_similarity=COLD_START_THRESHOLD - 0.01)
        assert result.is_boundary is True

        # Verify we're still in cold-start range
        assert detector._state['msg_count'] < COLD_START_MSGS

    def test_stable_conversation_no_boundary(self):
        """20 messages with nearly identical embeddings → no boundary detected."""
        detector, _ = _make_detector()
        base = _unit_vector(seed=42)

        results = []
        for i in range(20):
            emb = _perturb(base, noise=0.01, seed=i)
            results.append(detector.update(emb, best_similarity=0.95))

        # After cold start, no boundary should fire on a stable conversation
        post_cold = results[5:]
        boundaries = [r.is_boundary for r in post_cold]
        assert not any(boundaries), f"False positives: {boundaries}"

    def test_clear_topic_switch_detected(self):
        """10 stable messages + 3 very different → boundary fires."""
        detector, _ = _make_detector()
        base = _unit_vector(seed=10)
        other = _orthogonal(base, seed=20)

        # Warm up with stable topic
        for i in range(10):
            emb = _perturb(base, noise=0.01, seed=i)
            detector.update(emb, best_similarity=0.95)

        # Now inject orthogonal topic
        fired = False
        for i in range(3):
            emb = _perturb(other, noise=0.01, seed=i + 100)
            result = detector.update(emb, best_similarity=0.20)
            if result.is_boundary:
                fired = True

        assert fired, "Expected boundary to fire after clear topic switch"

    def test_single_outlier_absorbed_no_boundary(self):
        """One anomalous message in a stable conversation → leak absorbs it."""
        detector, _ = _make_detector()
        base = _unit_vector(seed=5)

        # Establish stable pattern (past cold start)
        for i in range(12):
            emb = _perturb(base, noise=0.01, seed=i)
            detector.update(emb, best_similarity=0.92)

        # Single outlier
        outlier = _orthogonal(base, seed=99)
        result = detector.update(outlier, best_similarity=0.18)
        outlier_fired = result.is_boundary

        # Should recover — next message should not fire
        emb = _perturb(base, noise=0.01, seed=200)
        recovery_result = detector.update(emb, best_similarity=0.92)

        assert not recovery_result.is_boundary, "Should not fire after single outlier recovery"
        # Note: single outlier *may* fire if accumulator was already building;
        # the key property is that it doesn't sustain — recovery_result must be False.

    def test_gradual_drift_detected_via_newma(self):
        """Slow semantic drift over 20 messages → NEWMA catches it."""
        detector, _ = _make_detector()
        base = _unit_vector(seed=7)
        target = _orthogonal(base, seed=8)

        fired = False
        for i in range(22):
            # Gradually interpolate from base toward target
            alpha = i / 22.0
            emb = (1 - alpha) * base + alpha * target
            emb = emb / np.linalg.norm(emb)
            # Similarity also drops gradually
            similarity = max(0.15, 0.9 - alpha * 0.75)
            result = detector.update(emb, best_similarity=similarity)
            if result.is_boundary:
                fired = True

        assert fired, "Expected NEWMA to detect gradual drift over 22 messages"

    def test_state_persistence_round_trip(self):
        """save_state() writes JSON; subsequent load reconstructs it."""
        captured_payload = {}

        with patch('services.adaptive_boundary_detector.RedisClientService') as mock_cls:
            mock_r = MagicMock()
            mock_cls.create_connection.return_value = mock_r
            mock_r.get.return_value = None

            def capture_setex(key, ttl, payload):
                captured_payload['key'] = key
                captured_payload['ttl'] = ttl
                captured_payload['data'] = payload

            mock_r.setex.side_effect = capture_setex

            from services.adaptive_boundary_detector import AdaptiveBoundaryDetector
            det = AdaptiveBoundaryDetector(thread_id="persist-test")
            base = _unit_vector(seed=3)
            det.update(base, best_similarity=0.85)
            det.save_state()

        assert captured_payload['key'] == 'adaptive_boundary:persist-test'
        assert captured_payload['ttl'] == 86400

        saved = json.loads(captured_payload['data'])
        assert saved['msg_count'] == 1

        # Now reload from saved state
        with patch('services.adaptive_boundary_detector.RedisClientService') as mock_cls2:
            mock_r2 = MagicMock()
            mock_cls2.create_connection.return_value = mock_r2
            mock_r2.get.return_value = captured_payload['data']

            det2 = AdaptiveBoundaryDetector(thread_id="persist-test")
            assert det2._state['msg_count'] == 1

    def test_redis_failure_degrades_to_cold_start(self):
        """Redis unavailable → cold-start mode, no crash."""
        with patch('services.adaptive_boundary_detector.RedisClientService') as mock_cls:
            mock_r = MagicMock()
            mock_cls.create_connection.side_effect = ConnectionError("Redis down")

            from services.adaptive_boundary_detector import AdaptiveBoundaryDetector
            det = AdaptiveBoundaryDetector(thread_id="redis-fail-test")

        # Should have cold-start state
        assert det._state['msg_count'] == 0

        # update() should work without crashing
        result = det.update(_unit_vector(seed=9), best_similarity=0.80)
        assert isinstance(result.is_boundary, bool)

    def test_regulator_params_influence_sensitivity(self):
        """Higher accumulator_boundary_base makes it harder to cross threshold."""
        base = _unit_vector(seed=11)
        other = _orthogonal(base, seed=12)

        def _count_boundaries(boundary_base: float) -> int:
            det, _ = _make_detector(
                thread_id=f"sensitivity-{boundary_base}",
                regulator_params={
                    'accumulator_boundary_base': boundary_base,
                    'accumulator_leak_rate': 0.4,
                    'newma_window_fast': 4,
                    'newma_window_slow': 18,
                }
            )
            count = 0
            for i in range(8):
                emb = _perturb(base, noise=0.01, seed=i)
                det.update(emb, best_similarity=0.92)

            for i in range(6):
                emb = _perturb(other, noise=0.01, seed=i + 50)
                r = det.update(emb, best_similarity=0.20)
                if r.is_boundary:
                    count += 1
            return count

        # High base → fewer (or equal) boundaries than low base
        low_base_count = _count_boundaries(1.5)
        high_base_count = _count_boundaries(4.0)
        assert high_base_count <= low_base_count, (
            f"Higher boundary_base should produce ≤ boundaries: "
            f"low={low_base_count} high={high_base_count}"
        )


class TestJustResetFromSilence:

    def test_false_on_fresh_messages(self):
        """just_reset_from_silence is False for normal messages (no gap)."""
        detector, _ = _make_detector()
        base = _unit_vector(seed=42)
        result = detector.update(base, best_similarity=0.90)
        assert result.just_reset_from_silence is False

    def test_true_after_stale_gap(self):
        """just_reset_from_silence is True when gap > STALE_GAP_SECONDS fires."""
        import time
        from services.adaptive_boundary_detector import STALE_GAP_SECONDS

        # Initial state with a last_timestamp set far in the past
        past_ts = time.time() - STALE_GAP_SECONDS - 10
        initial_state = {
            'msg_count': 5,
            'ewma_fast': None,
            'ewma_slow': None,
            'drift_ema': 0.0,
            'drift_var_ema': 1e-4,
            'sim_ema': 0.5,
            'drop_ema': 0.0,
            'drop_var_ema': 1e-4,
            'accumulator': 0.0,
            'last_timestamp': past_ts,
        }
        detector, _ = _make_detector(initial_state=initial_state)
        base = _unit_vector(seed=10)
        result = detector.update(base, best_similarity=0.85)
        assert result.just_reset_from_silence is True

    def test_false_on_next_message_after_reset(self):
        """just_reset_from_silence is False on the message immediately after reset."""
        import time
        from services.adaptive_boundary_detector import STALE_GAP_SECONDS

        # Give it a stale timestamp
        past_ts = time.time() - STALE_GAP_SECONDS - 10
        initial_state = {
            'msg_count': 5,
            'ewma_fast': None,
            'ewma_slow': None,
            'drift_ema': 0.0,
            'drift_var_ema': 1e-4,
            'sim_ema': 0.5,
            'drop_ema': 0.0,
            'drop_var_ema': 1e-4,
            'accumulator': 0.0,
            'last_timestamp': past_ts,
        }
        detector, _ = _make_detector(initial_state=initial_state)
        base = _unit_vector(seed=10)

        # First message — triggers reset
        result1 = detector.update(base, best_similarity=0.85)
        assert result1.just_reset_from_silence is True

        # Second message — no stale gap (last_timestamp was just updated)
        result2 = detector.update(base, best_similarity=0.85)
        assert result2.just_reset_from_silence is False

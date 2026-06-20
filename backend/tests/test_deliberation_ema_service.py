"""Feature tests for DeliberationEmaService.

Asserts EMA seeding, smoothing, and bucket-transition semantics against a real
MemoryStore instance. No mocks. The 'store' fixture from conftest.py wires the
real MemoryStore singleton so all internal _load_state/_save_state paths are exercised.

pytestmark applies pytest.mark.integration to every test in this file.
"""

import json

import pytest

from services.deliberation_ema_service import DeliberationEmaService, STATE_KEY
from services.memory_store import MemoryStore

pytestmark = pytest.mark.integration

# Thresholds from the shipped deliberation-score-classifier_meta.json.
_HIGH_THR = 0.67
_MED_THR = 0.46
_ALPHA = 0.6


class TestDeliberationEmaColdStart:

    def test_cold_start_seeds_ema_equal_to_scalar(self, store: MemoryStore) -> None:
        svc = DeliberationEmaService()
        scalar = 0.75
        ema_returned, bucket = svc.update_and_bucket(scalar)

        # Return value should match the input scalar exactly (cold start = no EMA).
        assert ema_returned == pytest.approx(scalar, abs=1e-6), (
            f"Cold-start EMA should equal the raw scalar; got {ema_returned}"
        )
        assert bucket == "high", (
            f"0.75 > high_thr={_HIGH_THR}; expected 'high', got '{bucket}'"
        )

        # Confirm the MemoryStore key was written with the correct EMA value.
        raw = store.get(STATE_KEY)
        assert raw is not None, f"Expected state persisted at key '{STATE_KEY}'"
        state = json.loads(raw)
        assert state["ema"] == pytest.approx(scalar, abs=1e-4)

    def test_cold_start_low_scalar_seeds_low_ema(self, store: MemoryStore) -> None:
        svc = DeliberationEmaService()
        ema, bucket = svc.update_and_bucket(0.10)
        assert ema == pytest.approx(0.10, abs=1e-6)
        assert bucket == "low"


class TestDeliberationEmaSmoothing:

    def test_three_low_scalars_keep_bucket_low(self, store: MemoryStore) -> None:
        svc = DeliberationEmaService()
        for _ in range(3):
            ema, bucket = svc.update_and_bucket(0.10)

        # After 3 identical low inputs the EMA is still exactly 0.10
        # (alpha * 0.1 + (1-alpha) * 0.1 = 0.1 regardless of alpha).
        assert ema == pytest.approx(0.10, abs=1e-6)
        assert bucket == "low", (
            f"Three 0.1 inputs should stay 'low'; got bucket='{bucket}' ema={ema:.4f}"
        )

    def test_single_high_spike_does_not_flip_from_low_immediately(self, store: MemoryStore) -> None:
        """EMA = alpha * low_ema + (1 - alpha) * spike.

        With alpha=0.6, low_ema=0.10, spike=0.90:
            EMA = 0.6 * 0.10 + 0.4 * 0.90 = 0.06 + 0.36 = 0.42
        0.42 < med_thr (0.46) — bucket stays 'low', not promoted to 'medium'.
        """
        svc = DeliberationEmaService()
        svc.update_and_bucket(0.10)  # turn 1
        svc.update_and_bucket(0.10)  # turn 2 — ema still 0.10
        ema, bucket = svc.update_and_bucket(0.90)  # spike

        expected_ema = _ALPHA * 0.10 + (1 - _ALPHA) * 0.90
        assert ema == pytest.approx(expected_ema, abs=1e-6)
        assert bucket == "low", (
            f"Single spike after two lows should not promote bucket; "
            f"ema={ema:.4f}, expected≈{expected_ema:.4f}, bucket='{bucket}'"
        )


class TestDeliberationEmaBucketTransition:

    def test_sustained_high_scalars_flip_bucket_to_high(self, store: MemoryStore) -> None:
        svc = DeliberationEmaService()
        # Seed a low EMA.
        svc.update_and_bucket(0.10)

        # Repeatedly apply scalar=0.90 until EMA crosses high_thr.
        # Closed-form convergence: after N steps, ema_N = alpha^N * 0.10 + (1-alpha^N) * 0.90
        # We just run the real loop and assert it converges within 10 turns.
        bucket = "low"
        for _ in range(10):
            ema, bucket = svc.update_and_bucket(0.90)
            if bucket == "high":
                break

        assert bucket == "high", (
            f"10 turns of scalar=0.90 did not flip bucket to 'high' (last ema={ema:.4f})"
        )
        assert ema > _HIGH_THR

    def test_reset_clears_ema_and_next_call_is_cold_start(self, store: MemoryStore) -> None:
        svc = DeliberationEmaService()
        svc.update_and_bucket(0.90)  # prime with a high score
        svc.reset()

        # After reset, key must be absent.
        assert store.get(STATE_KEY) is None, "reset() should have deleted the state key"

        # First call after reset is a cold start — EMA seeds at the raw scalar.
        ema, bucket = svc.update_and_bucket(0.10)
        assert ema == pytest.approx(0.10, abs=1e-6)
        assert bucket == "low"

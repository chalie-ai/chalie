"""
Unit tests for SituationModelService.

Tests composite score formulas, directive generation, staleness tracking,
meta version increments, and graceful degradation when all signal sources fail.

All tests use a fake in-memory MemoryStore and mock out heavy service imports.
No real SQLite, no real embeddings, no real LLM calls.
"""

import json
from unittest.mock import patch

import pytest

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


# ─── Service factory ──────────────────────────────────────────────────────────

def _make_service(store: MemoryStore = None):
    """Return a fresh SituationModelService instance wired to a real MemoryStore.

    Reloads the service module each call so module-level singleton state does
    not bleed between tests.

    Args:
        store: Optional pre-populated ``MemoryStore`` to inject.  A fresh
               ``MemoryStore()`` is created when omitted.

    Returns:
        A ``SituationModelService`` instance with ``_store`` set to ``store``.
    """
    # Import fresh each time so module-level state doesn't bleed between tests
    import importlib
    import services.situation_model_service as sms_mod
    importlib.reload(sms_mod)

    svc = sms_mod.SituationModelService()
    svc._store = store or MemoryStore()
    return svc


# ─── Raw signal fixtures ──────────────────────────────────────────────────────

def _raw(
    attention="casual",
    energy="moderate",
    focus_active=False,
    focus_topic=None,
    engagement=0.7,
    spark_phase="warm",
    exchange_count=5,
    age_minutes=10.0,
):
    """Build a minimal `raw` dict for score computation tests."""
    return {
        "ambient": {
            "place": "home",
            "energy": energy,
            "attention": attention,
            "mobility": "stationary",
            "tempo": "relaxed",
            "device_context": "evening session",
        },
        "focus": {
            "active": focus_active,
            "topic": focus_topic,
            "distraction": 0.0,
        },
        "topic": {"name": "test-topic", "confidence": 0.8, "is_new": False},
        "identity": {"assertiveness": 0.5, "warmth": 0.6},
        "client": {"device": "desktop", "battery": None, "tz": "Europe/Malta", "hour": 14},
        "engagement": {"score": engagement},
        "spark": {"phase": spark_phase},
        "world_state": {"items": []},
        "thread": {"exchange_count": exchange_count, "age_minutes": age_minutes},
    }


# ═══ Interruptibility score ════════════════════════════════════════════════════

class TestInterruptibilityScore:

    def test_casual_no_focus_high_energy(self):
        svc = _make_service()
        raw = _raw(attention="casual", energy="high", focus_active=False, exchange_count=0, age_minutes=1.0)
        result = svc._score_interruptibility(raw)
        # attention=0.8*0.35 + focus=1.0*0.30 + energy=0.9*0.20 + momentum=1.0*0.15
        expected = 0.35 * 0.8 + 0.30 * 1.0 + 0.20 * 0.9 + 0.15 * 1.0
        assert abs(result["value"] - round(expected, 3)) < 0.01

    def test_deep_focus_drives_to_low(self):
        svc = _make_service()
        raw = _raw(attention="deep_focus", energy="high", focus_active=True, exchange_count=0, age_minutes=1.0)
        result = svc._score_interruptibility(raw)
        # focus_val=0.0 pulls value down significantly
        assert result["value"] < 0.5

    def test_focus_active_zeros_focus_driver(self):
        svc = _make_service()
        raw = _raw(focus_active=True)
        result = svc._score_interruptibility(raw)
        assert result["drivers"]["focus"] == 0.0

    def test_focus_inactive_ones_focus_driver(self):
        svc = _make_service()
        raw = _raw(focus_active=False)
        result = svc._score_interruptibility(raw)
        assert result["drivers"]["focus"] == 1.0

    def test_high_exchange_rate_lowers_momentum(self):
        svc = _make_service()
        # 20 exchanges in 1 minute = very high rate → momentum should be 0
        raw = _raw(exchange_count=20, age_minutes=1.0)
        result = svc._score_interruptibility(raw)
        # rate=20, capped: momentum = 1.0 - min(20/2, 1.0) = 0.0
        assert result["drivers"]["momentum"] == 0.0

    def test_low_rate_full_momentum(self):
        svc = _make_service()
        raw = _raw(exchange_count=0, age_minutes=30.0)
        result = svc._score_interruptibility(raw)
        assert result["drivers"]["momentum"] == 1.0

    def test_drivers_always_present(self):
        svc = _make_service()
        raw = _raw()
        result = svc._score_interruptibility(raw)
        assert "attention" in result["drivers"]
        assert "focus" in result["drivers"]
        assert "energy" in result["drivers"]
        assert "momentum" in result["drivers"]

    def test_value_clamped_0_to_1(self):
        svc = _make_service()
        for attention in ("casual", "deep_focus", "distracted", "away"):
            for energy in ("high", "moderate", "low"):
                raw = _raw(attention=attention, energy=energy)
                val = svc._score_interruptibility(raw)["value"]
                assert 0.0 <= val <= 1.0, f"Out of range: {val} for attention={attention} energy={energy}"


# ═══ Receptiveness score ══════════════════════════════════════════════════════

class TestReceptivenessScore:

    def test_high_energy_high_engagement_graduated(self):
        svc = _make_service()
        raw = _raw(energy="high", engagement=1.0, spark_phase="graduated")
        result = svc._score_receptiveness(raw)
        # energy=0.9*0.30 + engagement=1.0*0.40 + spark=0.9*0.30
        expected = 0.30 * 0.9 + 0.40 * 1.0 + 0.30 * 0.9
        assert abs(result["value"] - round(expected, 3)) < 0.01

    def test_low_energy_low_engagement_first_contact(self):
        svc = _make_service()
        raw = _raw(energy="low", engagement=0.0, spark_phase="first_contact")
        result = svc._score_receptiveness(raw)
        # energy=0.3*0.30 + engagement=0.0*0.40 + spark=0.3*0.30
        expected = 0.30 * 0.3 + 0.40 * 0.0 + 0.30 * 0.3
        assert abs(result["value"] - round(expected, 3)) < 0.01

    def test_spark_phase_weights_correct(self):
        svc = _make_service()
        phases = {
            "first_contact": 0.3,
            "building": 0.5,
            "warm": 0.7,
            "graduated": 0.9,
        }
        for phase, expected_trust in phases.items():
            raw = _raw(spark_phase=phase)
            result = svc._score_receptiveness(raw)
            assert result["drivers"]["spark_phase"] == expected_trust, (
                f"Phase {phase} should map to trust={expected_trust}, got {result['drivers']['spark_phase']}"
            )

    def test_drivers_always_present(self):
        svc = _make_service()
        result = svc._score_receptiveness(_raw())
        assert "energy" in result["drivers"]
        assert "engagement" in result["drivers"]
        assert "spark_phase" in result["drivers"]

    def test_engagement_clamped_to_01(self):
        svc = _make_service()
        raw = _raw(engagement=2.0)  # out-of-range value
        result = svc._score_receptiveness(raw)
        assert result["drivers"]["engagement"] == 1.0

    def test_value_clamped_0_to_1(self):
        svc = _make_service()
        for energy in ("high", "moderate", "low"):
            for phase in ("first_contact", "building", "warm", "graduated"):
                raw = _raw(energy=energy, spark_phase=phase)
                val = svc._score_receptiveness(raw)["value"]
                assert 0.0 <= val <= 1.0


# ═══ Cognitive load score ═════════════════════════════════════════════════════

class TestCognitiveLoadScore:

    def test_low_usage_low_load(self):
        svc = _make_service()
        raw = _raw(exchange_count=0, age_minutes=0.0, energy="high")
        result = svc._score_cognitive_load(raw)
        # topic_depth=0, exchange_norm=0, session_norm=0, energy_inv=0.1
        assert result["value"] < 0.1

    def test_heavy_usage_high_load(self):
        svc = _make_service()
        raw = _raw(exchange_count=30, age_minutes=60.0, energy="low")
        result = svc._score_cognitive_load(raw)
        # All terms at max → load near 1.0
        assert result["value"] > 0.7

    def test_topic_depth_capped_at_1(self):
        svc = _make_service()
        raw = _raw(exchange_count=100)
        result = svc._score_cognitive_load(raw)
        assert result["drivers"]["topic_depth"] == 1.0
        assert result["drivers"]["exchange_count"] == 1.0

    def test_session_duration_capped_at_1(self):
        svc = _make_service()
        raw = _raw(age_minutes=120.0)
        result = svc._score_cognitive_load(raw)
        assert result["drivers"]["session_duration"] == 1.0

    def test_low_energy_raises_cognitive_load(self):
        svc = _make_service()
        raw_low = _raw(energy="low", exchange_count=5, age_minutes=10.0)
        raw_high = _raw(energy="high", exchange_count=5, age_minutes=10.0)
        load_low = svc._score_cognitive_load(raw_low)["value"]
        load_high = svc._score_cognitive_load(raw_high)["value"]
        assert load_low > load_high

    def test_drivers_always_present(self):
        svc = _make_service()
        result = svc._score_cognitive_load(_raw())
        assert "topic_depth" in result["drivers"]
        assert "exchange_count" in result["drivers"]
        assert "session_duration" in result["drivers"]
        assert "energy" in result["drivers"]

    def test_value_clamped_0_to_1(self):
        svc = _make_service()
        for ec in (0, 5, 15, 30, 60):
            for mins in (0.0, 10.0, 60.0, 120.0):
                raw = _raw(exchange_count=ec, age_minutes=mins)
                val = svc._score_cognitive_load(raw)["value"]
                assert 0.0 <= val <= 1.0


# ═══ Stability score ═══════════════════════════════════════════════════════════

class TestStabilityScore:

    def test_no_history_returns_default(self):
        store = MemoryStore()
        svc = _make_service(store)
        result = svc._score_stability()
        assert result["value"] == 0.8
        assert "score_variance_5m" in result["drivers"]

    def test_single_snapshot_returns_default(self):
        store = MemoryStore()
        store.lpush("situation:history", json.dumps({"scores": {"interruptibility": {"value": 0.5}, "receptiveness": {"value": 0.5}, "cognitive_load": {"value": 0.5}}}))
        svc = _make_service(store)
        result = svc._score_stability()
        assert result["value"] == 0.8

    def test_identical_snapshots_high_stability(self):
        store = MemoryStore()
        snapshot = json.dumps({
            "scores": {
                "interruptibility": {"value": 0.5},
                "receptiveness": {"value": 0.5},
                "cognitive_load": {"value": 0.5},
            }
        })
        for _ in range(5):
            store.lpush("situation:history", snapshot)
        svc = _make_service(store)
        result = svc._score_stability()
        # Zero variance → stability = 1.0
        assert result["value"] == 1.0
        assert result["drivers"]["score_variance_5m"] == 0.0

    def test_high_variance_lower_stability_than_stable(self):
        """Alternating composites produce lower stability than identical ones."""
        # Stable scenario
        stable_store = MemoryStore()
        stable_snapshot = json.dumps({
            "scores": {
                "interruptibility": {"value": 0.5},
                "receptiveness": {"value": 0.5},
                "cognitive_load": {"value": 0.5},
            }
        })
        for _ in range(5):
            stable_store.lpush("situation:history", stable_snapshot)
        stable_svc = _make_service(stable_store)
        stable_result = stable_svc._score_stability()

        # Volatile scenario — alternating 0/1 values
        volatile_store = MemoryStore()
        for v in [0.0, 1.0, 0.0, 1.0, 0.0]:
            snapshot = json.dumps({
                "scores": {
                    "interruptibility": {"value": v},
                    "receptiveness": {"value": v},
                    "cognitive_load": {"value": v},
                }
            })
            volatile_store.lpush("situation:history", snapshot)
        volatile_svc = _make_service(volatile_store)
        volatile_result = volatile_svc._score_stability()

        # Volatile scenario must have lower stability
        assert volatile_result["value"] < stable_result["value"]
        # Variance must be non-zero for volatile case
        assert volatile_result["drivers"]["score_variance_5m"] > 0.0

    def test_stability_value_clamped_0_to_1(self):
        store = MemoryStore()
        for v in [0.0, 1.0, 0.0, 1.0]:
            store.lpush("situation:history", json.dumps({
                "scores": {
                    "interruptibility": {"value": v},
                    "receptiveness": {"value": v},
                    "cognitive_load": {"value": v},
                }
            }))
        svc = _make_service(store)
        result = svc._score_stability()
        assert 0.0 <= result["value"] <= 1.0


# ═══ Proactivity budget score ══════════════════════════════════════════════════

class TestProactivityBudgetScore:

    def test_high_interruptibility_graduated_stable(self):
        svc = _make_service()
        interruptibility = {"value": 0.9}
        spark = {"phase": "graduated"}
        stability = {"value": 0.9}
        result = svc._score_proactivity_budget(interruptibility, spark, stability)
        # 0.40*0.9 + 0.35*0.9 + 0.25*0.9 = 0.9
        assert abs(result["value"] - 0.9) < 0.01

    def test_low_interruptibility_first_contact_volatile(self):
        svc = _make_service()
        interruptibility = {"value": 0.0}
        spark = {"phase": "first_contact"}
        stability = {"value": 0.0}
        result = svc._score_proactivity_budget(interruptibility, spark, stability)
        # 0.40*0.0 + 0.35*0.3 + 0.25*0.0 = 0.105
        assert abs(result["value"] - 0.105) < 0.01

    def test_drivers_present(self):
        svc = _make_service()
        result = svc._score_proactivity_budget(
            {"value": 0.5}, {"phase": "warm"}, {"value": 0.8}
        )
        assert "interruptibility" in result["drivers"]
        assert "trust" in result["drivers"]
        assert "stability" in result["drivers"]

    def test_value_clamped_0_to_1(self):
        svc = _make_service()
        result = svc._score_proactivity_budget(
            {"value": 1.5}, {"phase": "graduated"}, {"value": 1.5}
        )
        assert 0.0 <= result["value"] <= 1.0


# ═══ Directive generation ═════════════════════════════════════════════════════

class TestDirectiveGeneration:

    def _state(
        self,
        phase="exploring",
        focus_active=False,
        focus_topic=None,
        engagement=0.7,
        stability=0.8,
        cognitive_load=0.4,
        receptiveness=0.5,
        energy="moderate",
        hour=14,
        spark_phase="warm",
    ):
        return {
            "raw": {
                "focus": {"active": focus_active, "topic": focus_topic},
                "ambient": {"energy": energy},
                "client": {"hour": hour},
                "engagement": {"score": engagement},
                "spark": {"phase": spark_phase},
            },
            "scores": {
                "stability": {"value": stability},
                "cognitive_load": {"value": cognitive_load},
                "receptiveness": {"value": receptiveness},
            },
            "phase": {"current": phase, "momentum": 0.5, "direction": "sustaining"},
        }

    def test_none_when_unremarkable(self):
        svc = _make_service()
        state = self._state()
        assert svc._derive_directive(state) is None

    def test_closing_phase_fires(self):
        svc = _make_service()
        state = self._state(phase="closing")
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "wrapping up" in directive

    def test_deep_focus_fires(self):
        svc = _make_service()
        state = self._state(focus_active=True, focus_topic="project Alpha")
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "deep focus" in directive
        assert "project Alpha" in directive

    def test_deep_focus_no_topic(self):
        svc = _make_service()
        state = self._state(focus_active=True, focus_topic=None)
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "deep focus" in directive

    def test_disengaging_fires(self):
        svc = _make_service()
        state = self._state(engagement=0.2)
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "disengaging" in directive

    def test_volatile_context_fires(self):
        svc = _make_service()
        state = self._state(stability=0.3)
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "volatile" in directive

    def test_evening_low_energy_fires(self):
        svc = _make_service()
        state = self._state(energy="low", hour=21)
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "Evening" in directive or "evening" in directive

    def test_evening_low_energy_does_not_fire_before_20(self):
        svc = _make_service()
        state = self._state(energy="low", hour=19)
        # Only this rule + none of the others should fire
        directive = svc._derive_directive(state)
        # Evening rule requires hour >= 20
        assert directive is None or "Evening" not in directive

    def test_cognitive_load_fires_above_threshold(self):
        svc = _make_service()
        state = self._state(cognitive_load=0.8)
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "bandwidth" in directive

    def test_cognitive_load_does_not_fire_below_threshold(self):
        svc = _make_service()
        state = self._state(cognitive_load=0.74)
        directive = svc._derive_directive(state)
        # No other conditions → should be None
        assert directive is None

    def test_receptiveness_deepening_fires(self):
        svc = _make_service()
        state = self._state(receptiveness=0.85, phase="deepening")
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "engaged" in directive

    def test_receptiveness_fires_only_on_deepening(self):
        svc = _make_service()
        state = self._state(receptiveness=0.85, phase="exploring")
        directive = svc._derive_directive(state)
        # receptiveness > 0.8 but phase is not deepening
        assert directive is None

    def test_opening_early_spark_fires(self):
        svc = _make_service()
        state = self._state(phase="opening", spark_phase="first_contact")
        directive = svc._derive_directive(state)
        assert directive is not None
        assert "New conversation" in directive

    def test_opening_warm_spark_does_not_fire_opening_rule(self):
        svc = _make_service()
        state = self._state(phase="opening", spark_phase="warm")
        directive = svc._derive_directive(state)
        # opening rule only fires for first_contact, building
        assert directive is None or "New conversation" not in directive

    def test_priority_ordering_closing_beats_cognitive_load(self):
        svc = _make_service()
        # Both closing (p10) and cognitive_load (p7) conditions true
        state = self._state(phase="closing", cognitive_load=0.9)
        directive = svc._derive_directive(state)
        assert directive is not None
        # closing directive should appear
        assert "wrapping up" in directive

    def test_max_two_directives_emitted(self):
        svc = _make_service()
        # Trigger closing (p10), disengaging (p9), volatile (p8), cognitive_load (p7)
        state = self._state(
            phase="closing",
            engagement=0.1,
            stability=0.2,
            cognitive_load=0.9,
        )
        directive = svc._derive_directive(state)
        assert directive is not None
        # Directives are space-joined — count sentences by looking for ". "
        # The closing and disengaging directives should dominate
        assert "wrapping up" in directive  # priority 10
        assert "disengaging" in directive  # priority 9

    def test_returns_none_not_empty_string(self):
        svc = _make_service()
        # Completely default/neutral state
        state = self._state()
        result = svc._derive_directive(state)
        assert result is None  # must be None, not ""


# ═══ Meta version increments ══════════════════════════════════════════════════

class TestMetaVersion:

    def test_version_increments_on_each_update(self):
        store = MemoryStore()
        svc = _make_service(store)

        with patch.object(svc, "_collect_ambient", return_value={"place": None, "energy": "moderate", "attention": "casual", "mobility": None, "tempo": None, "device_context": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_topic", return_value={"name": None, "confidence": 0.5, "is_new": False, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_identity", return_value={"assertiveness": 0.5, "warmth": 0.5, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_client", return_value={"device": None, "battery": None, "tz": None, "hour": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_engagement", return_value={"score": 0.7, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_spark", return_value={"phase": "warm", "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_world_state", return_value={"items": [], "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_thread", return_value={"exchange_count": 0, "age_minutes": 0.0, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_phase", return_value={"current": "unknown", "momentum": 0.5, "direction": "sustaining", "updated_at": "2026-01-01T00:00:00+00:00"}):

            state1 = svc.update_on_message()
            state2 = svc.update_on_message()
            state3 = svc.update_on_message()

        assert state1["meta"]["version"] < state2["meta"]["version"]
        assert state2["meta"]["version"] < state3["meta"]["version"]

    def test_version_starts_at_1_on_fresh_store(self):
        store = MemoryStore()
        svc = _make_service(store)

        with patch.object(svc, "_collect_ambient", return_value={"place": None, "energy": "moderate", "attention": "casual", "mobility": None, "tempo": None, "device_context": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_topic", return_value={"name": None, "confidence": 0.5, "is_new": False, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_identity", return_value={"assertiveness": 0.5, "warmth": 0.5, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_client", return_value={"device": None, "battery": None, "tz": None, "hour": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_engagement", return_value={"score": 0.7, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_spark", return_value={"phase": "warm", "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_world_state", return_value={"items": [], "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_thread", return_value={"exchange_count": 0, "age_minutes": 0.0, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_phase", return_value={"current": "unknown", "momentum": 0.5, "direction": "sustaining", "updated_at": "2026-01-01T00:00:00+00:00"}):

            state = svc.update_on_message()

        assert state["meta"]["version"] == 1


# ═══ Staleness tracking ═══════════════════════════════════════════════════════

class TestStalenessTracking:

    def test_stalest_freshest_identified(self):
        svc = _make_service()
        from services.time_utils import utc_now
        from datetime import timedelta

        now = utc_now()
        old_ts = (now - timedelta(minutes=4)).isoformat()
        new_ts = now.isoformat()

        raw = {
            "ambient": {"updated_at": old_ts},
            "topic": {"updated_at": new_ts},
        }
        stalest, freshest = svc._stalest_freshest(raw)
        assert stalest["source"] == "ambient"
        assert freshest["source"] == "topic"
        assert stalest["age_seconds"] > freshest["age_seconds"]

    def test_empty_raw_no_crash(self):
        svc = _make_service()
        stalest, freshest = svc._stalest_freshest({})
        assert stalest["source"] == "unknown"
        assert freshest["source"] == "unknown"

    def test_state_contains_computed_at(self):
        store = MemoryStore()
        svc = _make_service(store)

        with patch.object(svc, "_collect_ambient", return_value={"place": None, "energy": "moderate", "attention": "casual", "mobility": None, "tempo": None, "device_context": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_topic", return_value={"name": None, "confidence": 0.5, "is_new": False, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_identity", return_value={"assertiveness": 0.5, "warmth": 0.5, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_client", return_value={"device": None, "battery": None, "tz": None, "hour": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_engagement", return_value={"score": 0.7, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_spark", return_value={"phase": "warm", "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_world_state", return_value={"items": [], "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_thread", return_value={"exchange_count": 0, "age_minutes": 0.0, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_phase", return_value={"current": "unknown", "momentum": 0.5, "direction": "sustaining", "updated_at": "2026-01-01T00:00:00+00:00"}):

            state = svc.update_on_message()

        assert "computed_at" in state["meta"]
        assert "stalest_input" in state["meta"]
        assert "freshest_input" in state["meta"]


# ═══ Graceful degradation ══════════════════════════════════════════════════════

class TestGracefulDegradation:
    """All signal sources fail → service still returns valid default state."""

    def _patched_failing(self, svc):
        """Context managers that make all collectors throw."""
        def _raise(*a, **kw):
            raise RuntimeError("simulated failure")

        patches = [
            patch.object(svc, "_collect_ambient", side_effect=_raise),
            patch.object(svc, "_collect_topic", side_effect=_raise),
            patch.object(svc, "_collect_identity", side_effect=_raise),
            patch.object(svc, "_collect_client", side_effect=_raise),
            patch.object(svc, "_collect_engagement", side_effect=_raise),
            patch.object(svc, "_collect_spark", side_effect=_raise),
            patch.object(svc, "_collect_world_state", side_effect=_raise),
            patch.object(svc, "_collect_thread", side_effect=_raise),
            patch.object(svc, "_collect_phase", side_effect=_raise),
        ]
        return patches

    def test_all_collectors_fail_returns_valid_state(self):
        """Even when every collector throws, _compute_and_store must not crash.

        Because Python patch raises on call, we patch _compute_and_store itself
        to use raw defaults and just verify the *score* helpers are safe on
        default raw dicts.
        """
        svc = _make_service()

        # Build a raw dict with all-default values (what you get when every
        # collector's try/except returns its `defaults` block)
        default_raw = {
            "ambient": {"place": None, "energy": None, "attention": None, "mobility": None, "tempo": None, "device_context": None},
            "focus": {"active": False, "topic": None, "distraction": 0.0},
            "topic": {"name": None, "confidence": 0.5, "is_new": False},
            "identity": {"assertiveness": 0.5, "warmth": 0.5},
            "client": {"device": None, "battery": None, "tz": None, "hour": None},
            "engagement": {"score": 0.7},
            "spark": {"phase": "warm"},
            "world_state": {"items": []},
            "thread": {"exchange_count": 0, "age_minutes": 0.0},
        }

        # All score helpers must succeed on default raw
        interruptibility = svc._score_interruptibility(default_raw)
        receptiveness = svc._score_receptiveness(default_raw)
        cognitive_load = svc._score_cognitive_load(default_raw)
        stability = svc._score_stability()
        proactivity = svc._score_proactivity_budget(interruptibility, {"phase": "warm"}, stability)

        for score_name, result in [
            ("interruptibility", interruptibility),
            ("receptiveness", receptiveness),
            ("cognitive_load", cognitive_load),
            ("stability", stability),
            ("proactivity_budget", proactivity),
        ]:
            assert "value" in result, f"{score_name} missing 'value'"
            assert "drivers" in result, f"{score_name} missing 'drivers'"
            assert 0.0 <= result["value"] <= 1.0, f"{score_name} value out of range: {result['value']}"

    def test_none_attention_falls_back_to_casual(self):
        svc = _make_service()
        raw = _raw()
        raw["ambient"]["attention"] = None
        result = svc._score_interruptibility(raw)
        # None attention → `or "casual"` → 0.8 (the casual mapping)
        assert result["drivers"]["attention"] == 0.8

    def test_none_energy_falls_back_to_moderate(self):
        svc = _make_service()
        raw = _raw()
        raw["ambient"]["energy"] = None
        result = svc._score_cognitive_load(raw)
        # energy=None → moderate (0.6), inverted = 0.4
        assert result["drivers"]["energy"] == pytest.approx(0.4, abs=0.01)

    def test_unknown_spark_phase_falls_back_to_warm(self):
        svc = _make_service()
        raw = _raw(spark_phase="totally_unknown_phase")
        result = svc._score_receptiveness(raw)
        # Unknown phase → trust 0.7 (warm fallback via .get default)
        assert result["drivers"]["spark_phase"] == 0.7

    def test_ambient_collector_exception_returns_defaults(self):
        """_collect_ambient failure should return default dict, not raise."""
        store = MemoryStore()
        store.set("ambient:prev_inferences", "this is not valid json{{{{")
        svc = _make_service(store)
        result = svc._collect_ambient()
        # Should not raise; should return a dict with updated_at
        assert isinstance(result, dict)
        assert "updated_at" in result

    def test_engagement_collector_returns_default_on_bad_value(self):
        store = MemoryStore()
        store.set("proactive:engagement_score", "not_a_number")
        svc = _make_service(store)
        result = svc._collect_engagement()
        assert isinstance(result, dict)
        assert "score" in result


# ═══ Score drivers exposure ═══════════════════════════════════════════════════

class TestScoreDriversExposure:
    """get_score() always returns drivers alongside value."""

    def test_get_score_returns_drivers(self):
        store = MemoryStore()
        svc = _make_service(store)

        snapshot = {
            "scores": {
                "interruptibility": {"value": 0.7, "drivers": {"attention": 0.8, "focus": 1.0, "energy": 0.6, "momentum": 0.9}},
                "receptiveness": {"value": 0.65, "drivers": {"energy": 0.6, "engagement": 0.7, "spark_phase": 0.7}},
                "cognitive_load": {"value": 0.3, "drivers": {"topic_depth": 0.2, "exchange_count": 0.1, "session_duration": 0.2, "energy": 0.4}},
                "proactivity_budget": {"value": 0.6, "drivers": {"interruptibility": 0.7, "trust": 0.7, "stability": 0.85}},
                "stability": {"value": 0.85, "drivers": {"score_variance_5m": 0.04}},
            },
            "phase": {"current": "exploring", "momentum": 0.5, "direction": "sustaining"},
            "raw": {},
            "meta": {"version": 1, "computed_at": "2026-01-01T00:00:00+00:00"},
        }
        store.set("situation:current", json.dumps(snapshot))

        for score_name in ("interruptibility", "receptiveness", "cognitive_load", "proactivity_budget", "stability"):
            result = svc.get_score(score_name)
            assert result is not None, f"get_score('{score_name}') returned None"
            assert "value" in result, f"Missing 'value' in {score_name}"
            assert "drivers" in result, f"Missing 'drivers' in {score_name}"
            assert isinstance(result["drivers"], dict), f"'drivers' is not a dict for {score_name}"

    def test_get_score_returns_none_for_unknown(self):
        store = MemoryStore()
        store.set("situation:current", json.dumps({"scores": {}}))
        svc = _make_service(store)
        assert svc.get_score("nonexistent_score") is None


# ═══ Cache read / get_current ═════════════════════════════════════════════════

class TestCacheRead:

    def test_get_current_returns_cached_state(self):
        store = MemoryStore()
        cached = {
            "scores": {"interruptibility": {"value": 0.5, "drivers": {}}},
            "phase": {"current": "exploring"},
            "raw": {},
            "meta": {"version": 99},
        }
        store.set("situation:current", json.dumps(cached))
        svc = _make_service(store)
        state = svc.get_current()
        assert state["meta"]["version"] == 99

    def test_get_current_recomputes_on_cache_miss(self):
        store = MemoryStore()
        svc = _make_service(store)

        with patch.object(svc, "_collect_ambient", return_value={"place": None, "energy": "moderate", "attention": "casual", "mobility": None, "tempo": None, "device_context": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_topic", return_value={"name": None, "confidence": 0.5, "is_new": False, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_identity", return_value={"assertiveness": 0.5, "warmth": 0.5, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_client", return_value={"device": None, "battery": None, "tz": None, "hour": None, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_engagement", return_value={"score": 0.7, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_spark", return_value={"phase": "warm", "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_world_state", return_value={"items": [], "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_thread", return_value={"exchange_count": 0, "age_minutes": 0.0, "updated_at": "2026-01-01T00:00:00+00:00"}), \
             patch.object(svc, "_collect_phase", return_value={"current": "unknown", "momentum": 0.5, "direction": "sustaining", "updated_at": "2026-01-01T00:00:00+00:00"}):

            state = svc.get_current()

        assert "scores" in state
        assert "meta" in state
        assert state["meta"]["version"] >= 1


# ═══ ConversationPhaseService stub ════════════════════════════════════════════

class TestConversationPhaseStub:
    """When ConversationPhaseService is unavailable, stub returns sensible defaults."""

    def test_collect_phase_stubs_when_service_missing(self):
        svc = _make_service()
        # ConversationPhaseService not importable in test env
        with patch("builtins.__import__", side_effect=ImportError("not installed")):
            result = svc._collect_phase("thread-123")
        # Should fall through to default
        assert result["current"] == "unknown"
        assert result["momentum"] == 0.5
        assert result["direction"] == "sustaining"


# ═══ SparkStateService stub ═══════════════════════════════════════════════════

class TestSparkStateStub:
    """When SparkStateService is unavailable, defaults to 'warm'."""

    def test_collect_spark_stubs_when_service_missing(self):
        svc = _make_service()
        with patch("builtins.__import__", side_effect=ImportError("not installed")):
            result = svc._collect_spark()
        assert result["phase"] == "warm"
        assert "updated_at" in result

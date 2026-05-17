"""
Feature tests for ModeGateService — per-turn cognitive mode classifier + EMA.

Coverage:
  - ModeGateService._update_state()          : asymmetric EMA math   [unit]
  - ModeGateService._load_config()           : missing / malformed   [unit]
  - ModeGateService._load_state / _save_state: MemoryStore round-trip [integration]
  - ModeGateService.tick()                   : end-to-end persist + active set [integration]

Tool-promotion is no longer the gate's job — innate skills are always
injected via each processor's ALWAYS_AVAILABLE list and external tools are
reached via find_tools.
The classifier + state machine is retained for prompt steering and future
mode-driven features.

Test markers:
  @pytest.mark.unit        — pure-function math; no IO
  @pytest.mark.integration — touches MemoryStore

Zero mocks. MemoryStore IS the production store.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zero_state() -> dict:
    from services.mode_gate_service import ModeGateService
    return dict.fromkeys(ModeGateService.MODES, 0.0)


def _fresh_service(fire_threshold: float = 0.60):
    """Return a ModeGateService whose fire thresholds are all forced to *fire_threshold*.

    Resets the module-level caches so each test gets a clean state and forces
    a deterministic threshold vector — no need for classifier_meta.json on disk.
    """
    import services.mode_gate_service as mgm
    mgm._config_loaded = None
    mgm._fire_thresholds_cache = None

    svc = mgm.ModeGateService()
    mgm._fire_thresholds_cache = dict.fromkeys(mgm.ModeGateService.MODES, fire_threshold)
    svc._fire_thresholds = dict(mgm._fire_thresholds_cache)
    return svc


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TESTS — pure state-machine math, no IO
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestUpdateStateFireRule:
    """Fire (prob >= threshold) snaps state up to max(state, prob)."""

    def test_fire_from_zero_sets_state_to_prob(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        probs = dict.fromkeys(svc.MODES, 0.0)
        probs['research'] = 0.82

        result = svc._update_state(state, probs)

        assert result['research'] == pytest.approx(0.82)

    def test_fire_raises_state_when_prob_is_higher(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['coding'] = 0.70
        probs = dict.fromkeys(svc.MODES, 0.0)
        probs['coding'] = 0.95

        result = svc._update_state(state, probs)

        assert result['coding'] == pytest.approx(0.95)

    def test_fire_clamps_at_ceiling(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        probs = dict.fromkeys(svc.MODES, 0.0)
        probs['analyze'] = 1.0
        state = svc._update_state(state, probs)
        probs['analyze'] = 1.0
        state = svc._update_state(state, probs)

        assert state['analyze'] <= 1.0


@pytest.mark.unit
class TestUpdateStateDecayRule:
    """Miss (prob < threshold) multiplies state by decay_factor; below floor → 0.0."""

    def test_miss_decays_state(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['research'] = 0.90
        probs = dict.fromkeys(svc.MODES, 0.0)

        result = svc._update_state(state, probs)

        assert result['research'] == pytest.approx(0.90 * 0.75)

    def test_miss_below_floor_collapses_to_zero(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['write'] = 0.012
        probs = dict.fromkeys(svc.MODES, 0.0)

        result = svc._update_state(state, probs)

        assert result['write'] == pytest.approx(0.0, abs=1e-9)

    def test_independent_modes_decay_independently(self):
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['research'] = 0.80
        state['coding'] = 0.50
        probs = dict.fromkeys(svc.MODES, 0.0)
        probs['brainstorm'] = 0.75

        result = svc._update_state(state, probs)

        assert result['brainstorm'] == pytest.approx(0.75)
        assert result['research'] == pytest.approx(0.80 * 0.75)
        assert result['coding'] == pytest.approx(0.50 * 0.75)


@pytest.mark.unit
class TestDecayTrajectory:
    """Canonical 4-turn decay tail.

    fire t0 → 0.90, miss t1 → 0.675, miss t2 → 0.506, miss t3 → 0.380,
    miss t4 → 0.285 (drops below activation=0.30).
    """

    def test_four_turn_decay_crosses_activation_at_t4(self):
        svc = _fresh_service(fire_threshold=0.60)
        activation = 0.30
        state = _zero_state()
        probs_fire = dict.fromkeys(svc.MODES, 0.0)
        probs_fire['research'] = 0.90
        probs_miss = dict.fromkeys(svc.MODES, 0.0)

        state = svc._update_state(state, probs_fire)
        assert state['research'] == pytest.approx(0.900)
        assert state['research'] >= activation

        state = svc._update_state(state, probs_miss)
        assert state['research'] == pytest.approx(0.675)
        assert state['research'] >= activation

        state = svc._update_state(state, probs_miss)
        assert state['research'] >= activation

        state = svc._update_state(state, probs_miss)
        assert state['research'] >= activation

        state = svc._update_state(state, probs_miss)
        assert state['research'] < activation


@pytest.mark.unit
class TestLoadConfigFallback:
    """_load_config() returns defaults when file is missing or malformed."""

    def test_missing_file_returns_defaults(self):
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        original_path = mgm._CONFIG_PATH
        try:
            mgm._CONFIG_PATH = "/nonexistent/path/mode_gate.yaml"
            cfg = mgm._load_config()
            assert cfg['decay_factor'] == pytest.approx(0.75)
            assert cfg['activation_threshold'] == pytest.approx(0.30)
        finally:
            mgm._CONFIG_PATH = original_path
            mgm._config_loaded = None
            mgm._fire_thresholds_cache = None

    def test_malformed_yaml_returns_defaults(self):
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(": this is not valid yaml: [\n")
            bad_path = f.name

        original_path = mgm._CONFIG_PATH
        try:
            mgm._CONFIG_PATH = bad_path
            cfg = mgm._load_config()
            assert cfg['decay_factor'] == pytest.approx(0.75)
        finally:
            mgm._CONFIG_PATH = original_path
            mgm._config_loaded = None
            mgm._fire_thresholds_cache = None
            os.unlink(bad_path)

    def test_yaml_override_wins_over_default(self):
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        yaml_content = (
            "decay_factor: 0.75\n"
            "activation_threshold: 0.30\n"
            "state_floor: 0.01\n"
            "state_ceiling: 1.0\n"
            "bootstrap_on_cold_start: false\n"
            "fire_threshold_overrides:\n"
            "  research: 0.40\n"
            "  coding: null\n"
            "  brainstorm: null\n"
            "  analyze: null\n"
            "  plan: null\n"
            "  write: null\n"
            "  math: null\n"
            "  converse: null\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        original_path = mgm._CONFIG_PATH
        try:
            mgm._CONFIG_PATH = yaml_path
            cfg = mgm._load_config()
            assert cfg['fire_threshold_overrides']['research'] == pytest.approx(0.40)
            assert cfg['fire_threshold_overrides']['coding'] is None
        finally:
            mgm._CONFIG_PATH = original_path
            mgm._config_loaded = None
            mgm._fire_thresholds_cache = None
            os.unlink(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION TESTS — touch MemoryStore
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestStateRoundTrip:
    """_save_state / _load_state persist and recover values via real MemoryStore."""

    def test_save_then_load_returns_same_values(self, store):
        svc = _fresh_service()
        written = dict.fromkeys(svc.MODES, 0.0)
        written['research'] = 0.72
        written['coding'] = 0.33

        svc._save_state(written)
        recovered = svc._load_state()

        assert recovered['research'] == pytest.approx(0.72, abs=1e-5)
        assert recovered['coding'] == pytest.approx(0.33, abs=1e-5)

    def test_load_on_empty_store_returns_all_zeros(self, store):
        svc = _fresh_service()
        state = svc._load_state()

        assert all(v == pytest.approx(0.0, abs=1e-9) for v in state.values())
        assert set(state.keys()) == set(svc.MODES)

    def test_save_overwrites_previous_state(self, store):
        svc = _fresh_service()
        first = dict.fromkeys(svc.MODES, 0.0)
        first['research'] = 0.80
        svc._save_state(first)

        second = dict.fromkeys(svc.MODES, 0.0)
        second['research'] = 0.30
        svc._save_state(second)

        recovered = svc._load_state()
        assert recovered['research'] == pytest.approx(0.30, abs=1e-5)

    def test_reset_state_clears_store(self, store):
        svc = _fresh_service()
        state = dict.fromkeys(svc.MODES, 0.0)
        state['analyze'] = 0.55
        svc._save_state(state)

        svc.reset_state()

        recovered = svc._load_state()
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in recovered.values())


@pytest.mark.integration
class TestTick:
    """tick() classifies, persists, and returns the active mode set."""

    def test_returns_active_modes_above_threshold(self, store):
        svc = _fresh_service()
        svc._fire_thresholds = dict.fromkeys(svc.MODES, 0.05)
        svc._classify = lambda _text: dict.fromkeys(svc.MODES, 0.90)

        active = svc.tick("anything", turn_id="t-1")

        assert isinstance(active, set)
        assert active, "expected at least one active mode at prob=0.90"
        assert active <= set(svc.MODES)

    def test_persists_state_across_calls(self, store):
        svc = _fresh_service()
        svc._fire_thresholds = dict.fromkeys(svc.MODES, 0.05)
        svc._classify = lambda _text: dict.fromkeys(svc.MODES, 0.85)

        svc.tick("first turn", turn_id="t-a")
        state = svc._load_state()

        assert any(v > 0.0 for v in state.values())

    def test_classifier_failure_returns_empty_set(self, store):
        svc = _fresh_service()
        svc._classify = lambda _text: {}

        active = svc.tick("anything", turn_id="t-fail")

        assert active == set()

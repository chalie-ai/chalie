"""
Feature tests for the Mode Gate feature (v0.4.0).

Coverage:
  - ModeGateService._update_state()   : asymmetric EMA math  [unit]
  - ModeGateService._resolve_active_tools() : mode → tool index  [integration]
  - ModeGateService._load_state / _save_state : MemoryStore round-trip  [integration]
  - ModeGateService.get_promoted_tool_names() shadow mode  [integration]
  - ModeGateService._load_config()    : missing file / YAML override  [unit]
  - ToolRegistryService._validate_mode_declarations() : typo / missing  [integration]
  - ToolRegistryService.get_tools_by_mode() : innate + external mapping  [integration]
  - ToolSchemaService.get_external_tool_schemas() : modes key stripped  [integration]

Test markers:
  @pytest.mark.unit        — pure-function math; no IO, no collaborators
  @pytest.mark.integration — touches MemoryStore, ToolRegistryService, real registry

Zero mocks. MemoryStore IS the production store.
"""

from __future__ import annotations

import sys
import os
import pytest
import tempfile

# Make sure backend/ is importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zero_state() -> dict:
    from services.mode_gate_service import ModeGateService
    return {m: 0.0 for m in ModeGateService.MODES}


def _fresh_service(fire_threshold: float = 0.60):
    """Return a ModeGateService whose fire thresholds are all forced to *fire_threshold*.

    We do this by resetting the module-level caches and temporarily monkeypatching
    _resolve_fire_thresholds so the service uses the value we want without needing
    a real classifier_meta.json file on disk.
    """
    import services.mode_gate_service as mgm
    # Reset module-level caches so each test gets a clean state
    mgm._config_loaded = None
    mgm._fire_thresholds_cache = None

    svc = mgm.ModeGateService()
    # Override the cached thresholds after construction so math is predictable
    mgm._fire_thresholds_cache = {m: fire_threshold for m in mgm.ModeGateService.MODES}
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
        probs = {m: 0.0 for m in svc.MODES}
        probs['research'] = 0.82

        result = svc._update_state(state, probs)

        assert result['research'] == pytest.approx(0.82)

    def test_fire_does_not_reduce_higher_state(self):
        """If current state is already 0.9 and new prob is 0.70, state stays 0.9."""
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['research'] = 0.90
        probs = {m: 0.0 for m in svc.MODES}
        probs['research'] = 0.70  # fires (>= 0.60) but lower than current state

        result = svc._update_state(state, probs)

        assert result['research'] == pytest.approx(0.90)

    def test_fire_raises_state_when_prob_is_higher(self):
        """If prob=0.95 and state=0.70, new state becomes 0.95."""
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['coding'] = 0.70
        probs = {m: 0.0 for m in svc.MODES}
        probs['coding'] = 0.95

        result = svc._update_state(state, probs)

        assert result['coding'] == pytest.approx(0.95)

    def test_fire_clamps_at_ceiling(self):
        """Two fires with prob > 1.0 equivalent are clamped to STATE_CEILING=1.0."""
        svc = _fresh_service(fire_threshold=0.60)
        # First fire
        state = _zero_state()
        probs = {m: 0.0 for m in svc.MODES}
        probs['analyze'] = 1.0
        state = svc._update_state(state, probs)
        # Second fire — state already at ceiling
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
        probs = {m: 0.0 for m in svc.MODES}  # all miss

        result = svc._update_state(state, probs)

        assert result['research'] == pytest.approx(0.90 * 0.75)

    def test_miss_below_floor_collapses_to_zero(self):
        """State that would decay below state_floor (0.01) becomes 0.0."""
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['write'] = 0.012  # one decay: 0.009 < 0.01 → 0.0
        probs = {m: 0.0 for m in svc.MODES}

        result = svc._update_state(state, probs)

        assert result['write'] == 0.0

    def test_miss_on_zero_state_stays_zero(self):
        """Decaying 0.0 should remain 0.0 (0.0 * 0.75 = 0.0 < 0.01 → 0.0)."""
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        probs = {m: 0.0 for m in svc.MODES}

        result = svc._update_state(state, probs)

        assert all(v == 0.0 for v in result.values())

    def test_independent_modes_decay_independently(self):
        """Only modes that miss should decay; firing modes should snap up."""
        svc = _fresh_service(fire_threshold=0.60)
        state = _zero_state()
        state['research'] = 0.80
        state['coding'] = 0.50
        probs = {m: 0.0 for m in svc.MODES}
        probs['brainstorm'] = 0.75  # fire on brainstorm only

        result = svc._update_state(state, probs)

        assert result['brainstorm'] == pytest.approx(0.75)
        assert result['research'] == pytest.approx(0.80 * 0.75)
        assert result['coding'] == pytest.approx(0.50 * 0.75)


@pytest.mark.unit
class TestDecayTrajectory:
    """Canonical 4-turn decay tail matches spec §4.3.

    fire at t0 → state=0.90
    miss t1 → 0.675  (active: 0.675 >= 0.30)
    miss t2 → 0.506  (active)
    miss t3 → 0.380  (active)
    miss t4 → 0.285  (drops below activation=0.30)
    """

    def test_four_turn_decay_crosses_activation_at_t4(self):
        svc = _fresh_service(fire_threshold=0.60)
        activation = 0.30
        state = _zero_state()
        probs_fire = {m: 0.0 for m in svc.MODES}
        probs_fire['research'] = 0.90
        probs_miss = {m: 0.0 for m in svc.MODES}

        # t0: fire
        state = svc._update_state(state, probs_fire)
        assert state['research'] == pytest.approx(0.900)
        assert state['research'] >= activation

        # t1: miss
        state = svc._update_state(state, probs_miss)
        assert state['research'] == pytest.approx(0.675)
        assert state['research'] >= activation

        # t2: miss
        state = svc._update_state(state, probs_miss)
        assert state['research'] == pytest.approx(0.675 * 0.75, rel=1e-3)
        assert state['research'] >= activation

        # t3: miss
        state = svc._update_state(state, probs_miss)
        assert state['research'] >= activation

        # t4: miss — crosses below activation
        state = svc._update_state(state, probs_miss)
        assert state['research'] < activation


@pytest.mark.unit
class TestLoadConfigFallback:
    """_load_config() returns defaults when file is missing or malformed YAML."""

    def test_missing_file_returns_defaults(self):
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        original_path = mgm._CONFIG_PATH
        try:
            mgm._CONFIG_PATH = "/nonexistent/path/mode_gate.yaml"
            cfg = mgm._load_config()
            assert cfg['shadow_mode'] is True
            assert cfg['decay_factor'] == 0.75
            assert cfg['activation_threshold'] == 0.30
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
            assert cfg['decay_factor'] == 0.75
        finally:
            mgm._CONFIG_PATH = original_path
            mgm._config_loaded = None
            mgm._fire_thresholds_cache = None
            os.unlink(bad_path)

    def test_yaml_override_wins_over_default(self):
        """A non-null fire_threshold_override in YAML replaces the meta value."""
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        yaml_content = (
            "shadow_mode: false\n"
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
            # research override of 0.40 must be present in the loaded config
            assert cfg['fire_threshold_overrides']['research'] == pytest.approx(0.40)
            # null override stays null (falls back to meta / 0.5)
            assert cfg['fire_threshold_overrides']['coding'] is None
        finally:
            mgm._CONFIG_PATH = original_path
            mgm._config_loaded = None
            mgm._fire_thresholds_cache = None
            os.unlink(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION TESTS — touch MemoryStore and/or real registries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestStateRoundTrip:
    """_save_state / _load_state persist and recover values via real MemoryStore."""

    def test_save_then_load_returns_same_values(self, store):
        svc = _fresh_service()
        written = {m: 0.0 for m in svc.MODES}
        written['research'] = 0.72
        written['coding'] = 0.33

        svc._save_state(written)
        recovered = svc._load_state()

        assert recovered['research'] == pytest.approx(0.72, abs=1e-5)
        assert recovered['coding'] == pytest.approx(0.33, abs=1e-5)

    def test_load_on_empty_store_returns_all_zeros(self, store):
        svc = _fresh_service()
        state = svc._load_state()

        assert all(v == 0.0 for v in state.values())
        assert set(state.keys()) == set(svc.MODES)

    def test_save_overwrites_previous_state(self, store):
        svc = _fresh_service()
        first = {m: 0.0 for m in svc.MODES}
        first['research'] = 0.80
        svc._save_state(first)

        second = {m: 0.0 for m in svc.MODES}
        second['research'] = 0.30
        svc._save_state(second)

        recovered = svc._load_state()
        assert recovered['research'] == pytest.approx(0.30, abs=1e-5)

    def test_reset_state_clears_store(self, store):
        svc = _fresh_service()
        state = {m: 0.0 for m in svc.MODES}
        state['analyze'] = 0.55
        svc._save_state(state)

        svc.reset_state()

        recovered = svc._load_state()
        assert all(v == 0.0 for v in recovered.values())


@pytest.mark.integration
class TestShadowModeReturnsEmpty:
    """In shadow_mode=True, get_promoted_tool_names always returns [] while still
    persisting state and emitting the [MODE-GATE] log line."""

    def test_shadow_mode_returns_empty_list(self, store):
        import services.mode_gate_service as mgm
        mgm._config_loaded = None
        mgm._fire_thresholds_cache = None

        svc = _fresh_service()
        # Force shadow mode on
        svc._shadow = True
        # Force classifier to return high probs (would promote many tools)
        svc._fire_thresholds = {m: 0.05 for m in svc.MODES}

        # Replace _classify to avoid needing the ONNX model
        def _fake_classify(text):
            return {m: 0.90 for m in svc.MODES}

        svc._classify = _fake_classify

        result = svc.get_promoted_tool_names("search news about Mars rovers")
        assert result == []

    def test_shadow_mode_still_persists_state(self, store):
        """Even in shadow mode the state must be written so warm-up happens."""
        svc = _fresh_service()
        svc._shadow = True
        svc._fire_thresholds = {m: 0.05 for m in svc.MODES}

        def _fake_classify(text):
            return {m: 0.85 for m in svc.MODES}

        svc._classify = _fake_classify
        svc.get_promoted_tool_names("find recent AI news")

        state = svc._load_state()
        # At least one mode should have fired and be above 0
        assert any(v > 0.0 for v in state.values())


@pytest.mark.integration
class TestResolveActiveTools:
    """_resolve_active_tools returns the correct union from SKILL_MODES and TOOL_METADATA."""

    def test_research_mode_includes_search_and_news(self):
        svc = _fresh_service()
        # research → search, news per spec §5.4
        tools = svc._resolve_active_tools({'research'})
        assert 'search' in tools
        assert 'news' in tools

    def test_research_mode_includes_read_and_document(self):
        """Innate skills read + document both declared under research (§5.3)."""
        svc = _fresh_service()
        tools = svc._resolve_active_tools({'research'})
        assert 'read' in tools
        assert 'document' in tools

    def test_plan_mode_includes_goals(self):
        """goals innate skill is mapped to plan mode."""
        svc = _fresh_service()
        tools = svc._resolve_active_tools({'plan'})
        assert 'goals' in tools

    def test_empty_active_set_returns_empty(self):
        svc = _fresh_service()
        tools = svc._resolve_active_tools(set())
        assert tools == []

    def test_weather_not_in_any_mode_bucket(self):
        """weather has no modes field in TOOL_METADATA — must never appear in any bucket."""
        svc = _fresh_service()
        all_tools_across_all_modes: set = set()
        for m in svc.MODES:
            all_tools_across_all_modes.update(svc._resolve_active_tools({m}))
        assert 'weather' not in all_tools_across_all_modes

    def test_multi_mode_union(self):
        """Active {analyze, plan} returns union of both mode buckets."""
        svc = _fresh_service()
        analyze_tools = set(svc._resolve_active_tools({'analyze'}))
        plan_tools = set(svc._resolve_active_tools({'plan'}))
        combined_tools = set(svc._resolve_active_tools({'analyze', 'plan'}))
        assert combined_tools == analyze_tools | plan_tools


@pytest.mark.integration
class TestGetToolsByMode:
    """ToolRegistryService.get_tools_by_mode() builds the correct index."""

    def _get_fresh_index(self):
        import services.tool_registry_service as trs
        # Clear cache so we get a fresh build
        trs._tools_by_mode_cache = None
        from services.tool_registry_service import ToolRegistryService
        return ToolRegistryService().get_tools_by_mode()

    def test_innate_skills_in_correct_modes(self):
        index = self._get_fresh_index()
        # introspect → converse, analyze (spec §5.3)
        assert 'introspect' in index.get('converse', set())
        assert 'introspect' in index.get('analyze', set())
        # goals → plan only
        assert 'goals' in index.get('plan', set())
        # schedule → plan, brainstorm
        assert 'schedule' in index.get('plan', set())
        assert 'schedule' in index.get('brainstorm', set())

    def test_external_tools_with_modes_are_indexed(self):
        index = self._get_fresh_index()
        # search declared modes: [research, brainstorm]
        assert 'search' in index.get('research', set())
        assert 'search' in index.get('brainstorm', set())
        # news declared modes: [research, brainstorm]
        assert 'news' in index.get('research', set())

    def test_weather_absent_from_all_mode_buckets(self):
        index = self._get_fresh_index()
        for mode_bucket in index.values():
            assert 'weather' not in mode_bucket

    def test_primitives_not_in_any_mode_bucket(self):
        """memory, find_tools, review_tool_calls are always-on primitives — not gated."""
        from services.innate_skills.registry import COGNITIVE_PRIMITIVES
        index = self._get_fresh_index()
        for primitive in COGNITIVE_PRIMITIVES:
            for mode_bucket in index.values():
                assert primitive not in mode_bucket, (
                    f"Primitive '{primitive}' must not appear in mode gate index"
                )

    def test_cache_invalidated_after_unregister(self):
        """_invalidate_mode_cache() clears the index so next call rebuilds it."""
        import services.tool_registry_service as trs
        trs._tools_by_mode_cache = None
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        # Populate cache
        _ = registry.get_tools_by_mode()
        assert trs._tools_by_mode_cache is not None
        # Invalidate
        trs._invalidate_mode_cache()
        assert trs._tools_by_mode_cache is None


@pytest.mark.integration
class TestModeDeclarationValidation:
    """_validate_mode_declarations raises RuntimeError on typos and missing entries."""

    def test_invalid_mode_string_raises_runtime_error(self):
        """An innate skill declaring a misspelled mode must fail loudly at boot."""
        from services.innate_skills.registry import SKILL_MODES

        bad_modes_patch = dict(SKILL_MODES)
        bad_modes_patch['introspect'] = ['reserch']  # typo — should be 'research'

        import services.innate_skills.registry as reg
        original = reg.SKILL_MODES
        reg.SKILL_MODES = bad_modes_patch

        # Also reset tool registry singleton so validation re-runs
        import services.tool_registry_service as trs
        original_instance = trs._instance

        try:
            trs._instance = None
            trs._tools_by_mode_cache = None
            from services.tool_registry_service import ToolRegistryService
            with pytest.raises(RuntimeError, match="reserch"):
                svc = ToolRegistryService()
                svc._validate_mode_declarations()
        finally:
            reg.SKILL_MODES = original
            trs._instance = original_instance
            trs._tools_by_mode_cache = None

    def test_missing_non_primitive_innate_raises_runtime_error(self):
        """A non-primitive skill with no SKILL_MODES entry must fail at boot."""
        from services.innate_skills.registry import SKILL_MODES

        # Remove 'goals' from SKILL_MODES — ALL_SKILL_NAMES still contains it (enforced by registry)
        trimmed_modes = {k: v for k, v in SKILL_MODES.items() if k != 'goals'}

        import services.innate_skills.registry as reg
        import services.tool_registry_service as trs
        original_modes = reg.SKILL_MODES
        original_instance = trs._instance

        reg.SKILL_MODES = trimmed_modes

        try:
            trs._instance = None
            trs._tools_by_mode_cache = None
            from services.tool_registry_service import ToolRegistryService
            with pytest.raises(RuntimeError, match="goals"):
                svc = ToolRegistryService()
                svc._validate_mode_declarations()
        finally:
            reg.SKILL_MODES = original_modes
            trs._instance = original_instance
            trs._tools_by_mode_cache = None

    def test_empty_modes_list_is_legal(self):
        """modes: [] on an innate skill must NOT raise — it means find_tools only."""
        from services.innate_skills.registry import SKILL_MODES

        # Set 'goals' to empty list — should be valid
        patched_modes = dict(SKILL_MODES)
        patched_modes['goals'] = []

        import services.innate_skills.registry as reg
        import services.tool_registry_service as trs
        original_modes = reg.SKILL_MODES
        original_instance = trs._instance

        reg.SKILL_MODES = patched_modes

        try:
            trs._instance = None
            trs._tools_by_mode_cache = None
            from services.tool_registry_service import ToolRegistryService
            # Must not raise
            svc = ToolRegistryService()
            svc._validate_mode_declarations()
        finally:
            reg.SKILL_MODES = original_modes
            trs._instance = original_instance
            trs._tools_by_mode_cache = None


@pytest.mark.integration
class TestExternalSchemaModeStripped:
    """get_external_tool_schemas() must never forward the 'modes' key to the LLM."""

    def test_modes_key_absent_from_search_schema(self):
        from services.tool_schema_service import get_external_tool_schemas
        schemas = get_external_tool_schemas(['search'])
        assert len(schemas) == 1
        schema = schemas[0]
        assert 'modes' not in schema
        assert 'modes' not in schema.get('input_schema', {})

    def test_modes_key_absent_from_news_schema(self):
        from services.tool_schema_service import get_external_tool_schemas
        schemas = get_external_tool_schemas(['news'])
        assert len(schemas) == 1
        assert 'modes' not in schemas[0]

    def test_schema_still_has_required_llm_fields(self):
        """Stripping modes must not remove name, description, or input_schema."""
        from services.tool_schema_service import get_external_tool_schemas
        schemas = get_external_tool_schemas(['search'])
        schema = schemas[0]
        assert 'name' in schema
        assert 'description' in schema
        assert 'input_schema' in schema

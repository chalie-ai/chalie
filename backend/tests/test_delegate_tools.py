# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""§9a area K (K1–K10) — Delegate tools (subagent-as-tools), spec §5b / §10f.

Tests authored per the GOVERNING TEST PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the CODE is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.

The 4 delegate tools and their EXACT tool surfaces (§9a K3):
  - web_search  → always_available = ['search', 'read']
  - summariser  → always_available = ['read', 'document', 'web_search']
  - research    → always_available = ['memory', 'search', 'read', 'find_tools']
  - web_browse  → always_available = ['browser', 'read']

Spec references: §5b, §10f, §4.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DELEGATE_NAMES = ("web_search", "summariser", "research", "web_browse")

_EXPECTED_SURFACES = {
    "web_search": ["search", "read"],
    "summariser": ["read", "document", "web_search"],
    "research": ["memory", "search", "read", "find_tools"],
    "web_browse": ["browser", "read"],
}


def _get_ability(name):
    """Return the concrete Ability instance for *name* from the registry."""
    from abilities._registry import AbilityRegistry
    return AbilityRegistry.get(name)


def _captured_config(name, params=None):
    """Run the delegate's run() with MessageProcessor.process patched, and
    return the ProcessorConfig the delegate built internally.

    The delegate constructs its own config inside run() and passes it to
    MessageProcessor.process(); we intercept that call and capture the config.
    """
    ability = _get_ability(name)
    captured = {}

    def _fake_process(raw_input, config, *args, **kwargs):
        captured["raw_input"] = raw_input
        captured["config"] = config
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "stub result"

    if params is None:
        params = {"goal": "do the thing", "query": "do the thing"}

    with patch("services.message_processor.MessageProcessor.process",
               side_effect=_fake_process):
        ability.run("user", params, None)

    return captured


# ---------------------------------------------------------------------------
# K1. Clean-context loop                                                  (§5b)
# ---------------------------------------------------------------------------

class TestK1CleanContext:
    """K1: Each delegate config is a clean-context loop —
    skip_transcript=True, suppress_history=True, no personality, no history.
    get_previous_messages() yields '' under suppress_history.
    """

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_skip_transcript_true(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.skip_transcript is True

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_skip_input_row_true(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.skip_input_row is True

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_suppress_history_true(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.suppress_history is True

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_no_personality_user_definition_empty(self, name):
        """No personality / no user definition — build_user_definition → ''."""
        cfg = _captured_config(name)["config"]
        assert cfg.build_user_definition(MagicMock()) == ""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_suppress_history_yields_empty_previous_messages(self, name):
        """With suppress_history=True, get_previous_messages() returns ''.

        This is verified through the flat-path short-circuit on a real
        MessageProcessor whose config has suppress_history=True.
        """
        from services.message_processor import MessageProcessor
        cfg = _captured_config(name)["config"]
        mp = object.__new__(MessageProcessor)
        mp.config = cfg
        assert mp.get_previous_messages() == ""


# ---------------------------------------------------------------------------
# K2. Finite tool surface; discovers nothing else                         (§5b)
# ---------------------------------------------------------------------------

class TestK2FiniteSurface:
    """K2: Each delegate has a finite always_available surface and
    discoverable=[] (cannot discover anything else via find_tools)."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_discoverable_is_empty(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.discoverable == []

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_always_available_is_finite_nonempty(self, name):
        cfg = _captured_config(name)["config"]
        assert isinstance(cfg.always_available, list)
        assert len(cfg.always_available) >= 1


# ---------------------------------------------------------------------------
# K3. Tool surfaces match the mapping                                     (§5b)
# ---------------------------------------------------------------------------

class TestK3SurfaceMapping:
    """K3: The exact always_available list per delegate:
    web_search→[search,read]; summariser→[read,document,web_search];
    research→[memory,search,read,find_tools]; web_browse→[browser,read].
    """

    @pytest.mark.parametrize("name,expected", list(_EXPECTED_SURFACES.items()))
    def test_surface_exact_match(self, name, expected):
        cfg = _captured_config(name)["config"]
        assert cfg.always_available == expected, (
            f"{name} always_available={cfg.always_available!r}, "
            f"expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# K4. Summariser gets web_search, NOT raw search                          (§5b)
# ---------------------------------------------------------------------------

class TestK4SummariserGetsWebSearch:
    """K4: The summariser delegate gets web_search (so it can delegate
    further), NOT raw search."""

    def test_summariser_has_web_search(self):
        cfg = _captured_config("summariser")["config"]
        assert "web_search" in cfg.always_available

    def test_summariser_does_not_have_raw_search(self):
        cfg = _captured_config("summariser")["config"]
        assert "search" not in cfg.always_available


# ---------------------------------------------------------------------------
# K5. No delegate recursion (research can't call research)                (§5b)
# ---------------------------------------------------------------------------

class TestK5NoRecursion:
    """K5: Delegates must not be able to call delegate tools. research cannot
    call research; no delegate lists a delegate tool in its surface except the
    sanctioned summariser→web_search single-step delegation."""

    def test_research_cannot_call_research(self):
        cfg = _captured_config("research")["config"]
        assert "research" not in cfg.always_available
        assert "research" not in cfg.discoverable

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_delegate_not_in_its_own_surface(self, name):
        cfg = _captured_config(name)["config"]
        assert name not in cfg.always_available
        assert name not in cfg.discoverable

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_delegate_blocks_all_delegate_tools_except_sanctioned(self, name):
        """Every delegate name (except a sanctioned one in its own
        always_available) must be blocked from recursion. The only delegate a
        delegate may invoke is web_search (summariser), and that one must NOT
        appear in blocked."""
        cfg = _captured_config(name)["config"]
        for other in _DELEGATE_NAMES:
            if other in cfg.always_available:
                # sanctioned single-step delegation — must not be blocked
                assert other not in cfg.blocked
            else:
                assert other in cfg.blocked, (
                    f"{name} must block delegate tool {other!r} to prevent recursion"
                )


# ---------------------------------------------------------------------------
# K6. Runs through normal MessageProcessor.process — no subclass           (§5b)
# ---------------------------------------------------------------------------

class TestK6RunsThroughNormalProcess:
    """K6: A delegate runs the normal MessageProcessor.process() — it builds a
    config and calls process(); there is no MessageProcessor subclass."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_run_calls_message_processor_process(self, name):
        captured = _captured_config(name)
        # process() was patched and captured a config — proving the delegate
        # delegates to the single flat entry point.
        assert "config" in captured
        from services.processor_config import ProcessorConfig
        assert isinstance(captured["config"], ProcessorConfig)

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_passes_goal_as_raw_input(self, name):
        captured = _captured_config(name, params={"goal": "the goal", "query": "the goal"})
        assert captured["raw_input"] == "the goal"

    def test_delegates_are_not_message_processor_subclasses(self):
        """K6 / §5b property #5: a delegate "reuses the same
        MessageProcessor.process() — no subclass, no special loop. The delegate
        IS a normal ACT loop with different config." Concretely, each delegate is
        an ``Ability`` that calls the flat ``MessageProcessor.process()`` by
        composition; it is NOT itself a ``MessageProcessor`` subclass, and there
        is no dedicated delegate-specific ``MessageProcessor`` subclass.

        Global ``MessageProcessor.__subclasses__() == []`` is the post-§9b
        end-state and belongs to T13's regression guards (area P/Q), not T11 —
        the old subclasses (UserMessageProcessor, DMNMessageProcessor,
        GeoPatternProcessor, …) are sequenced for deletion there and still back
        live code paths at T11 by design.
        """
        from services.message_processor import MessageProcessor
        from abilities._base import Ability

        # Each delegate ability is an Ability (composition), not a
        # MessageProcessor subclass.
        for name in _DELEGATE_NAMES:
            ability = _get_ability(name)
            assert isinstance(ability, Ability), (
                f"delegate {name!r} must be an Ability instance"
            )
            assert not isinstance(ability, MessageProcessor), (
                f"delegate {name!r} must NOT be a MessageProcessor (it calls "
                f"MessageProcessor.process() by composition — §5b property #5)"
            )

        # No class that subclasses MessageProcessor is a dedicated delegate
        # processor (named for a delegate or marked "Delegate").
        delegate_names_lower = {n.replace("_", "") for n in _DELEGATE_NAMES}
        for sub in MessageProcessor.__subclasses__():
            sub_lower = sub.__name__.lower()
            assert "delegate" not in sub_lower, (
                f"unexpected delegate-specific MessageProcessor subclass: "
                f"{sub.__name__} (delegates must reuse the flat process())"
            )
            for dname in delegate_names_lower:
                assert dname not in sub_lower, (
                    f"MessageProcessor subclass {sub.__name__} is named for "
                    f"delegate {dname!r}; delegates must reuse the flat "
                    f"process(), not subclass it (§5b property #5)"
                )


# ---------------------------------------------------------------------------
# K7. Delegate config silent — broadcast_to=None, memory_seed=False        (§5b)
# ---------------------------------------------------------------------------

class TestK7DelegateConfigSilent:
    """K7: Delegate configs are silent — broadcast_to=None (no live UI feed)
    and memory_seed=False (delegates don't auto-seed memory)."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_broadcast_to_none(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.broadcast_to is None

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_memory_seed_false(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.memory_seed is False

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_post_turn_none(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.post_turn is None


# ---------------------------------------------------------------------------
# K8. run() always synchronous; framework decides async wrapping           (§5b)
# ---------------------------------------------------------------------------

class TestK8RunAlwaysSynchronous:
    """K8: Each delegate's run() is always synchronous (no threading inside
    run); the framework (Ability.dispatch) decides async wrapping. The tool
    only declares ASYNC_CAPABLE=True."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_async_capable_true(self, name):
        ability = _get_ability(name)
        assert ability.ASYNC_CAPABLE is True

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_run_returns_result_synchronously(self, name):
        """run() returns the result directly (no thread spawned in run)."""
        import threading
        ability = _get_ability(name)
        spawned = []
        original_thread = threading.Thread

        def _track(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            spawned.append(t)
            return t

        with patch("services.message_processor.MessageProcessor.process",
                   return_value="synchronous answer"), \
             patch("threading.Thread", side_effect=_track):
            result = ability.run("user", {"goal": "g", "query": "g"}, None)

        # run() itself spawns no threads — async is the framework's job.
        assert spawned == []
        assert "synchronous answer" in str(result)


# ---------------------------------------------------------------------------
# K9. Delegate honors its wall-clock deadline                              (§5b/§4)
# ---------------------------------------------------------------------------

class TestK9WallClockDeadline:
    """K9: Each delegate passes a finite wall-clock deadline to process()."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_deadline_is_passed_to_process(self, name):
        before = time.time()
        captured = _captured_config(name)
        after = time.time()
        deadline = captured["kwargs"].get("deadline")
        if deadline is None and captured["args"]:
            # deadline may be passed positionally (3rd positional after
            # raw_input, config → metadata, deadline). Walk positional args.
            for a in captured["args"]:
                if isinstance(a, (int, float)):
                    deadline = a
                    break
        assert deadline is not None, "delegate must pass a wall-clock deadline"
        # The deadline must be in the future relative to dispatch time.
        assert deadline > before
        # And it must be a finite, bounded horizon (<= 1 day past now).
        assert deadline <= after + 86400

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_max_iterations_is_finite(self, name):
        cfg = _captured_config(name)["config"]
        assert cfg.max_iterations is not None
        assert cfg.max_iterations > 0


# ---------------------------------------------------------------------------
# K10. No subagent type registry / no make_subagent_config factory         (§5b/§7f)
# ---------------------------------------------------------------------------

class TestK10NoSubagentRegistry:
    """K10: There is no SUBAGENT_TYPES registry and no make_subagent_config
    factory — each delegate builds its own config inline."""

    def test_no_subagent_types_module(self):
        with pytest.raises(ImportError):
            from abilities.subagent import SUBAGENT_TYPES  # noqa: F401

    def test_no_make_subagent_config_in_configs(self):
        import configs.channels as ch
        assert not hasattr(ch, "make_subagent_config")

    def test_no_subagent_ability_module(self):
        with pytest.raises(ImportError):
            import abilities.subagent  # noqa: F401

    def test_no_subagent_processor_module(self):
        with pytest.raises(ImportError):
            import services.subagent_processor  # noqa: F401

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_delegate_config_channel_is_delegate_namespaced(self, name):
        """Each delegate config channel is 'delegate:<name>' and role=<name>,
        usage_class='subconscious' — proving per-tool inline configs, not a
        shared subagent type."""
        cfg = _captured_config(name)["config"]
        assert cfg.channel == f"delegate:{name}"
        assert cfg.role == name
        assert cfg.usage_class == "subconscious"


# ---------------------------------------------------------------------------
# Registration / surface sanity (AC-18: ≥4 delegates operational)
# ---------------------------------------------------------------------------

class TestDelegatesRegistered:
    """All 4 delegate tools must be registered and async-capable."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_delegate_registered(self, name):
        from abilities._registry import AbilityRegistry
        ability = AbilityRegistry.get(name)
        assert ability.NAME == name

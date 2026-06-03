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

The 2 delegate tools and their EXACT tool surfaces (§9a K3):
  - web_search  → always_available = ['search', 'read']
  - web_browse  → always_available = ['browser', 'read']

Spec references: §5b, §10f, §4.
"""

import time
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DELEGATE_NAMES = ("web_search", "web_browse")


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
# K6. Runs through normal MessageProcessor.process — no subclass           (§5b)
# ---------------------------------------------------------------------------

class TestK6RunsThroughNormalProcess:
    """K6: A delegate runs the normal MessageProcessor.process() — it builds a
    config and calls process(); there is no MessageProcessor subclass."""

    @pytest.mark.parametrize("name", _DELEGATE_NAMES)
    def test_passes_goal_as_raw_input(self, name):
        captured = _captured_config(name, params={"goal": "the goal", "query": "the goal"})
        assert captured["raw_input"] == "the goal"


# ---------------------------------------------------------------------------
# K8. run() always synchronous; framework decides async wrapping           (§5b)
# ---------------------------------------------------------------------------

class TestK8RunAlwaysSynchronous:
    """K8: Each delegate's run() is always synchronous (no threading inside
    run); the framework (Ability.dispatch) decides async wrapping. The tool
    only declares ASYNC_CAPABLE=True."""

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

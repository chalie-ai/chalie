# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guard for ``AbilityRegistry.build_tools`` (NOT a §9a blind-spec
file).

The ACT-loop refactor left ``build_tools`` as its T2 stub (``return []``)
while every caller was rewired to it. The flat loop therefore sent ZERO tools
to the model on every turn — find_tools / memory / code_eval / web_search and
every delegate were invisible, so no tool could ever be dispatched. Every
prior unit test mocked ``build_tools`` to ``[]``, so the regression was
invisible to the suite and only surfaced in a live e2e (the model replied
"I need to check what tools are available" and then stopped).

These tests exercise the REAL implementation against the REAL registry so the
tool surface can never silently collapse again. Spec: message-processing.md
§6 (build_tools resolves always_available + discovered tools; act_summary is
injected here and popped by Ability.dispatch).
"""

import pytest

from abilities._registry import AbilityRegistry
from configs.channels import DEFAULT_ALWAYS_AVAILABLE, make_user_config
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _make_mp(config, discovered=None):
    mp = object.__new__(MessageProcessor)
    mp.config = config
    mp._discovered_tools = list(discovered or [])
    return mp


def test_user_config_surfaces_always_available_tools():
    """The user channel must pin its always_available tier — never empty."""
    mp = _make_mp(make_user_config({"channel": "user"}))
    tools = AbilityRegistry.build_tools(mp)
    names = {t["name"] for t in tools}
    assert names == set(DEFAULT_ALWAYS_AVAILABLE)
    assert names == {"find_skills", "find_tools", "memory"}


def test_act_summary_injected_as_required_on_every_tool():
    """Every surfaced tool gets act_summary as a required property (spec §6)."""
    mp = _make_mp(make_user_config({"channel": "user"}))
    tools = AbilityRegistry.build_tools(mp)
    assert tools, "build_tools returned no tools for the user config"
    for t in tools:
        props = t["input_schema"]["properties"]
        assert "act_summary" in props
        assert "act_summary" in t["input_schema"]["required"]


def test_discovered_tools_are_appended_and_deduped():
    """find_tools-discovered abilities join the surface; dupes keep first-seen."""
    code_eval = AbilityRegistry.get("code_eval")
    discovered = [{
        "name": code_eval.NAME,
        "description": code_eval.SUMMARY,
        "input_schema": code_eval.INPUT_SCHEMA,
    }, {
        # duplicate of an always_available tool must NOT double-appear
        "name": "memory",
        "description": "dup",
        "input_schema": {"type": "object", "properties": {}},
    }]
    mp = _make_mp(make_user_config({"channel": "user"}), discovered=discovered)
    names = [t["name"] for t in AbilityRegistry.build_tools(mp)]
    assert "code_eval" in names
    assert names.count("memory") == 1


def test_build_tools_does_not_mutate_ability_classvar():
    """act_summary injection must deep-copy — never pollute the ClassVar."""
    code_eval = AbilityRegistry.get("code_eval")
    discovered = [{
        "name": code_eval.NAME,
        "description": code_eval.SUMMARY,
        "input_schema": code_eval.INPUT_SCHEMA,
    }]
    mp = _make_mp(make_user_config({"channel": "user"}), discovered=discovered)
    AbilityRegistry.build_tools(mp)
    assert "act_summary" not in code_eval.INPUT_SCHEMA.get("properties", {})


def test_empty_always_available_yields_empty_surface():
    """Compaction/encoder paths bind an empty tier → empty tool list (no crash)."""
    class _Cfg:
        always_available = []
    mp = _make_mp(_Cfg())
    assert AbilityRegistry.build_tools(mp) == []


def test_no_config_bound_is_safe():
    """A processor without a config must not raise — returns []."""
    mp = object.__new__(MessageProcessor)
    mp._discovered_tools = []
    # no .config attribute set
    assert AbilityRegistry.build_tools(mp) == []

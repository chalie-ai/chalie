# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guard for ``AbilityRegistry.build_tools`` (NOT a §9a blind-spec
file).

The ACT-loop refactor once left ``build_tools`` as its T2 stub (``return []``)
while every caller was rewired to it — the flat loop sent ZERO tools to the
model on every turn, and every prior unit test mocked ``build_tools`` to ``[]``
so the suite never saw it (mem 07c8c134). These tests exercise the REAL
implementation against the REAL registry so the surface can never silently
collapse again.

Post-redesign ``build_tools`` resolves ``mp.active_tools`` — the live list of
tool NAMES seeded with ``config.always_available`` by ``_setup`` and appended to
by ``find_tools``. The collapse-guard now has two halves: the seed (``_setup``)
and the resolve (``build_tools``); both are pinned below.
"""

import pytest

from abilities._registry import AbilityRegistry
from configs.channels import DEFAULT_ALWAYS_AVAILABLE, UserConfig
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _make_mp(active, config=None):
    """A flat MP carrying a seeded ACTIVE_TOOLS list (names) — no full __init__."""
    mp = object.__new__(MessageProcessor)
    if config is not None:
        mp.config = config
    mp._active_tools = list(active)
    return mp


def test_setup_seeds_active_tools_from_always_available(monkeypatch):
    """THE seed half: _setup must initialise ACTIVE_TOOLS = config.always_available,
    or the always_available tier never reaches the model (mem 07c8c134, relocated)."""
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "", None)
    mp.config = UserConfig({"channel": "user"})
    mp.uid = None
    # Stub the heavy bits _setup runs AFTER the seed (DB write, thinking gate, seed-0).
    monkeypatch.setattr("services.transcript_service.write_input_row", lambda *a, **k: 1)
    monkeypatch.setattr(mp, "_run_thinking_gate", lambda: None)
    monkeypatch.setattr(mp, "_seed_turn_zero", lambda: None)

    mp._setup()

    assert set(mp.active_tools) == set(DEFAULT_ALWAYS_AVAILABLE)


def test_active_tools_resolve_to_always_available_surface():
    """The resolve half: active_tools = always_available → those tools reach the model."""
    mp = _make_mp(list(DEFAULT_ALWAYS_AVAILABLE), config=UserConfig({"channel": "user"}))
    names = {t["name"] for t in AbilityRegistry.build_tools(mp)}
    assert names == set(DEFAULT_ALWAYS_AVAILABLE)
    assert names == {"find_skills", "find_tools", "memory"}


def test_act_summary_injected_as_required_on_every_tool():
    """Every surfaced tool gets act_summary as a required property (spec §6)."""
    tools = AbilityRegistry.build_tools(_make_mp(list(DEFAULT_ALWAYS_AVAILABLE)))
    assert tools, "build_tools returned no tools for a seeded active_tools list"
    for t in tools:
        props = t["input_schema"]["properties"]
        assert "act_summary" in props
        assert "act_summary" in t["input_schema"]["required"]


def test_find_tools_appended_names_are_resolved_and_deduped():
    """find_tools-appended names join the surface; dupes keep first-seen."""
    names = [t["name"] for t in AbilityRegistry.build_tools(_make_mp(["memory", "code_eval", "memory"]))]
    assert "code_eval" in names
    assert names.count("memory") == 1


def test_build_tools_does_not_mutate_ability_classvar():
    """act_summary injection must deep-copy — never pollute the ClassVar."""
    code_eval = AbilityRegistry.get("code_eval")
    AbilityRegistry.build_tools(_make_mp(["code_eval"]))
    assert "act_summary" not in code_eval.INPUT_SCHEMA.get("properties", {})


def test_unknown_name_is_skipped_not_fatal():
    """A name with no registered ability is logged and skipped, never raises."""
    names = {t["name"] for t in AbilityRegistry.build_tools(_make_mp(["memory", "no_such_ability_xyz"]))}
    assert names == {"memory"}


def test_empty_active_tools_yields_empty_surface():
    """Compaction / encoder channels seed an empty active_tools → empty surface."""
    assert AbilityRegistry.build_tools(_make_mp([])) == []


def test_no_active_tools_attr_is_safe():
    """A processor with no active_tools bound must not raise — returns []."""
    mp = object.__new__(MessageProcessor)  # no _active_tools, no config
    assert AbilityRegistry.build_tools(mp) == []

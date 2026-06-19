# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



from typing import Optional, cast

import pytest

from abilities._registry import AbilityRegistry
from configs.channels import DEFAULT_ALWAYS_AVAILABLE, UserConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


def _make_mp(active: list[str], config: Optional[ProcessorConfig] = None) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    if config is not None:
        mp.config = config
    mp._active_tools = list(active)
    return mp


def test_setup_seeds_active_tools_from_always_available(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_active_tools_resolve_to_always_available_surface() -> None:
    """The resolve half: active_tools = always_available → those tools reach the model."""
    mp = _make_mp(list(DEFAULT_ALWAYS_AVAILABLE), config=UserConfig({"channel": "user"}))
    names = {t["name"] for t in AbilityRegistry.build_tools(mp)}
    assert names == set(DEFAULT_ALWAYS_AVAILABLE)
    assert names == {"find_skills", "find_tools", "memory"}


def test_act_summary_injected_as_required_on_every_tool() -> None:
    """Every surfaced tool gets act_summary as a required property (spec §6)."""
    tools = AbilityRegistry.build_tools(_make_mp(list(DEFAULT_ALWAYS_AVAILABLE)))
    assert tools, "build_tools returned no tools for a seeded active_tools list"
    for t in tools:
        props = cast(dict[str, object], t["input_schema"])["properties"]
        assert "act_summary" in cast(dict[str, object], props)
        assert "act_summary" in cast(list[object], cast(dict[str, object], t["input_schema"])["required"])


def test_find_tools_appended_names_are_resolved_and_deduped() -> None:
    """find_tools-appended names join the surface; dupes keep first-seen."""
    names = [t["name"] for t in AbilityRegistry.build_tools(_make_mp(["memory", "code_eval", "memory"]))]
    assert "code_eval" in names
    assert names.count("memory") == 1


def test_build_tools_does_not_mutate_ability_classvar() -> None:
    """act_summary injection must deep-copy — never pollute the declared params."""
    code_eval = AbilityRegistry.get("code_eval")
    AbilityRegistry.build_tools(_make_mp(["code_eval"]))
    assert "act_summary" not in cast(dict[str, object], code_eval.get_parameters().get("properties", {}))


def test_unknown_name_is_skipped_not_fatal() -> None:
    """A name with no registered ability is logged and skipped, never raises."""
    names = {t["name"] for t in AbilityRegistry.build_tools(_make_mp(["memory", "no_such_ability_xyz"]))}
    assert names == {"memory"}


def test_empty_active_tools_yields_empty_surface() -> None:
    """Compaction / encoder channels seed an empty active_tools → empty surface."""
    assert AbilityRegistry.build_tools(_make_mp([])) == []


def test_no_active_tools_attr_is_safe() -> None:
    """A processor with no active_tools bound must not raise — returns []."""
    mp = object.__new__(MessageProcessor)  # no _active_tools, no config
    assert AbilityRegistry.build_tools(mp) == []

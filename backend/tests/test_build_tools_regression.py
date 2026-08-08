# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from typing import cast

import pytest

from abilities._registry import AbilityRegistry
from configs.channels import DEFAULT_ALWAYS_AVAILABLE, UserConfig
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


def _make_mp(active: list[str], config: ProcessorConfig | None = None) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    # Schema injection reads mp.config directly (typed contract) — every bound
    # processor has one, so the fake must too.
    mp.config = config if config is not None else UserConfig({"channel": "user"})
    mp.active_tools = list(active)
    return mp


def test_active_tools_resolve_to_always_available_surface() -> None:
    """The resolve half: active_tools = always_available → those tools reach the model."""
    mp = _make_mp(list(DEFAULT_ALWAYS_AVAILABLE), config=UserConfig({"channel": "user"}))
    names = {cast(str, t["name"]) for t in AbilityRegistry.build_tools(mp)}
    assert names == set(DEFAULT_ALWAYS_AVAILABLE)
    assert names == {"find_skills", "find_tools", "mcp_manager", "recall"}


def test_act_summary_injected_as_required_on_every_tool() -> None:
    """Every surfaced tool gets act_summary as a required property."""
    tools = AbilityRegistry.build_tools(_make_mp(list(DEFAULT_ALWAYS_AVAILABLE)))
    assert tools, "build_tools returned no tools for a seeded active_tools list"
    for t in tools:
        props = cast(dict[str, object], t["input_schema"])["properties"]
        assert "act_summary" in cast(dict[str, object], props)
        assert "act_summary" in cast(list[object], cast(dict[str, object], t["input_schema"])["required"])


def test_find_tools_appended_names_are_resolved_and_deduped() -> None:
    """find_tools-appended names join the surface; dupes keep first-seen."""
    names = [cast(str, t["name"]) for t in AbilityRegistry.build_tools(_make_mp(["recall", "weather", "recall"]))]
    assert "weather" in names
    assert names.count("recall") == 1


def test_build_tools_does_not_mutate_ability_classvar() -> None:
    """act_summary injection must deep-copy — never pollute the declared params."""
    weather = AbilityRegistry.get("weather")
    AbilityRegistry.build_tools(_make_mp(["weather"]))
    assert "act_summary" not in cast(dict[str, object], weather.get_parameters().get("properties", {}))


def test_unknown_name_is_skipped_not_fatal() -> None:
    """A name with no registered ability is logged and skipped, never raises."""
    names = {cast(str, t["name"]) for t in AbilityRegistry.build_tools(_make_mp(["recall", "no_such_ability_xyz"]))}
    assert names == {"recall"}


def test_empty_active_tools_yields_empty_surface() -> None:
    """Compaction / encoder channels seed an empty active_tools → empty surface."""
    assert AbilityRegistry.build_tools(_make_mp([])) == []


def test_no_active_tools_attr_is_safe() -> None:
    """A processor with no active_tools bound must not raise — returns []."""
    mp = object.__new__(MessageProcessor)  # no active_tools, no config
    assert AbilityRegistry.build_tools(mp) == []

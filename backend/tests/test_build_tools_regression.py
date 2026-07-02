# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import sqlite3
from typing import cast
from unittest.mock import patch

import pytest

from abilities._registry import AbilityRegistry
from configs.channels import DEFAULT_ALWAYS_AVAILABLE, UserConfig
from services.database_service import get_shared_db_service
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.provider_api import ProviderApiRequest, ProviderApiResponse
from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


class _RecordingProvider:
    """The sanctioned external-boundary stand-in: captures each send and returns a
    one-shot ``NOTHING`` with a token estimate of 1 so the pre-flight cap check
    always passes. The only thing _setup's thinking gate / seed-0 cannot reach for
    real in a test — everything else (db, config, transcript writes) is real."""

    def __init__(self) -> None:
        self.sends: list[ProviderApiRequest] = []

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        self.sends.append(dto)
        return ProviderApiResponse(text="NOTHING", model="recorder", tool_calls=None)


def _make_mp(active: list[str], config: ProcessorConfig | None = None) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    if config is not None:
        mp.config = config
    mp._active_tools = list(active)
    return mp


def test_setup_seeds_active_tools_and_opens_a_turn(db: sqlite3.Connection) -> None:
    """The real production pre-loop on the user channel: ``_setup`` seeds
    active_tools from the channel's always_available tier AND opens the turn by
    writing a turn_id-stamped input row (atomic COALESCE allocation, read back).

    Drives the real ``MessageProcessor._setup`` against the real db with a real
    ``UserConfig``; the only stand-in is ``Providers._resolve`` — the LLM boundary
    the thinking gate and seed-0 fire through. Zero internal mocks."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    config = UserConfig({"channel": "user"})
    recorder = _RecordingProvider()
    with patch("services.providers.Providers._resolve", return_value=recorder):
        mp = MessageProcessor(config, raw_input="hello")
        mp._setup()

    # active_tools seeded from the channel's always_available tier.
    assert set(mp.active_tools) == set(DEFAULT_ALWAYS_AVAILABLE)

    # The turn was really opened: the input row exists on the user channel tagged
    # with the turn_id _setup resolved — proof the atomic allocation wired through.
    assert cast(int, mp.turn_id) >= 1
    with get_shared_db_service().connection() as conn:
        row = conn.execute(
            "SELECT turn_id, role FROM transcript WHERE id = ?", (mp.uid,)
        ).fetchone()
    assert row is not None
    assert row[0] == mp.turn_id
    assert row[1] == "user"


def test_active_tools_resolve_to_always_available_surface() -> None:
    """The resolve half: active_tools = always_available → those tools reach the model."""
    mp = _make_mp(list(DEFAULT_ALWAYS_AVAILABLE), config=UserConfig({"channel": "user"}))
    names = {cast(str, t["name"]) for t in AbilityRegistry.build_tools(mp)}
    assert names == set(DEFAULT_ALWAYS_AVAILABLE)
    assert names == {"find_skills", "find_tools", "memory"}


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
    names = [cast(str, t["name"]) for t in AbilityRegistry.build_tools(_make_mp(["memory", "code_eval", "memory"]))]
    assert "code_eval" in names
    assert names.count("memory") == 1


def test_build_tools_does_not_mutate_ability_classvar() -> None:
    """act_summary injection must deep-copy — never pollute the declared params."""
    code_eval = AbilityRegistry.get("code_eval")
    AbilityRegistry.build_tools(_make_mp(["code_eval"]))
    assert "act_summary" not in cast(dict[str, object], code_eval.get_parameters().get("properties", {}))


def test_unknown_name_is_skipped_not_fatal() -> None:
    """A name with no registered ability is logged and skipped, never raises."""
    names = {cast(str, t["name"]) for t in AbilityRegistry.build_tools(_make_mp(["memory", "no_such_ability_xyz"]))}
    assert names == {"memory"}


def test_empty_active_tools_yields_empty_surface() -> None:
    """Compaction / encoder channels seed an empty active_tools → empty surface."""
    assert AbilityRegistry.build_tools(_make_mp([])) == []


def test_no_active_tools_attr_is_safe() -> None:
    """A processor with no active_tools bound must not raise — returns []."""
    mp = object.__new__(MessageProcessor)  # no _active_tools, no config
    assert AbilityRegistry.build_tools(mp) == []

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ubiquiti-ability-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975.

Pins TKT-890's schema-honesty contract: the prose-DSL params are gone, the
action enum exactly matches ACTION_HANDLERS, and the new action ids are
correctly seeded in policy_defaults.json.
"""

import json

import pytest

from configs.channels import UserConfig
from services.file_mapper_service import FileMapperService
from tests._tool_result_harness import MP, allow_policy, seed_transcript

pytestmark = pytest.mark.unit

# Read actions ship ``allow``; everything that touches the network ships ``ask``
# on chat. Flipping the ask rows to allow lets the gate pass through to the real
# run() so the base's not-connected path actually executes (exactly what a user
# does when they pick "always allow"). No mock — the production policy table
# driving the production gate.
_ASK_ACTIONS = (
    "block_client",
    "unblock_client",
    "disconnect_client",
    "restart_device",
    "locate_device",
    "stop_locate_device",
    "power_cycle_port",
    "update_wifi",
    "create_port_forward",
    "update_port_forward",
    "update_traffic_rule",
    "authorize_guest",
)
_ALLOW_ACTIONS = (
    "list_devices",
    "list_clients",
    "device_status",
    "site_health",
    "list_wifi",
    "list_port_forwards",
    "list_traffic_rules",
)


def _allow_ubiquiti_actions(db, channel: str = "chat") -> None:
    for action in (*_ALLOW_ACTIONS, *_ASK_ACTIONS):
        allow_policy(db, f"ubiquiti.{action}", channel=channel)


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and every ubiquiti action flipped to ``allow``
    in the real policy table so the gate passes through to the production run()."""
    _allow_ubiquiti_actions(db)
    return MP(seed_transcript(db, content="is the office AP up?"), UserConfig({}))


# ── Schema honesty: prose-DSL is gone, enum == ACTION_HANDLERS ──────────────────


def test_prose_dsl_params_absent_from_parameters():
    """``command`` and ``sub_action`` — the undocumented prose-DSL the model had
    to guess — are REMOVED from the schema. ``mac`` is gone too: addressing is the
    single ``target`` param."""
    from abilities._registry import AbilityRegistry

    props = AbilityRegistry.get("ubiquiti").get_parameters()["properties"]
    assert "command" not in props
    assert "sub_action" not in props
    assert "mac" not in props
    assert "target" in props


def test_action_enum_matches_action_handlers_exactly():
    """Every action the schema advertises really dispatches — the enum is EXACTLY
    the ACTION_HANDLERS keys, so a weak model cannot pick an action that has no
    handler."""
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("ubiquiti")
    enum = ability.get_parameters()["properties"]["action"]["enum"]
    assert set(enum) == set(ability.ACTION_HANDLERS)
    # The flat real operations the audit demanded are all present.
    for action in (
        "list_devices", "list_clients", "device_status", "site_health",
        "block_client", "unblock_client", "disconnect_client",
        "restart_device", "locate_device", "authorize_guest",
    ):
        assert action in enum


# ── Policy: the new action ids resolve in the seeded defaults ───────────────────


def test_policy_defaults_seed_the_new_action_ids():
    """The static policy_defaults.json carries a row for EVERY new action on the
    chat channel — read ops ``allow``, mutating / outward-facing ops ``ask`` —
    and ``deny`` on the agent channels for the network-touching ops. This mirrors
    how the seed drives the real PolicyManager gate; a missing row would lazily
    default a network action to ``ask``→deny."""
    with open(FileMapperService.get_policy_defaults_path()) as fh:
        seed = json.load(fh)

    chat = {
        r["permission"]: r["setting"]
        for r in seed
        if r["channel"] == "chat" and r["permission"].startswith("ubiquiti.")
    }
    for action in _ALLOW_ACTIONS:
        assert chat.get(f"ubiquiti.{action}") == "allow", action
    for action in _ASK_ACTIONS:
        assert chat.get(f"ubiquiti.{action}") == "ask", action

    # The old prose-DSL action ids must be GONE from the seed.
    all_perms = {r["permission"] for r in seed}
    for legacy in (
        "ubiquiti.control_client", "ubiquiti.control_device",
        "ubiquiti.manage_wlan", "ubiquiti.manage_port_forward",
        "ubiquiti.manage_traffic_rule", "ubiquiti.get_info",
    ):
        assert legacy not in all_perms, legacy

    # Network-touching ops are denied on the agent channels.
    for ch in ("external_agent", "subconscious"):
        rows = {
            r["permission"]: r["setting"]
            for r in seed
            if r["channel"] == ch and r["permission"].startswith("ubiquiti.")
        }
        assert rows.get("ubiquiti.block_client") == "deny", ch
        assert rows.get("ubiquiti.restart_device") == "deny", ch
        assert rows.get("ubiquiti.list_devices") == "allow", ch

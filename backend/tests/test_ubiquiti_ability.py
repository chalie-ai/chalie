# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ubiquiti-ability-specific business-logic tests migrated from the removed
per-ability conformance file.

Pins the schema-honesty contract: the prose-DSL params are gone, the
action enum exactly matches ACTION_HANDLERS, and the new action ids are
correctly seeded in policy_defaults.json.
"""

import json
from typing import cast

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

# Read actions ship ``allow``; everything that touches the network ships ``ask``
# on chat.
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


# ── Schema honesty: prose-DSL is gone, enum == ACTION_HANDLERS ──────────────────


def test_prose_dsl_params_absent_from_parameters() -> None:
    from abilities._registry import AbilityRegistry

    props = cast("dict[str, object]", AbilityRegistry.get("ubiquiti").get_parameters()["properties"])
    assert "command" not in props
    assert "sub_action" not in props
    assert "mac" not in props
    assert "target" in props


def test_action_enum_matches_action_handlers_exactly() -> None:
    from abilities._registry import AbilityRegistry
    from abilities.ubiquiti import UbiquitiAbility

    ability = AbilityRegistry.get("ubiquiti")
    enum = cast("list[str]", cast("dict[str, object]", cast("dict[str, object]", ability.get_parameters()["properties"])["action"])["enum"])
    assert set(enum) == set(cast(UbiquitiAbility, ability).ACTION_HANDLERS)
    # The flat real operations the audit demanded are all present.
    for action in (
        "list_devices", "list_clients", "device_status", "site_health",
        "block_client", "unblock_client", "disconnect_client",
        "restart_device", "locate_device", "authorize_guest",
    ):
        assert action in enum


# ── Policy: the new action ids resolve in the seeded defaults ───────────────────


def test_policy_defaults_seed_the_new_action_ids() -> None:
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

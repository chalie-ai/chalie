# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the ubiquiti tool's ToolResult contract (TKT-890).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``UbiquitiAbility`` (now a ``CapabilityAbility`` subclass), the real
``PolicyManager.wrap`` gate (with the real ``policy`` table from the ``db``
fixture), the real ``UbiquitiAbility.run`` and the real ``ActTrail`` write.

The ubiquiti (UniFi) capability is NOT connected in the test environment (no
controller credentials) — that is the production reality this suite leans on.
Every dispatch either:

* trips the dispatcher's ACTION_REQUIRED pre-gate (``code=missing-params`` /
  ``code=unknown-action`` with a ``valid:`` ladder), which fires BEFORE the
  policy gate and before the capability is ever loaded; or
* reaches the base's ``code=not-connected`` surface (capability loaded, refuses
  because no controller credentials).

What TKT-890 changes, exercised end to end:

* **Schema collapse:** the undocumented prose-DSL params ``command`` /
  ``sub_action`` are GONE from ``get_parameters()``. The model addresses real
  operations through a flat ``action`` enum + ``target`` (device/client) + a few
  typed params — no free-text command string anywhere.
* **Foundation:** ``UbiquitiAbility`` is a ``CapabilityAbility`` subclass — the
  not-connected / unknown-action / handler-dispatch flow comes from the base.
* **Contract:** every action returns a ``ToolResult``; errors carry stable kebab
  codes (NOT the ``code="error"`` mechanical-wrap placeholder) + hints + a
  ``valid:`` ladder where applicable.
* **Schema honesty:** the advertised action enum is EXACTLY the keys of
  ACTION_HANDLERS, so every action the model can pick really dispatches.
* **Policy:** the new action ids resolve in the seeded ``policy`` table — read
  ops ``allow`` on chat, mutating / outward-facing ops ``ask`` on chat and
  ``deny`` on the agent channels.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
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


# ── Foundation: UbiquitiAbility is a CapabilityAbility ──────────────────────────


def test_ubiquiti_is_a_capability_ability():
    """The ability is migrated onto the shared ``CapabilityAbility`` base — the
    hand-rolled capability-delegation block is gone."""
    from abilities._capability import CapabilityAbility
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("ubiquiti")
    assert isinstance(ability, CapabilityAbility)
    assert ability.CAPABILITY_KEY == "ubiquiti"


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


# ── Error ladder: unknown action (ACTION_REQUIRED pre-gate) ─────────────────────


def test_unknown_action_lists_real_actions_in_valid(db, chat_mp):
    """An unknown action is pre-gated into ONE ``code=unknown-action`` error whose
    ``valid:`` line names the REAL actions (NOT the placeholder)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "set-port-profile", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=unknown-action" in out
    assert "code=error]" not in out
    assert "valid:" in out
    for action in ("list_devices", "block_client", "restart_device"):
        assert action in out


# ── Error ladder: missing required params per action family (pre-gate) ──────────


def test_block_client_without_target_reports_missing_params(db, chat_mp):
    """``block_client`` with no ``target`` is pre-gated into a single
    ``code=missing-params`` error naming ``target`` — BEFORE the policy gate and
    before the capability is ever loaded."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "block_client", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "target" in out


def test_restart_device_without_target_reports_missing_params(db, chat_mp):
    """``restart_device`` with no ``target`` is pre-gated into
    ``code=missing-params`` naming ``target``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "restart_device", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "target" in out


def test_power_cycle_port_without_port_idx_reports_missing_params(db, chat_mp):
    """``power_cycle_port`` needs ``target`` + ``port_idx``; with only a target it
    is pre-gated into ``code=missing-params`` naming ``port_idx``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti",
        {"action": "power_cycle_port", "target": "aa:bb:cc:dd:ee:ff", "act_summary": "x"},
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "port_idx" in out


def test_update_wifi_without_wlan_id_reports_missing_params(db, chat_mp):
    """``update_wifi`` needs ``wlan_id`` + ``updates``; with neither it is
    pre-gated into ``code=missing-params`` naming both."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "update_wifi", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "wlan_id" in out
    assert "updates" in out


def test_update_port_forward_without_rule_id_reports_missing_params(db, chat_mp):
    """``update_port_forward`` needs ``rule_id`` + ``updates``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "update_port_forward", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "rule_id" in out


def test_create_port_forward_without_rule_reports_missing_params(db, chat_mp):
    """``create_port_forward`` needs a ``rule`` definition."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "create_port_forward", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "rule" in out


def test_authorize_guest_without_target_reports_missing_params(db, chat_mp):
    """``authorize_guest`` needs ``target``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "authorize_guest", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=missing-params" in out
    assert "target" in out


# ── Not-connected surface per action family (capability loaded, refuses) ────────


def test_list_devices_not_connected_errors_with_hint(db, chat_mp):
    """``list_devices`` (a valid read action needing no params) passes the
    pre-gate, loads the capability, and surfaces the base's stable
    ``code=not-connected`` with a hint naming the UniFi integration — NOT the
    legacy JSON ``{status: error}`` body and NOT the ``code="error"`` placeholder."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "list_devices", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=not-connected" in out
    assert "code=error]" not in out
    assert "not connected" in out.lower()
    assert "hint:" in out
    assert "unifi" in out.lower()


def test_list_clients_not_connected_errors_cleanly(db, chat_mp):
    """``list_clients`` passes the pre-gate and surfaces ``code=not-connected``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "list_clients", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_site_health_not_connected_errors_cleanly(db, chat_mp):
    """``site_health`` needs no params, passes the pre-gate, surfaces
    ``code=not-connected``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "site_health", "act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_device_status_with_target_not_connected_errors_cleanly(db, chat_mp):
    """``device_status`` with a target passes the pre-gate and surfaces
    ``code=not-connected`` — the addressing param is honoured, never a prose
    command string."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti",
        {"action": "device_status", "target": "aa:bb:cc:dd:ee:ff", "act_summary": "x"},
    )

    assert "[ubiquiti(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_block_client_with_target_not_connected_errors_cleanly(db, chat_mp):
    """A well-formed ``block_client`` (target supplied) passes validation and the
    pre-gate, then surfaces the base's ``code=not-connected`` — proving the action
    reaches the connected gate without any command-string DSL."""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti",
        {"action": "block_client", "target": "aa:bb:cc:dd:ee:ff", "act_summary": "x"},
    )

    assert "[ubiquiti(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_no_action_is_pregated_to_unknown_action(db, chat_mp):
    """With no ``action`` at all, the dispatcher's ACTION_REQUIRED pre-gate (keyed
    on the empty string, which is not in the map) returns ``code=unknown-action``
    with the real-action ``valid:`` ladder — BEFORE run() and before the
    capability is ever loaded. (The DEFAULT_ACTION only applies once an action has
    cleared the pre-gate; an action-less call never reaches run(), exactly as
    calendar and email behave.)"""
    out = ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"act_summary": "x"}
    )

    assert "[ubiquiti(status=error, code=unknown-action" in out
    assert "code=error]" not in out
    assert "valid:" in out
    assert "list_devices" in out


# ── Act-trail records the rendered envelope ─────────────────────────────────────


def test_act_trail_records_the_envelope(db, chat_mp):
    """The act-trail records the same non-empty ubiquiti envelope against the
    transcript anchor — the cross-step write really lands."""
    ToolDispatcher(chat_mp).dispatch(
        "ubiquiti", {"action": "list_devices", "act_summary": "x"}
    )
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("[ubiquiti(status=" in row["result"] for row in trail)


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

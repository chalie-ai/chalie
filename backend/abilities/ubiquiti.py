"""UbiquitiAbility — monitor and control a UniFi network via the Ubiquiti capability.

Delegates to ``UbiquitiCapability``'s REST handlers via the shared
:class:`CapabilityAbility` base, which owns capability loading, the
not-connected / unknown-action / handler-unavailable errors, handler dispatch,
and result wrapping. This ability supplies its metadata, the flat-action wiring,
the ACTION_REQUIRED pre-gate map, and the model→capability param translation.

**The schema is the point of this tool.** The old surface was nine coarse
actions (``control_client`` / ``manage_wlan`` / …) plus an undocumented prose-DSL
the model had to guess: a free-text ``command`` (``"block-sta"``, ``"set-locate"``)
and a ``sub_action`` (``"list"`` / ``"update"`` / ``"create"``). A weak local
model could not reliably produce those strings, which defeats the entire purpose
of a tool meant for one-shot home-network questions ("is the office AP up?").

This rewrite replaces that with a FLAT action enum naming every real operation
(``block_client``, ``restart_device``, ``list_wifi`` …), a single ``target``
addressing param (device/client MAC), and a few typed optional params. There is
no ``command`` and no ``sub_action`` anywhere — the capability's coarse handler +
sub-command pair is an internal translation detail (:data:`_ACTION_SPEC`) the
model never sees. Each flat action is invocable correctly from its one-line
description alone.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``; the
dispatcher renders the wire envelope. This ability never formats an envelope and
never imports the skill-tag formatter.
"""

import logging
from typing import ClassVar, NamedTuple, cast

from abilities._capability import CapabilityAbility
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.capability_params_bag import CapabilityParamsBag
from configs.enums.ability_category import AbilityCategory

logger = logging.getLogger(__name__)
LOG_PREFIX = "[UBIQUITI ABILITY]"


class _Spec(NamedTuple):
    """The internal translation from one flat model-facing action to the coarse
    capability handler + the fixed sub-command params that handler expects.
    ``inject`` is the fixed prose-DSL the OLD schema forced the model to guess —
    now supplied by the ability so the model never sees it.
    """

    handler: str
    inject: dict[str, object]


# The single source of truth: flat action → (capability handler, injected DSL).
# The capability still speaks the coarse handler + command/sub_action dialect;
# this table is the one place that dialect is spoken, keyed off a flat verb.
_ACTION_SPEC: dict[str, _Spec] = {
    # ── Read ────────────────────────────────────────────────────────────────
    "list_devices": _Spec("list_devices", {}),
    "list_clients": _Spec("list_clients", {}),
    "device_status": _Spec("get_info", {}),  # target → get_info(target=<mac>)
    "site_health": _Spec("get_info", {"target": "health"}),
    "list_wifi": _Spec("manage_wlan", {"sub_action": "list"}),
    "list_port_forwards": _Spec("manage_port_forward", {"sub_action": "list"}),
    "list_traffic_rules": _Spec("manage_traffic_rule", {"sub_action": "list"}),
    # ── Client control ──────────────────────────────────────────────────────
    "block_client": _Spec("control_client", {"command": "block-sta"}),
    "unblock_client": _Spec("control_client", {"command": "unblock-sta"}),
    "disconnect_client": _Spec("control_client", {"command": "kick-sta"}),
    # ── Device control ──────────────────────────────────────────────────────
    "restart_device": _Spec("control_device", {"command": "restart"}),
    "locate_device": _Spec("control_device", {"command": "set-locate"}),
    "stop_locate_device": _Spec("control_device", {"command": "unset-locate"}),
    "power_cycle_port": _Spec("control_device", {"command": "power-cycle"}),
    # ── WiFi / forwarding / traffic writes ──────────────────────────────────
    "update_wifi": _Spec("manage_wlan", {"sub_action": "update"}),
    "create_port_forward": _Spec("manage_port_forward", {"sub_action": "create"}),
    "update_port_forward": _Spec("manage_port_forward", {"sub_action": "update"}),
    "update_traffic_rule": _Spec("manage_traffic_rule", {"sub_action": "update"}),
    # ── Guest ───────────────────────────────────────────────────────────────
    "authorize_guest": _Spec("authorize_guest", {}),
}

# Actions whose ``target`` addresses a device/client and is translated to the
# capability's ``mac`` param. ``device_status`` accepts a MAC too (omitted →
# the capability defaults to the site-health overview, which is what site_health
# does explicitly), so it is NOT required-addressed.
_TARGET_TO_MAC = (
    "device_status",
    "block_client", "unblock_client", "disconnect_client",
    "restart_device", "locate_device", "stop_locate_device", "power_cycle_port",
    "authorize_guest",
)


class UbiquitiAbility(CapabilityAbility):
    CAPABILITY_KEY: ClassVar[str] = "ubiquiti"
    DEFAULT_ACTION: ClassVar[str] = "list_devices"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the UniFi integration in the Brain dashboard."
    )

    # Maps the model-facing flat action onto the capability's coarse handler
    # name. The base uses it to look up the handler and to build the valid=
    # ladder for unknown-action errors, so the enum below is derived from it.
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        action: spec.handler for action, spec in _ACTION_SPEC.items()
    }
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("unifi", "network control", "router")

    # Pre-gated by the dispatcher BEFORE run() (and before the policy gate): a
    # known action missing a required param yields one code=missing-params error
    # naming all missing params; an unknown action yields one code=unknown-action
    # whose valid= names every real action. Read/list actions have no hard
    # requirement; device_status defaults to the site overview when target is
    # omitted, so it is not required-addressed.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list_devices": (),
        "list_clients": (),
        "device_status": (),
        "site_health": (),
        "list_wifi": (),
        "list_port_forwards": (),
        "list_traffic_rules": (),
        "block_client": (Keys.target,),
        "unblock_client": (Keys.target,),
        "disconnect_client": (Keys.target,),
        "restart_device": (Keys.target,),
        "locate_device": (Keys.target,),
        "stop_locate_device": (Keys.target,),
        "power_cycle_port": (Keys.target, Keys.port_idx),
        "update_wifi": (Keys.wlan_id, Keys.updates),
        "create_port_forward": (Keys.rule,),
        "update_port_forward": (Keys.rule_id, Keys.updates),
        "update_traffic_rule": (Keys.rule_id, Keys.updates),
        "authorize_guest": (Keys.target,),
    }
    NAME: ClassVar[str] = "ubiquiti"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.SMART_HOME

    def get_summary(self) -> str:
        return (
            "Monitor and control a UniFi (Ubiquiti) network: list devices and "
            "connected clients, check device or site health, block / unblock / "
            "disconnect a client, restart or locate a device, list and update WiFi "
            "networks, port-forward and traffic rules, and authorize a guest device. "
            "Available when the user asks about their home or office network. "
            "To control or update a device, client, or rule you need its MAC or id: "
            "run the matching list action first (list_devices, list_clients, "
            "list_wifi, list_port_forwards, list_traffic_rules) to get a valid "
            "identifier."
        )

    def get_examples(self) -> list[str]:
        return [
            "what UniFi devices are on my network",
            "who's connected to the WiFi right now",
            "is the office access point up",
            "block the client aa:bb:cc:dd:ee:ff",
            "restart the living room access point",
            "show my WiFi networks",
            "is my network healthy",
            "authorize this guest device for 2 hours",
        ]

    def get_search_tooltip(self) -> str:
        return "UniFi network control"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(ACTION_HANDLERS),
                "description": (
                    "The UniFi operation to perform. "
                    "list_devices — list all network devices (APs, switches, gateways) with status. "
                    "list_clients — list connected clients. "
                    "device_status — health/detail for one device by target MAC (omit target for the site overview). "
                    "site_health — the overall site health summary. "
                    "block_client / unblock_client / disconnect_client — control a client by target MAC. "
                    "restart_device / locate_device / stop_locate_device — control a device by target MAC. "
                    "power_cycle_port — power-cycle a device's PoE port (needs target + port_idx). "
                    "list_wifi — list WiFi networks. update_wifi — change a WiFi network (needs wlan_id + updates). "
                    "list_port_forwards — list port-forward rules. create_port_forward — add one (needs rule). "
                    "update_port_forward — change one (needs rule_id + updates). "
                    "list_traffic_rules — list traffic rules. update_traffic_rule — change one (needs rule_id + updates). "
                    "authorize_guest — authorize a guest device by target MAC."
                ),
            },
            Keys.target: {
                "type": "string",
                "description": (
                    "MAC address of the device or client to act on, e.g. "
                    "'aa:bb:cc:dd:ee:ff'. Required for device_status detail, client "
                    "control, device control, and authorize_guest."
                ),
            },
            Keys.active_only: {
                "type": "boolean",
                "description": (
                    "list_clients: when true (default) only currently connected "
                    "clients; false includes all known clients."
                ),
                "default": True,
            },
            Keys.port_idx: {
                "type": "integer",
                "description": "power_cycle_port: the switch PoE port index to cycle.",
            },
            Keys.wlan_id: {
                "type": "string",
                "description": "update_wifi: id of the WiFi network to change (from list_wifi).",
            },
            Keys.rule_id: {
                "type": "string",
                "description": (
                    "update_port_forward / update_traffic_rule: id of the rule to "
                    "change (from the matching list action)."
                ),
            },
            Keys.updates: {
                "type": "object",
                "description": (
                    "update_wifi / update_port_forward / update_traffic_rule: the "
                    "fields to change, e.g. {'enabled': false} to disable, or "
                    "{'x_passphrase': 'newpass'} to change a WiFi password."
                ),
            },
            Keys.rule: {
                "type": "object",
                "description": "create_port_forward: the full port-forward rule definition.",
            },
            Keys.minutes: {
                "type": "integer",
                "description": "authorize_guest: session duration in minutes (default 60).",
                "default": 60,
            },
            Keys.up_kbps: {
                "type": "integer",
                "description": "authorize_guest: upload bandwidth cap in Kbps (optional).",
            },
            Keys.down_kbps: {
                "type": "integer",
                "description": "authorize_guest: download bandwidth cap in Kbps (optional).",
            },
            Keys.quota_mb: {
                "type": "integer",
                "description": "authorize_guest: total data quota in MB (optional).",
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — translate the flat action, then delegate to the base ─────

    def run(self, params: CapabilityParamsBag) -> ToolResult:
        action = self._resolve_action(params)
        handler_params = dict(params.extra)

        spec = _ACTION_SPEC.get(action)
        if spec is not None:
            # Translate the flat addressing param into the capability's MAC, and
            # inject the fixed sub-command DSL the coarse handler expects. The
            # model supplied neither command nor sub_action — that prose-DSL is an
            # internal detail this table owns. The translated ``target`` is dropped
            # so it never leaks into a REST body; ``device_status`` keeps it because
            # its handler (get_info) reads ``target`` directly.
            if action in _TARGET_TO_MAC:
                target = (cast(str, handler_params.get(Keys.target)) or "").strip()
                if target and action != "device_status":
                    handler_params["mac"] = target
                    handler_params.pop(Keys.target, None)
            for key, value in spec.inject.items():
                handler_params.setdefault(key, value)

        result = self._dispatch(action, handler_params)
        if result.status != "success" or not isinstance(result.body, dict):
            # not-connected / unknown-action / handler-unavailable already carry a
            # canonical code from the base — surface them untouched.
            return result
        return _shape_result(action, result.body)


# ---------------------------------------------------------------------------
# Result shaping — a trimmed, model-friendly body per action family
# ---------------------------------------------------------------------------

# The list-shaped handler bodies and their row key, so list actions surface a
# uniform JSON-rows body with a count meta.
_LIST_BODIES = {
    "devices": "list_devices",
    "clients": "list_clients",
    "wlans": "list_wifi",
    "rules": "list_port_forwards",
}


def _shape_result(action: str, body: dict[str, object]) -> ToolResult:
    """A handler that surfaced its own ``error`` key (a backend failure or a target
    that the controller did not find, NOT a bad input the pre-gate would catch) is
    passed through as the success body — the contract reserves ``err()`` for
    anticipated input failures the ability itself detects; the dispatcher wraps
    the unexpected."""
    if "error" in body:
        return ToolResult.ok(body, action=action)

    for row_key in _LIST_BODIES:
        if row_key in body and isinstance(body[row_key], list):
            rows = cast(list[object], body[row_key])
            if not rows:
                return ToolResult.no_results(action=action)
            return ToolResult.ok(
                {row_key: rows},
                action=action,
                count=body.get("count", len(rows)),
            )

    # device_status / site_health / control / write echoes: structured already.
    return ToolResult.ok(body, action=action)

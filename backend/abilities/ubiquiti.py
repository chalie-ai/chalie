"""UbiquitiAbility -- control and monitor a UniFi network via the Ubiquiti capability.

Delegates to UbiquitiCapability's tool handlers. Uses skill tag wrapper like
home.py (no rich-media rendering).
"""

import json
import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class UbiquitiAbility(Ability):
    NAME = "ubiquiti"
    SEARCH_TOOLTIP = (
        "UniFi router & WiFi network control: list/block clients, manage SSIDs, "
        "restart access points, port forwarding, traffic rules, guest authorization"
    )
    SUMMARY = (
        "Control and monitor a UniFi network via the Ubiquiti/UniFi controller: "
        "list UniFi hardware devices (access points, switches, routers) and connected "
        "WiFi/LAN clients, get site health and device info, block/unblock/kick clients "
        "by MAC address, restart or locate access points, manage SSIDs and WiFi "
        "passwords, set up port forwarding rules, toggle traffic rules, and authorize "
        "guest WiFi sessions."
    )
    EXAMPLES = [
        "What UniFi network devices do I have?",
        "Which devices are connected to my network right now?",
        "Block that unknown device on my network",
        "Disable my guest WiFi network",
        "Kick this client off the WiFi — disconnect that MAC from the router",
        "Restart the access point in the office and check UniFi site health",
        "Set up port forwarding for port 25565 on the UniFi router",
        "Authorize this MAC address on the guest SSID for 2 hours",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_devices",
                    "list_clients",
                    "get_info",
                    "control_client",
                    "control_device",
                    "manage_wlan",
                    "manage_port_forward",
                    "manage_traffic_rule",
                    "authorize_guest",
                ],
                "description": "The action to perform on the UniFi network.",
            },
            "mac": {
                "type": "string",
                "description": "MAC address of the target device or client (e.g. 'aa:bb:cc:dd:ee:ff').",
            },
            "target": {
                "type": "string",
                "description": "For get_info: 'health' for site overview, or a device MAC for detail.",
            },
            "command": {
                "type": "string",
                "description": (
                    "For control_client: 'block-sta', 'unblock-sta', 'kick-sta'. "
                    "For control_device: 'restart', 'set-locate', 'unset-locate', 'power-cycle'."
                ),
            },
            "sub_action": {
                "type": "string",
                "enum": ["list", "update", "create"],
                "description": (
                    "For manage_wlan and manage_traffic_rule: 'list' or 'update'. "
                    "For manage_port_forward: 'list', 'update', or 'create'."
                ),
            },
            "wlan_id": {
                "type": "string",
                "description": "ID of the WiFi network to update.",
            },
            "rule_id": {
                "type": "string",
                "description": "ID of the port forward or traffic rule to update.",
            },
            "updates": {
                "type": "object",
                "description": (
                    "Fields to update (e.g. {'enabled': false} to disable, "
                    "{'x_passphrase': 'newpass'} to change WiFi password)."
                ),
            },
            "rule": {
                "type": "object",
                "description": "Full rule definition for creating a new port forward rule.",
            },
            "active_only": {
                "type": "boolean",
                "description": "For list_clients: true for connected only, false for all known.",
                "default": True,
            },
            "port_idx": {
                "type": "integer",
                "description": "For control_device power-cycle: the switch port index.",
            },
            "minutes": {
                "type": "integer",
                "description": "For authorize_guest: session duration in minutes.",
                "default": 60,
            },
            "up_kbps": {
                "type": "integer",
                "description": "For authorize_guest: upload bandwidth limit in Kbps.",
            },
            "down_kbps": {
                "type": "integer",
                "description": "For authorize_guest: download bandwidth limit in Kbps.",
            },
            "quota_mb": {
                "type": "integer",
                "description": "For authorize_guest: data quota in MB.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 30

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_devices").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("ubiquiti")

        if cap is None:
            result = {
                "status": "error",
                "error": "Ubiquiti capability not found. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("ubiquiti", json.dumps(result), action=action)}

        if not cap.is_connected():
            result = {
                "status": "error",
                "error": "Ubiquiti capability not connected. Check your UniFi controller settings in the Brain dashboard.",
            }
            return {"text": _skill_tag("ubiquiti", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown ubiquiti action: {action}"}
            return {"text": _skill_tag("ubiquiti", json.dumps(result), action=action)}

        result = self.handle(handler, params, telemetry)

        return {"text": _skill_tag("ubiquiti", json.dumps(result), action=action)}

"""HomeAbility -- control smart home devices and automations via Home Assistant.

Delegates to HomeCapability's tool handlers. Uses skill tag wrapper like
email.py (no rich-media rendering).
"""

import json
import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult

logger = logging.getLogger(__name__)


class HomeAbility(Ability):
    def get_name(self) -> str:
        return "home"

    def get_summary(self) -> str:
        return (
            "Control smart home devices and automations via the connected "
            "Home Assistant instance. Available when the user asks about "
            "devices, lights, climate, sensors, or automations."
        )

    def get_examples(self) -> list[str]:
        return [
            "turn on the living room light",
            "what's the temperature in the bedroom",
            "is the front door locked",
            "turn off all lights downstairs",
            "what automations do I have",
            "trigger the good morning routine",
            "what devices are currently on",
            "set the thermostat to 21 degrees",
        ]

    def get_search_tooltip(self) -> str:
        return "smart home control"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_devices",
                    "get_state",
                    "control",
                    "list_automations",
                    "trigger_automation",
                    "subscribe_events",
                ],
                "description": (
                    "list_devices -- list entities, optionally filtered by domain or area. "
                    "get_state -- get current state of one entity by entity_id. "
                    "control -- call a service on an entity (turn_on, turn_off, set_temperature, etc.). "
                    "list_automations -- list all automations with their enabled state. "
                    "trigger_automation -- manually trigger an automation by automation_id. "
                    "subscribe_events -- subscribe to real-time state changes for an entity."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": "HA entity ID, e.g. 'light.living_room'.",
            },
            "domain": {
                "type": "string",
                "description": "list_devices: filter by HA domain (light, switch, sensor, climate, lock, etc.).",
            },
            "area": {
                "type": "string",
                "description": "list_devices: filter by area name.",
            },
            "service": {
                "type": "string",
                "description": "control: service to call, e.g. 'turn_on', 'turn_off', 'toggle'.",
            },
            "service_data": {
                "type": "object",
                "description": "control: extra service data (brightness, temperature, etc.).",
            },
            "automation_id": {
                "type": "string",
                "description": "trigger_automation: automation entity_id.",
            },
        },
        "required": ["action"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        action = params.get("action", "list_devices").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("home")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Home capability not connected. Configure it in the Brain dashboard.",
            }
            return ToolResult.ok(json.dumps(result), action=action)

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown home action: {action}"}
            return ToolResult.ok(json.dumps(result), action=action)

        from services.innate_skills._capability import dispatch_capability_handler
        result = dispatch_capability_handler(handler, params, self.telemetry)

        return ToolResult.ok(json.dumps(result), action=action)

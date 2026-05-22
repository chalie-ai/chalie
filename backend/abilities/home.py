"""HomeAbility -- control smart home devices and automations via Home Assistant.

Delegates to HomeCapability's tool handlers. Uses skill tag wrapper like
email.py (no rich-media rendering).
"""

import json
import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class HomeAbility(Ability):
    NAME = "home"
    SEARCH_TOOLTIP = "smart home control"
    SUMMARY = (
        "Control smart home devices and automations via the connected "
        "Home Assistant instance. Available when the user asks about "
        "devices, lights, climate, sensors, or automations."
    )
    EXAMPLES = [
        "turn on the living room light",
        "what's the temperature in the bedroom",
        "is the front door locked",
        "turn off all lights downstairs",
        "what automations do I have",
        "trigger the good morning routine",
        "run the good morning routine",
        "set the thermostat to 21 degrees",
    ]
    INPUT_SCHEMA = {
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
                "description": (
                    "trigger_automation: HA entity_id (e.g. 'automation.good_morning') "
                    "OR a unique substring of the automation's friendly name "
                    "(e.g. 'good morning')."
                ),
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 30

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_devices").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("home")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Home capability not connected. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("home", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown home action: {action}"}
            return {"text": _skill_tag("home", json.dumps(result), action=action)}

        result = self.handle(handler, params, telemetry)

        return {"text": _skill_tag("home", json.dumps(result), action=action)}

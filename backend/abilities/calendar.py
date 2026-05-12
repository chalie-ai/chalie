"""
CalendarAbility — List, view, and update calendar events via the connected CalDAV account.

Delegates to MailCapability's CalDAV handler. The capability is lazy-loaded
inside execute() to avoid circular imports and to handle the case where the
capability is not yet connected.
"""

import json
import logging
from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CALENDAR ABILITY]"


class CalendarAbility(Ability):
    NAME = "calendar"
    SUMMARY = (
        "List, view, and update calendar events from the connected CalDAV account. "
        "Available when the user asks about meetings, appointments, or schedule."
    )
    EXAMPLES = [
        "what's on my calendar today",
        "show my meetings this week",
        "do I have anything scheduled tomorrow",
        "what time is my next meeting",
        "update the project meeting time to 3pm",
        "get details for that appointment",
        "what events do I have coming up",
        "show me my schedule for next Monday",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_events", "get_event", "update_event"],
                "description": (
                    "The calendar action to perform. "
                    "list_events — list events within a date range. "
                    "get_event — fetch full details of a specific event by its CalDAV UID. "
                    "update_event — update fields of an existing event by its CalDAV UID."
                ),
            },
            "date_from": {
                "type": "string",
                "description": "list_events: ISO date lower bound (YYYY-MM-DD). Defaults to today.",
            },
            "date_to": {
                "type": "string",
                "description": "list_events: ISO date upper bound (YYYY-MM-DD). Defaults to 7 days from date_from.",
            },
            "calendar_name": {
                "type": "string",
                "description": "list_events: filter to a specific calendar by name.",
            },
            "limit": {
                "type": "integer",
                "description": "list_events: maximum number of events to return.",
            },
            "uid": {
                "type": "string",
                "description": "get_event / update_event: CalDAV UID of the event.",
            },
            "summary": {
                "type": "string",
                "description": "update_event: new event title.",
            },
            "dtstart": {
                "type": "string",
                "description": "update_event: new start datetime in ISO 8601 format.",
            },
            "dtend": {
                "type": "string",
                "description": "update_event: new end datetime in ISO 8601 format.",
            },
            "location": {
                "type": "string",
                "description": "update_event: new location string.",
            },
            "description": {
                "type": "string",
                "description": "update_event: new event description.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 30

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_events").lower()

        from capabilities import load_capabilities
        caps = load_capabilities()
        cap = caps.get("mail")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Calendar capability not connected. Configure the mail integration in the Brain dashboard.",
            }
            return {"text": _skill_tag("calendar", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}

        _ACTION_TO_HANDLER = {
            "list_events": "list_events",
            "get_event": "get_event",
            "update_event": "update_event",
        }

        handler_name = _ACTION_TO_HANDLER.get(action)
        if handler_name is None:
            result = {"status": "error", "error": f"Unknown calendar action: {action}"}
            return {"text": _skill_tag("calendar", json.dumps(result), action=action)}

        handler = tool_map.get(handler_name)
        if handler is None:
            result = {
                "status": "error",
                "error": (
                    f"Handler '{handler_name}' not available. "
                    "CalDAV may not be connected for this mail account."
                ),
            }
            return {"text": _skill_tag("calendar", json.dumps(result), action=action)}

        try:
            action_params = {k: v for k, v in params.items() if not k.startswith("_") and k != "action"}
            raw = handler(topic="", params=action_params, telemetry=telemetry)
            if isinstance(raw, dict):
                result = raw
            else:
                result = {"status": "ok", "data": raw}
        except Exception as exc:
            logger.error(f"{LOG_PREFIX} action={action} failed: {exc}", exc_info=True)
            result = {"status": "error", "error": str(exc)}

        return {"text": _skill_tag("calendar", json.dumps(result), action=action)}

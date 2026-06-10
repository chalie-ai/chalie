"""
CalendarAbility — List, view, and update calendar events.

Read operations (list_events, get_event) query scheduled_items via the
shared ``query_items`` engine in the schedule ability — the scheduler owns
the single SQL path for that table.

Write operations (update_event, create_event, delete_event) delegate to
MailCapability's CalDAV handler which pushes changes to the server.
"""

import json as _json
import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CALENDAR ABILITY]"

_RICH_MEDIA_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a calendar card; without it, the user sees only "
    "plain text. Write a brief summary. Example: "
    "\"<span id='{tag}'>You have 3 meetings tomorrow, starting with standup at 09:00.</span>\""
)


class CalendarAbility(Ability):
    def get_name(self) -> str:
        return "calendar"

    def get_summary(self) -> str:
        return (
            "List, view, and update calendar events from the connected CalDAV account. "
            "Available when the user asks about meetings, appointments, or schedule."
        )

    def get_examples(self) -> list[str]:
        return [
            "what's on my calendar today",
            "show my meetings this week",
            "do I have anything scheduled tomorrow",
            "what time is my next meeting",
            "update the project meeting time to 3pm",
            "get details for that appointment",
            "what events do I have coming up",
            "show me my schedule for next Monday",
        ]

    def get_search_tooltip(self) -> str:
        return "calendar events"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
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

    def run(self, params: dict) -> ToolResult:
        action = params.get("action", "list_events").lower()
        ordinal = params.get("_rich_media_ordinal")

        # --- Read operations: query scheduled_items via the shared engine ---
        if action in ("list_events", "get_event"):
            try:
                result = _read_events(action, params)
            except Exception as exc:
                logger.exception(f"{LOG_PREFIX} action={action} failed: {exc}")
                result = {"status": "error", "error": str(exc)}
            if ordinal is not None and "error" not in result:
                return ToolResult.ok(_serialise_rich(result, action, ordinal), action=action)
            return ToolResult.ok(_json.dumps(result), action=action)

        # --- Write operations: delegate to capability's CalDAV handler ---
        from capabilities import load_capabilities
        cap = (load_capabilities()).get("mail")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Calendar capability not connected. Configure the mail integration in the Brain dashboard.",
            }
            return ToolResult.ok(_json.dumps(result), action=action)

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown calendar action: {action}"}
            return ToolResult.ok(_json.dumps(result), action=action)

        from services.innate_skills._capability import dispatch_capability_handler
        result = dispatch_capability_handler(handler, params, self.telemetry)

        if ordinal is not None and "error" not in result:
            return ToolResult.ok(_serialise_rich(result, action, ordinal), action=action)
        return ToolResult.ok(_json.dumps(result), action=action)


# ---------------------------------------------------------------------------
# Read helpers — query scheduled_items via schedule.query_items
# ---------------------------------------------------------------------------

_EVENT_COLUMNS = ("message", "due_at", "metadata", "external_uid")


def _read_events(action: str, params: dict) -> dict:
    from abilities.schedule import query_items

    if action == "get_event":
        uid = (params.get("uid") or "").strip()
        if not uid:
            return {"error": "uid is required"}
        rows = query_items(
            hidden=1, source="mail", item_type="event",
            external_uid=f"caldav:{uid}",
            columns=_EVENT_COLUMNS, limit=1,
        )
        if not rows:
            return {"error": f"Event not found (UID: {uid})"}
        return _format_event(rows[0])

    # list_events
    limit = min(int(params.get("limit", 50)), 200)
    calendar_name = params.get("calendar_name")

    rows = query_items(
        hidden=1, source="mail", item_type="event",
        date_from=params.get("date_from"),
        date_to=params.get("date_to"),
        limit=limit,
        columns=_EVENT_COLUMNS,
    )

    events = []
    for row in rows:
        ev = _format_event(row)
        if calendar_name and ev.get("calendar_name", "").lower() != calendar_name.lower():
            continue
        events.append(ev)

    return {"events": events, "count": len(events)}


def _format_event(row: dict) -> dict:
    meta = parse_json_column(row.get("metadata"))
    ext_uid = row.get("external_uid", "")
    uid = ext_uid.removeprefix("caldav:") if ext_uid else meta.get("uid", "")
    return {
        "uid": uid,
        "title": row.get("message", ""),
        "dtstart": row.get("due_at"),
        "dtend": meta.get("dtend"),
        "location": meta.get("location"),
        "attendees": meta.get("attendees", []),
        "all_day": meta.get("all_day", False),
        "calendar_name": meta.get("calendar_name"),
    }


def _serialise_rich(result: dict, action: str, ordinal: int) -> str:
    tag = f"calendar_{ordinal}"
    payload = {"action_performed": action}
    if "events" in result:
        payload["events"] = result["events"]
        payload["count"] = result.get("count", len(result["events"]))
    else:
        payload["event"] = result
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return f"{_json.dumps(payload)}\n\n{instruction}"

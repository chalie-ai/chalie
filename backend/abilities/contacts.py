"""
ContactsAbility — Search and look up contacts from the connected address book (CardDAV).

Delegates to MailCapability's CardDAV handler. The capability is lazy-loaded
inside execute() to avoid circular imports and to handle the case where the
capability is not yet connected.
"""

import json
import logging
from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CONTACTS ABILITY]"


class ContactsAbility(Ability):
    NAME = "contacts"
    SUMMARY = (
        "Search and look up contacts from the connected address book (CardDAV). "
        "Available when the user asks for someone's phone number, email, or contact details."
    )
    EXAMPLES = [
        "find John's phone number",
        "look up Sarah's email address",
        "who is in my contacts",
        "get me the contact details for Mike",
        "search my address book for someone named Alex",
        "what's the phone number for the dentist",
        "show me all my contacts",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get"],
                "description": (
                    "The contacts action to perform. "
                    "list — search contacts by name or keyword, or list all contacts. "
                    "get — fetch full details for a specific contact by name or email."
                ),
            },
            "query": {
                "type": "string",
                "description": "list: search query to filter contacts by name or other fields.",
            },
            "limit": {
                "type": "integer",
                "description": "list: maximum number of contacts to return.",
            },
            "identifier": {
                "type": "string",
                "description": "get: name or email address identifying the contact to look up.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 15

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list").lower()

        from capabilities import load_capabilities
        caps = load_capabilities()
        cap = caps.get("mail")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Contacts capability not connected. Configure the mail integration in the Brain dashboard.",
            }
            return {"text": _skill_tag("contacts", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}

        _ACTION_TO_HANDLER = {
            "list": "list_contacts",
            "get": "get_contact",
        }

        handler_name = _ACTION_TO_HANDLER.get(action)
        if handler_name is None:
            result = {"status": "error", "error": f"Unknown contacts action: {action}"}
            return {"text": _skill_tag("contacts", json.dumps(result), action=action)}

        handler = tool_map.get(handler_name)
        if handler is None:
            result = {
                "status": "error",
                "error": (
                    f"Handler '{handler_name}' not available. "
                    "CardDAV may not be connected for this mail account."
                ),
            }
            return {"text": _skill_tag("contacts", json.dumps(result), action=action)}

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

        return {"text": _skill_tag("contacts", json.dumps(result), action=action)}

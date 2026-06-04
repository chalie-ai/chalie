"""
ContactsAbility — Search and look up contacts from the connected address book (CardDAV).

Delegates to MailCapability's CardDAV handler. The capability is lazy-loaded
inside execute() to avoid circular imports and to handle the case where the
capability is not yet connected.
"""

import json
import logging
from abilities._ability import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CONTACTS ABILITY]"

_RICH_MEDIA_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a contact card; without it, the user sees only "
    "plain text. Write a brief summary. Example: "
    "\"<span id='{tag}'>Here's John's contact info.</span>\""
)


class ContactsAbility(Ability):
    NAME = "contacts"
    SEARCH_TOOLTIP = "contact book"
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

    def run(self, params: dict) -> dict | str:
        action = params.get("action", "list").lower()
        ordinal = params.get("_rich_media_ordinal")

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

        from services.innate_skills._capability import dispatch_capability_handler
        result = dispatch_capability_handler(handler, params, self.telemetry)

        if ordinal is not None and "error" not in result:
            return _serialise_rich(result, action, ordinal)
        return {"text": _skill_tag("contacts", json.dumps(result), action=action)}


def _serialise_rich(result: dict, action: str, ordinal: int) -> str:
    tag = f"contacts_{ordinal}"
    payload = {"action_performed": action}
    if "contacts" in result:
        payload["contacts"] = result["contacts"]
        payload["count"] = result.get("count", len(result["contacts"]))
    elif "contact" in result:
        payload["contact"] = result["contact"]
    else:
        payload.update({k: v for k, v in result.items() if k != "status"})
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return f"{json.dumps(payload)}\n\n{instruction}"

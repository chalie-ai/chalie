"""
ContactsAbility — Search and look up contacts from the connected address book (CardDAV).

Delegates to MailCapability's CardDAV handler via the shared
``CapabilityAbility`` base, which owns capability loading, the not-connected /
unknown-action errors, handler dispatch, and result wrapping. This ability
supplies only its metadata and the capability/action wiring.
"""

import logging
from typing import ClassVar

from abilities._capability import CapabilityAbility

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CONTACTS ABILITY]"


class ContactsAbility(CapabilityAbility):
    CAPABILITY_KEY: ClassVar[str] = "mail"
    DEFAULT_ACTION: ClassVar[str] = "list"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the mail integration in the Brain dashboard."
    )
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        "list": "list_contacts",
        "get": "get_contact",
    }

    def get_name(self) -> str:
        return "contacts"

    def get_summary(self) -> str:
        return (
            "Search and look up contacts from the connected address book (CardDAV). "
            "Available when the user asks for someone's phone number, email, or contact details."
        )

    def get_examples(self) -> list[str]:
        return [
            "find John's phone number",
            "look up Sarah's email address",
            "who is in my contacts",
            "get me the contact details for Mike",
            "search my address book for someone named Alex",
            "what's the phone number for the dentist",
            "show me all my contacts",
        ]

    def get_search_tooltip(self) -> str:
        return "contact book"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
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

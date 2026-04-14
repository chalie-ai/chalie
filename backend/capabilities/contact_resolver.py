"""ContactResolver — passive people index from IMAP and CalDAV data.

Mines email sender headers and calendar attendees into DataGraphService
as ``kind='user_specific'``, ``key='contact:<email>'`` entries. Provides
:func:`resolve` for cross-capability identity lookup (e.g. turning a
raw email address into a display name, or vice versa).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CONTACT_KEY_PREFIX = "contact:"


def _dgs():
    from services.data_graph_service import get_data_graph_service
    return get_data_graph_service()


def index_person(email: str, name: str | None = None, source: str = "unknown") -> None:
    """Store or reinforce a person identity in the data graph.

    Uses ``kind='user_specific'``, ``key='contact:<email>'``,
    ``value=<display_name>``.  DataGraphService handles reinforcement on
    repeated sightings.
    """
    display = (name or "").strip()
    if not email or "@" not in str(email) or not display:
        return
    try:
        _dgs().store(
            kind="user_specific",
            key=f"{_CONTACT_KEY_PREFIX}{email.strip().lower()}",
            value=display,
            source=source,
        )
    except Exception as exc:
        logger.debug("[contact_resolver] index_person failed for %s: %s", email, exc)


def resolve(identifier: str, limit: int = 5) -> list[dict]:
    """Look up person entries matching *identifier*.

    Uses DataGraphService.recall() with RRF across key/value vec + FTS5.
    """
    if not identifier or not str(identifier).strip():
        return []
    try:
        rows = _dgs().recall(
            query=identifier.strip(),
            kinds=["user_specific"],
            limit=limit,
        )
        contacts = []
        for r in rows:
            key = r.get("key", "")
            if key.startswith(_CONTACT_KEY_PREFIX):
                email = key[len(_CONTACT_KEY_PREFIX):]
                contacts.append({"email": email, "name": r.get("value", "")})
        return contacts
    except Exception as exc:
        logger.debug("[contact_resolver] resolve(%r) failed: %s", identifier, exc)
        return []


def get_tool() -> dict:
    """Return a tool definition dict for ``resolve_contact``.

    Registered at startup so the LLM can resolve names, partial emails,
    or identifiers to known contacts from the people index built by
    IMAP senders and CalDAV attendees.
    """

    def _execute(topic, params, config=None, telemetry=None):
        query = (params.get("query") or "").strip()
        if not query:
            return {"contacts": [], "count": 0}
        limit = min(int(params.get("limit", 10)), 20)
        matches = resolve(query, limit=limit)
        return {"contacts": matches, "count": len(matches)}

    return {
        "name": "resolve_contact",
        "description": (
            "Look up a person by name, email address, or partial identifier. "
            "Returns matching contacts with their email and display name. "
            "Use before sending email to resolve a name to an address, "
            "or to identify meeting attendees."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": (
                    "Name, email address, or partial identifier to search for "
                    "(e.g. 'Sarah', 'chen@', 'alice@corp.com')."
                ),
            },
            "limit": {
                "type": "integer",
                "required": False,
                "description": "Max results to return (default 10, max 20).",
            },
        },
        "returns": {
            "contacts": {
                "type": "array",
                "description": "Matching people with 'email' and 'name' fields.",
            },
            "count": {"type": "integer", "description": "Number of matches."},
        },
        "constraints": {"timeout_seconds": 5},
        "handler": _execute,
    }

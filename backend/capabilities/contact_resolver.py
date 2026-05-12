"""ContactResolver — passive people index from IMAP and CardDAV data.

Two storage formats coexist under ``kind='user_specific'``:

* **IMAP senders** (lightweight): ``key='contact:<email>'``,
  ``value='<display_name>'``.  Written by :func:`index_person`.
* **CardDAV profiles** (rich): ``key='contact:<Display Name>'``,
  ``value='<JSON profile object>'``.  Written by :func:`index_contact_profile`.

:func:`resolve` transparently handles both formats.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_CONTACT_KEY_PREFIX = "contact:"


def _dgs():
    from services.data_graph_service import get_data_graph_service
    return get_data_graph_service()


def index_person(email: str, name: str | None = None, source: str = "unknown") -> None:
    """Store or reinforce a lightweight person identity in the data graph.

    Uses ``key='contact:<email>'``, ``value=<display_name>``.
    Called from IMAP sender header parsing.
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


def index_contact_profile(profile: dict, source: str = "carddav") -> None:
    """Store a full contact profile in the data graph.

    Uses ``key='contact:<Display Name>'``, ``value=<JSON profile>``.
    Called from CardDAV sync with the full parsed vCard dict.
    """
    fn = (profile.get("fn") or "").strip()
    if not fn:
        return
    try:
        _dgs().store(
            kind="user_specific",
            key=f"{_CONTACT_KEY_PREFIX}{fn}",
            value=json.dumps(profile, ensure_ascii=False),
            source=source,
        )
    except Exception as exc:
        logger.debug("[contact_resolver] index_contact_profile failed for %s: %s", fn, exc)


def _parse_contact_row(key: str, raw_value: str) -> dict | None:
    """Parse a single data_graph row into a contact dict.

    Handles both JSON profile (CardDAV) and legacy email→name (IMAP) formats.
    """
    if not key.startswith(_CONTACT_KEY_PREFIX):
        return None
    try:
        profile = json.loads(raw_value)
        if isinstance(profile, dict):
            return profile
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    suffix = key[len(_CONTACT_KEY_PREFIX):]
    return {"email": suffix, "name": raw_value}


def resolve(identifier: str, limit: int = 5) -> list[dict]:
    """Look up person entries matching *identifier*.

    Uses DataGraphService.recall() with RRF across key/value vec + FTS5.
    Returns full profile dicts for CardDAV entries, or ``{"email", "name"}``
    for legacy IMAP entries.
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
            parsed = _parse_contact_row(r.get("key", ""), r.get("value", ""))
            if parsed is not None:
                contacts.append(parsed)
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

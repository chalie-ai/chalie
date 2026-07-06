"""ContactResolver — passive people index from IMAP and CardDAV data.

Two storage formats coexist under ``kind='contact'``:

* **IMAP senders** (lightweight): ``key='contact:<email>'``,
  ``value='<display_name>'``.  Written by :func:`index_person`.
* **CardDAV profiles** (rich): ``key='contact:<Display Name>'``,
  ``value='<JSON profile object>'``.  Written by :func:`index_contact_profile`.

Contacts live in their own data_graph kind (``kind='contact'``) rather than the
shared ``user_specific`` namespace so contact reads (:func:`resolve`, the
contacts ability's ``list``/``get``) can be scoped to ``kind='contact'`` and
never get crowded out of a shared top-N recall/fetch window by unrelated user
facts / traits. :func:`resolve` transparently handles both storage formats.
"""

from __future__ import annotations

import json
import logging
from typing import cast

from models.contact import ContactRow
from services.contact_service import ContactService

logger = logging.getLogger(__name__)

_CONTACT_KEY_PREFIX = "contact:"


def index_person(email: str, name: str | None = None, source: str = "unknown") -> None:
    """Store or reinforce a lightweight person identity in the data graph.

    Uses ``key='contact:<email>'``, ``value=<display_name>``.
    Called from IMAP sender header parsing.
    """
    display = (name or "").strip()
    if not email or "@" not in str(email) or not display:
        return
    try:
        ContactService().store(
            key=f"{_CONTACT_KEY_PREFIX}{email.strip().lower()}",
            value=display,
            source=source,
        )
    except Exception as exc:
        logger.debug("[contact_resolver] index_person failed for %s: %s", email, exc)


def index_contact_profile(profile: dict[str, object], source: str = "carddav") -> None:
    """Store a full contact profile in the data graph.

    Uses ``key='contact:<Display Name>'``, ``value=<JSON profile>``.
    Called from CardDAV sync with the full parsed vCard dict.
    """
    fn = (cast(str, profile.get("fn")) or "").strip()
    if not fn:
        return
    try:
        ContactService().store(
            key=f"{_CONTACT_KEY_PREFIX}{fn}",
            value=json.dumps(profile, ensure_ascii=False),
            source=source,
        )
    except Exception as exc:
        logger.debug("[contact_resolver] index_contact_profile failed for %s: %s", fn, exc)


def _parse_contact_row(key: str, raw_value: str) -> dict[str, object] | None:
    """Parse a single data_graph row into a contact dict."""
    try:
        profile = json.loads(raw_value)
        if isinstance(profile, dict):
            return profile
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    suffix = key[len(_CONTACT_KEY_PREFIX):]
    return {"email": suffix, "name": raw_value}


def resolve(identifier: str, limit: int = 5) -> list[dict[str, object]]:
    """Look up person entries matching *identifier* via FTS5 over the
    ``contact`` kind (:meth:`~models.contact.ContactRow.search`), ranked by
    FTS rank.

    Scoped to ``kind='contact'`` so non-contact user facts / traits sharing the
    data_graph cannot crowd a real contact out of the top-N window.
    """
    if not identifier or not str(identifier).strip():
        return []
    try:
        rows = ContactRow.search(identifier.strip(), limit)
        contacts = []
        for r in rows:
            parsed = _parse_contact_row(r.key, r.value or "")
            if parsed is not None:
                contacts.append(parsed)
        return contacts
    except Exception as exc:
        logger.debug("[contact_resolver] resolve(%r) failed: %s", identifier, exc)
        return []

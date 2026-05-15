"""CarddavHandler — contact sync for the unified mail capability.

Fetches vCards from Google and Apple CardDAV endpoints, parses them into
normalised contact dicts, and stores them in the knowledge graph via
``contact_resolver.index_person``.  Provides ``list_contacts`` and
``get_contact`` tool handlers for LLM-accessible identity lookup.

Only Google and Apple support CardDAV; call sites are responsible for
checking ``provider.carddav_url`` before instantiating this handler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _dgs():
    from services.data_graph_service import get_data_graph_service
    return get_data_graph_service()


class CarddavHandler:
    """Sync CardDAV contacts into the knowledge graph."""

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------

    def open_client(self, url: str, username: str, password: str):
        """Open a DAVClient for the given CardDAV endpoint.

        Returns a connected ``caldav.DAVClient`` or ``None`` on failure.
        """
        try:
            import caldav  # noqa: PLC0415

            client = caldav.DAVClient(url=url, username=username, password=password)
            # Probe the principal to confirm credentials are valid.
            client.principal()
            return client
        except Exception as exc:
            logger.warning("[carddav] open_client failed for %s: %s", url, exc)
            return None

    # -----------------------------------------------------------------------
    # Sync
    # -----------------------------------------------------------------------

    def sync_contacts(self, client) -> list[dict]:
        """Fetch all vCards from every address book on *client*.

        Returns a (possibly empty) list of normalised contact dicts.  Failures
        for individual cards are logged and skipped.
        """
        import caldav as _caldav  # noqa: PLC0415

        contacts: list[dict] = []
        try:
            principal = client.principal()
            address_books = _get_address_books(principal)
        except Exception as exc:
            logger.warning("[carddav] could not list address books: %s", exc)
            return contacts

        # Fallback: if principal-based discovery found nothing, treat the
        # client's own URL as a direct address-book collection.  This covers
        # servers (e.g. Radicale) where the caldav library lacks CardDAV
        # address-book discovery but the collection URL is known.
        if not address_books:
            try:
                direct = _caldav.Calendar(client=client, url=str(client.url))
                address_books = [direct]
                logger.info("[carddav] using direct collection URL as address book")
            except Exception:
                pass

        for abook in address_books:
            try:
                cards = _fetch_vcard_objects(abook)
            except Exception as exc:
                logger.warning("[carddav] could not fetch cards from address book: %s", exc)
                continue

            for card in cards:
                try:
                    raw = _extract_vcard_data(card)
                    if not raw:
                        continue
                    contact = self.parse_vcard(raw)
                    if contact:
                        contacts.append(contact)
                except Exception as exc:
                    logger.debug("[carddav] skipping malformed card: %s", exc)

        logger.info("[carddav] synced %d contacts", len(contacts))
        return contacts

    # -----------------------------------------------------------------------
    # vCard parsing
    # -----------------------------------------------------------------------

    def parse_vcard(self, vcard_data: str) -> dict | None:
        """Parse a single vCard string into a normalised contact dict.

        Returns ``None`` if the vCard has no usable identity (no FN and no
        email).  All other fields fall back gracefully to empty strings / lists.
        """
        try:
            import vobject  # noqa: PLC0415
        except ImportError:
            logger.error("[carddav] vobject library not installed")
            return None

        try:
            card = vobject.readOne(vcard_data)
        except Exception as exc:
            logger.debug("[carddav] vobject parse error: %s", exc)
            return None

        uid = _vcard_text(card, "uid")
        fn = _vcard_text(card, "fn")
        given_name, family_name = _vcard_name(card)
        nickname = _vcard_text(card, "nickname")
        emails = _vcard_typed_list(card, "email")
        phones = _vcard_typed_list(card, "tel")
        org = _vcard_org(card)
        title = _vcard_text(card, "title")

        if not fn and not emails:
            return None

        return {
            "uid": uid,
            "fn": fn,
            "given_name": given_name,
            "family_name": family_name,
            "nickname": nickname,
            "emails": emails,
            "phones": phones,
            "org": org,
            "title": title,
        }

    # -----------------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------------

    def index_contacts(self, contacts: list[dict]) -> None:
        """Store *contacts* as full profile objects in the data graph.

        Each contact is stored with ``key='contact:<Display Name>'`` and
        ``value`` as a JSON object containing all available vCard fields.
        """
        from capabilities.contact_resolver import index_contact_profile  # noqa: PLC0415
        for contact in contacts:
            fn = (contact.get("fn") or "").strip()
            if not fn:
                continue
            try:
                index_contact_profile(contact, source="carddav")
            except Exception as exc:
                logger.debug("[carddav] index_contacts failed for %s: %s", fn, exc)

    # -----------------------------------------------------------------------
    # Monitor (full sync cycle)
    # -----------------------------------------------------------------------

    def monitor(self, client) -> None:
        """Run one full sync cycle: fetch → parse → index."""
        contacts = self.sync_contacts(client)
        if contacts:
            self.index_contacts(contacts)

    # -----------------------------------------------------------------------
    # Tool handlers
    # -----------------------------------------------------------------------

    def list_contacts(self, params: dict) -> dict:
        """List contacts from the knowledge store.

        params:
            query (str, optional): Free-text search term.
            limit (int, optional): Max results, default 20.

        Returns ``{"contacts": [...], "count": int}``.
        """
        from capabilities.contact_resolver import (  # noqa: PLC0415
            _parse_contact_row,
            resolve,
        )

        limit = min(int(params.get("limit", 20)), 50)
        query = (params.get("query") or "").strip()

        if query:
            matches = resolve(query, limit=limit)
            return {"contacts": matches, "count": len(matches)}

        try:
            rows = _dgs().fetch(kinds=["user_specific"], order_by="key ASC", limit=limit)
            contacts = []
            for r in rows:
                parsed = _parse_contact_row(r.get("key", ""), r.get("value", ""))
                if parsed is not None:
                    contacts.append(parsed)
        except Exception as exc:
            logger.debug("[carddav] list_contacts fallback failed: %s", exc)
            contacts = []

        return {"contacts": contacts, "count": len(contacts)}

    def get_contact(self, params: dict) -> dict:
        """Get a single contact by email or name.

        params:
            identifier (str, required): Email address or display name.

        Returns ``{"contact": {...}}`` or ``{"error": "Contact not found"}``.
        """
        from capabilities.contact_resolver import resolve  # noqa: PLC0415

        identifier = (params.get("identifier") or "").strip()
        if not identifier:
            return {"error": "identifier is required"}

        matches = resolve(identifier, limit=1)
        if not matches:
            return {"error": "Contact not found"}

        return {"contact": matches[0]}


# ---------------------------------------------------------------------------
# vCard field helpers
# ---------------------------------------------------------------------------


def _vcard_text(card, prop: str) -> str:
    """Return the string value of *prop* from *card*, or empty string."""
    try:
        obj = card.contents.get(prop)
        if not obj:
            return ""
        return str(obj[0].value).strip()
    except Exception:
        return ""


def _vcard_name(card) -> tuple[str, str]:
    """Return ``(given_name, family_name)`` from the N property."""
    try:
        n_list = card.contents.get("n")
        if not n_list:
            return "", ""
        n = n_list[0].value
        # vobject parses N into a Name object with .given and .family
        given = str(getattr(n, "given", "") or "").strip()
        family = str(getattr(n, "family", "") or "").strip()
        return given, family
    except Exception:
        return "", ""


def _vcard_typed_list(card, prop: str) -> list[dict]:
    """Return all values for *prop* with their TYPE parameter.

    Each entry is ``{"value": "<str>", "type": "<work|home|other>"}``
    so consumers can distinguish work vs personal emails/phones.
    """
    try:
        items = card.contents.get(prop) or []
        result = []
        for obj in items:
            if not obj.value:
                continue
            val = str(obj.value).strip()
            type_param = ""
            if hasattr(obj, "params") and obj.params:
                raw = obj.params.get("TYPE", [])
                if isinstance(raw, list) and raw:
                    type_param = str(raw[0]).lower()
                elif isinstance(raw, str):
                    type_param = raw.lower()
            result.append({"value": val, "type": type_param or "other"})
        return result
    except Exception:
        return []


def _vcard_org(card) -> str:
    """Return the ORG property value, flattening list values."""
    try:
        org_list = card.contents.get("org")
        if not org_list:
            return ""
        val = org_list[0].value
        if isinstance(val, (list, tuple)):
            return " ".join(str(v) for v in val if v).strip()
        return str(val).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# caldav address-book helpers (isolate API differences)
# ---------------------------------------------------------------------------


def _get_address_books(principal) -> list:
    """Return address books from *principal*, trying multiple API paths."""
    # caldav >=1.3 exposes addressbook_home_set
    try:
        home = principal.addressbook_home_set
        if home is not None:
            if hasattr(home, "addressbooks"):
                return list(home.addressbooks())
            if hasattr(home, "__iter__"):
                return list(home)
    except Exception:
        pass

    # Fallback: some builds expose it on the principal directly
    try:
        books = principal.addressbooks()
        if books is not None:
            return list(books)
    except Exception:
        pass

    return []


def _fetch_vcard_objects(address_book) -> list:
    """Fetch individual vCard objects from an address book."""
    if hasattr(address_book, "vcard_objects"):
        return list(address_book.vcard_objects())
    if hasattr(address_book, "objects"):
        return list(address_book.objects())
    return []


def _extract_vcard_data(card_obj) -> str:
    """Extract the raw vCard string from a caldav card object."""
    if hasattr(card_obj, "vcard_to_string"):
        return str(card_obj.vcard_to_string())
    if hasattr(card_obj, "data"):
        data = card_obj.data
        if not data and hasattr(card_obj, "load"):
            card_obj.load()
            data = card_obj.data
        if data:
            return str(data)
    return ""

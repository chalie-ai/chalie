"""CarddavHandler — contact sync for the unified mail capability."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional, Protocol

    class _DgsProtocol(Protocol):
        def store(self, kind: str, key: str, value: str, *, source: Optional[str] = None) -> object: ...
        def recall(self, query: str, *, kinds: Optional[list[str]] = None, limit: int = 10) -> list[dict[str, object]]: ...
        def fetch(self, *, kinds: Optional[list[str]] = None, order_by: str = "", limit: Optional[int] = None) -> list[dict[str, object]]: ...

    class _CaldavClientProto(Protocol):
        url: object
        def principal(self) -> object: ...

    class _VCardProp(Protocol):
        value: object
        params: dict[str, object]

    class _VCardObj(Protocol):
        contents: dict[str, list["_VCardProp"]]

    class _AddressBookHome(Protocol):
        def addressbooks(self) -> "list[object]": ...

    class _Principal(Protocol):
        addressbook_home_set: "_AddressBookHome | None"
        def addressbooks(self) -> "list[object]": ...

    class _HasVcardObjects(Protocol):
        def vcard_objects(self) -> "list[object]": ...

    class _HasObjects(Protocol):
        def objects(self) -> "list[object]": ...

    class _HasVcardToString(Protocol):
        def vcard_to_string(self) -> str: ...

    class _HasData(Protocol):
        data: object
        def load(self) -> None: ...

logger = logging.getLogger(__name__)


def _dgs() -> "_DgsProtocol":
    from services.data_graph_service import get_data_graph_service
    return cast("_DgsProtocol", get_data_graph_service())


class CarddavHandler:

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------

    def open_client(self, url: str, username: str, password: str) -> "_CaldavClientProto | None":
        """Probe principal to confirm credentials are valid."""
        try:
            import caldav  # noqa: PLC0415

            client = cast("_CaldavClientProto", cast("Callable[..., object]", caldav.DAVClient)(url=url, username=username, password=password))
            # Probe the principal to confirm credentials are valid.
            client.principal()
            return client
        except Exception as exc:
            logger.warning("[carddav] open_client failed for %s: %s", url, exc)
            return None

    # -----------------------------------------------------------------------
    # Sync
    # -----------------------------------------------------------------------

    def sync_contacts(self, client: "_CaldavClientProto") -> list[dict[str, object]]:
        import caldav as _caldav  # noqa: PLC0415

        contacts: list[dict[str, object]] = []
        try:
            principal = cast("_Principal", client.principal())
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
                direct = cast("Callable[..., object]", _caldav.Calendar)(client=client, url=str(client.url))
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

    def parse_vcard(self, vcard_data: str) -> dict[str, object] | None:
        """Return ``None`` if no usable identity (no FN and no email)."""
        try:
            import vobject  # noqa: PLC0415
        except ImportError:
            logger.error("[carddav] vobject library not installed")
            return None

        try:
            card = cast("_VCardObj", vobject.readOne(vcard_data))
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

    def index_contacts(self, contacts: list[dict[str, object]]) -> None:
        from capabilities.contact_resolver import index_contact_profile  # noqa: PLC0415
        for contact in contacts:
            fn = (cast(str, contact.get("fn")) or "").strip()
            if not fn:
                continue
            try:
                index_contact_profile(contact, source="carddav")
            except Exception as exc:
                logger.debug("[carddav] index_contacts failed for %s: %s", fn, exc)

    # -----------------------------------------------------------------------
    # Monitor (full sync cycle)
    # -----------------------------------------------------------------------

    def monitor(self, client: "_CaldavClientProto") -> None:
        contacts = self.sync_contacts(client)
        if contacts:
            self.index_contacts(contacts)

    # -----------------------------------------------------------------------
    # Tool handlers
    # -----------------------------------------------------------------------

    def list_contacts(self, params: dict[str, object]) -> dict[str, object]:
        from capabilities.contact_resolver import (  # noqa: PLC0415
            _parse_contact_row,
            resolve,
        )

        limit = min(int(cast(int, params.get("limit", 20))), 50)
        query = (cast(str, params.get("query")) or "").strip()

        if query:
            matches = resolve(query, limit=limit)
            return {"contacts": matches, "count": len(matches)}

        try:
            rows = _dgs().fetch(kinds=["contact"], order_by="key ASC", limit=limit)
            contacts: list[dict[str, object]] = []
            for r in rows:
                parsed = _parse_contact_row(cast(str, r.get("key", "")), cast(str, r.get("value", "")))
                if parsed is not None:
                    contacts.append(parsed)
        except Exception as exc:
            logger.debug("[carddav] list_contacts fallback failed: %s", exc)
            contacts = []

        return {"contacts": contacts, "count": len(contacts)}

    def get_contact(self, params: dict[str, object]) -> dict[str, object]:
        from capabilities.contact_resolver import resolve  # noqa: PLC0415

        identifier = (cast(str, params.get("identifier")) or "").strip()
        if not identifier:
            return {"error": "identifier is required"}

        matches = resolve(identifier, limit=1)
        if not matches:
            return {"error": "Contact not found"}

        return {"contact": matches[0]}


# ---------------------------------------------------------------------------
# vCard field helpers
# ---------------------------------------------------------------------------


def _vcard_text(card: "_VCardObj", prop: str) -> str:
    try:
        obj = card.contents.get(prop)
        if not obj:
            return ""
        return str(obj[0].value).strip()
    except Exception:
        return ""


def _vcard_name(card: "_VCardObj") -> tuple[str, str]:
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


def _vcard_typed_list(card: "_VCardObj", prop: str) -> list[dict[str, object]]:
    try:
        items = card.contents.get(prop) or []
        result: list[dict[str, object]] = []
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


def _vcard_org(card: "_VCardObj") -> str:
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


def _get_address_books(principal: "_Principal") -> list[object]:
    # caldav >=1.3 exposes addressbook_home_set
    try:
        home = principal.addressbook_home_set
        if home is not None:
            if hasattr(home, "addressbooks"):
                return list(home.addressbooks())
            if hasattr(home, "__iter__"):
                return list(cast("list[object]", home))
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


def _fetch_vcard_objects(address_book: object) -> list[object]:
    if hasattr(address_book, "vcard_objects"):
        return list(cast("_HasVcardObjects", address_book).vcard_objects())
    if hasattr(address_book, "objects"):
        return list(cast("_HasObjects", address_book).objects())
    return []


def _extract_vcard_data(card_obj: object) -> str:
    if hasattr(card_obj, "vcard_to_string"):
        return str(cast("_HasVcardToString", card_obj).vcard_to_string())
    if hasattr(card_obj, "data"):
        data = cast("_HasData", card_obj).data
        if not data and hasattr(card_obj, "load"):
            cast("_HasData", card_obj).load()
            data = cast("_HasData", card_obj).data
        if data:
            return str(data)
    return ""

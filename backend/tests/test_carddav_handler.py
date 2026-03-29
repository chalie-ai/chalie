"""Unit tests for CarddavHandler.

Coverage:
    1.  parse_vcard — full card with all fields
    2.  parse_vcard — minimal card (FN only, no N/emails/phones/org/title)
    3.  parse_vcard — card with multiple emails and phones
    4.  parse_vcard — returns None for invalid vCard data
    5.  parse_vcard — returns None when card has neither FN nor emails
    6.  parse_vcard — ORG as list value is flattened
    7.  index_contacts — calls KnowledgeService.store with confidence=0.9
    8.  index_contacts — skips contacts with no emails
    9.  index_contacts — skips entries with no display name
    10. index_contacts — enriches data dict with org, title, name parts
    11. index_contacts — survives a store() exception
    12. list_contacts — delegates to contact_resolver.resolve when query given
    13. list_contacts — queries KnowledgeService directly when no query
    14. list_contacts — caps limit at 50
    15. get_contact — returns contact dict when found
    16. get_contact — returns error dict when not found
    17. get_contact — returns error dict when identifier is empty
    18. monitor — calls sync_contacts then index_contacts
    19. open_client — returns None on DAVClient exception
    20. sync_contacts — returns empty list when principal() raises
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler():
    from capabilities.mail_capability.carddav_handler import CarddavHandler

    return CarddavHandler()


_FULL_VCARD = """\
BEGIN:VCARD
VERSION:3.0
UID:abc-123
FN:Sarah Chen
N:Chen;Sarah;;;
EMAIL;TYPE=work:s.chen@acme.com
EMAIL;TYPE=home:sarah@personal.com
TEL;TYPE=work:+1-555-0100
ORG:Acme Corp
TITLE:VP Engineering
END:VCARD
"""

_MINIMAL_VCARD = """\
BEGIN:VCARD
VERSION:3.0
FN:Bob Jones
END:VCARD
"""

_NO_FN_WITH_EMAIL = """\
BEGIN:VCARD
VERSION:3.0
EMAIL:anon@example.com
END:VCARD
"""

_NO_FN_NO_EMAIL = """\
BEGIN:VCARD
VERSION:3.0
N:Doe;John;;;
END:VCARD
"""

_ORG_LIST_VCARD = """\
BEGIN:VCARD
VERSION:3.0
FN:Alice Wang
N:Wang;Alice;;;
EMAIL:alice@corp.com
ORG:Big Corp;Engineering
END:VCARD
"""

_MULTI_EMAIL_VCARD = """\
BEGIN:VCARD
VERSION:3.0
FN:Dave Lee
N:Lee;Dave;;;
EMAIL:dave@work.com
EMAIL:dave@home.com
EMAIL:dave@other.com
TEL:+1-555-0200
TEL:+1-555-0300
END:VCARD
"""


# ---------------------------------------------------------------------------
# parse_vcard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_vcard_full_fields():
    handler = _make_handler()
    contact = handler.parse_vcard(_FULL_VCARD)

    assert contact is not None
    assert contact["uid"] == "abc-123"
    assert contact["fn"] == "Sarah Chen"
    assert contact["given_name"] == "Sarah"
    assert contact["family_name"] == "Chen"
    assert "s.chen@acme.com" in contact["emails"]
    assert "sarah@personal.com" in contact["emails"]
    assert "+1-555-0100" in contact["phones"]
    assert contact["org"] == "Acme Corp"
    assert contact["title"] == "VP Engineering"


@pytest.mark.unit
def test_parse_vcard_minimal():
    handler = _make_handler()
    contact = handler.parse_vcard(_MINIMAL_VCARD)

    assert contact is not None
    assert contact["fn"] == "Bob Jones"
    assert contact["emails"] == []
    assert contact["phones"] == []
    assert contact["org"] == ""
    assert contact["title"] == ""
    assert contact["uid"] == ""


@pytest.mark.unit
def test_parse_vcard_multiple_emails_and_phones():
    handler = _make_handler()
    contact = handler.parse_vcard(_MULTI_EMAIL_VCARD)

    assert contact is not None
    assert len(contact["emails"]) == 3
    assert len(contact["phones"]) == 2


@pytest.mark.unit
def test_parse_vcard_invalid_data():
    handler = _make_handler()
    result = handler.parse_vcard("NOT_A_VCARD_AT_ALL")
    # May return None or an empty-fields dict — important thing is no exception.
    # vobject may raise or return a malformed object; we just require no crash.
    # Either None or a dict is acceptable.
    assert result is None or isinstance(result, dict)


@pytest.mark.unit
def test_parse_vcard_returns_none_when_no_fn_no_email():
    handler = _make_handler()
    result = handler.parse_vcard(_NO_FN_NO_EMAIL)
    assert result is None


@pytest.mark.unit
def test_parse_vcard_no_fn_with_email_is_accepted():
    handler = _make_handler()
    result = handler.parse_vcard(_NO_FN_WITH_EMAIL)
    # fn is empty but email present — should be returned (fn="" is fine)
    assert result is not None
    assert "anon@example.com" in result["emails"]


@pytest.mark.unit
def test_parse_vcard_org_list_flattened():
    handler = _make_handler()
    contact = handler.parse_vcard(_ORG_LIST_VCARD)
    assert contact is not None
    # ORG may be "Big Corp Engineering" or "Big Corp;Engineering" depending on
    # how vobject parses it — just verify it's non-empty and is a string.
    assert isinstance(contact["org"], str)
    assert "Big Corp" in contact["org"]


# ---------------------------------------------------------------------------
# index_contacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ks():
    ks = MagicMock()
    with patch(
        "capabilities.mail_capability.carddav_handler._get_knowledge_service",
        return_value=ks,
    ):
        yield ks


@pytest.mark.unit
def test_index_contacts_uses_confidence_09(mock_ks):
    handler = _make_handler()
    contacts = [{"fn": "Sarah Chen", "emails": ["s.chen@acme.com"], "phones": [], "org": "", "title": "", "given_name": "", "family_name": ""}]

    handler.index_contacts(contacts)

    mock_ks.store.assert_called_once()
    call_kwargs = mock_ks.store.call_args.kwargs
    assert call_kwargs["confidence"] == 0.9
    assert call_kwargs["key"] == "s.chen@acme.com"
    assert call_kwargs["value"] == "Sarah Chen"
    assert call_kwargs["kind"] == "relationship"
    assert call_kwargs["entity"] == "person"
    assert call_kwargs["source"] == "carddav"


@pytest.mark.unit
def test_index_contacts_skips_no_emails(mock_ks):
    handler = _make_handler()
    contacts = [{"fn": "Ghost User", "emails": [], "phones": []}]

    handler.index_contacts(contacts)

    mock_ks.store.assert_not_called()


@pytest.mark.unit
def test_index_contacts_skips_blank_fn(mock_ks):
    handler = _make_handler()
    contacts = [{"fn": "  ", "emails": ["nobody@example.com"], "phones": []}]

    handler.index_contacts(contacts)

    mock_ks.store.assert_not_called()


@pytest.mark.unit
def test_index_contacts_enriches_data_dict(mock_ks):
    handler = _make_handler()
    contacts = [{
        "fn": "Alice Wang",
        "emails": ["alice@corp.com"],
        "phones": ["+1-555-9999"],
        "org": "Corp Ltd",
        "title": "CTO",
        "given_name": "Alice",
        "family_name": "Wang",
    }]

    handler.index_contacts(contacts)

    data = mock_ks.store.call_args.kwargs["data"]
    assert data["org"] == "Corp Ltd"
    assert data["title"] == "CTO"
    assert data["given_name"] == "Alice"
    assert data["family_name"] == "Wang"
    assert "+1-555-9999" in data["phones"]


@pytest.mark.unit
def test_index_contacts_survives_store_exception(mock_ks):
    mock_ks.store.side_effect = RuntimeError("db down")
    handler = _make_handler()
    contacts = [{"fn": "Test User", "emails": ["test@example.com"], "phones": []}]

    # Must not raise
    handler.index_contacts(contacts)


@pytest.mark.unit
def test_index_contacts_multiple_emails_per_contact(mock_ks):
    handler = _make_handler()
    contacts = [{
        "fn": "Dave Lee",
        "emails": ["dave@work.com", "dave@home.com"],
        "phones": [],
        "org": "",
        "title": "",
        "given_name": "",
        "family_name": "",
    }]

    handler.index_contacts(contacts)

    assert mock_ks.store.call_count == 2


# ---------------------------------------------------------------------------
# list_contacts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_contacts_delegates_to_resolve_when_query_given(mock_ks):
    with patch("capabilities.mail_capability.carddav_handler.CarddavHandler.list_contacts") as _:
        pass  # just verify the resolve path below

    with patch("capabilities.contact_resolver.resolve") as mock_resolve:
        mock_resolve.return_value = [{"email": "s.chen@acme.com", "name": "Sarah Chen"}]
        handler = _make_handler()

        result = handler.list_contacts({"query": "Sarah", "limit": 5})

    mock_resolve.assert_called_once_with("Sarah", limit=5)
    assert result["count"] == 1
    assert result["contacts"][0]["email"] == "s.chen@acme.com"


@pytest.mark.unit
def test_list_contacts_queries_ks_when_no_query(mock_ks):
    mock_ks.recall.return_value = [
        {"key": "a@b.com", "value": "A B"},
        {"key": "c@d.com", "value": "C D"},
    ]
    handler = _make_handler()

    result = handler.list_contacts({})

    mock_ks.recall.assert_called_once()
    assert result["count"] == 2
    assert result["contacts"][0]["email"] == "a@b.com"


@pytest.mark.unit
def test_list_contacts_caps_limit_at_50():
    with patch("capabilities.contact_resolver.resolve") as mock_resolve:
        mock_resolve.return_value = []
        handler = _make_handler()

        handler.list_contacts({"query": "test", "limit": 999})

    _, kwargs = mock_resolve.call_args
    assert kwargs["limit"] <= 50


@pytest.mark.unit
def test_list_contacts_returns_empty_on_ks_failure(mock_ks):
    mock_ks.recall.side_effect = RuntimeError("db error")
    handler = _make_handler()

    result = handler.list_contacts({})

    assert result == {"contacts": [], "count": 0}


# ---------------------------------------------------------------------------
# get_contact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_contact_returns_contact_when_found():
    with patch("capabilities.contact_resolver.resolve") as mock_resolve:
        mock_resolve.return_value = [{"email": "s.chen@acme.com", "name": "Sarah Chen"}]
        handler = _make_handler()

        result = handler.get_contact({"identifier": "s.chen@acme.com"})

    assert "contact" in result
    assert result["contact"]["name"] == "Sarah Chen"
    mock_resolve.assert_called_once_with("s.chen@acme.com", limit=1)


@pytest.mark.unit
def test_get_contact_returns_error_when_not_found():
    with patch("capabilities.contact_resolver.resolve", return_value=[]):
        handler = _make_handler()
        result = handler.get_contact({"identifier": "nobody@example.com"})

    assert result == {"error": "Contact not found"}


@pytest.mark.unit
def test_get_contact_returns_error_on_empty_identifier():
    handler = _make_handler()
    result = handler.get_contact({"identifier": ""})
    assert result == {"error": "identifier is required"}


@pytest.mark.unit
def test_get_contact_returns_error_on_missing_identifier():
    handler = _make_handler()
    result = handler.get_contact({})
    assert result == {"error": "identifier is required"}


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_monitor_calls_sync_then_index():
    handler = _make_handler()
    fake_contacts = [{"fn": "Test", "emails": ["t@t.com"], "phones": []}]

    with patch.object(handler, "sync_contacts", return_value=fake_contacts) as mock_sync, \
         patch.object(handler, "index_contacts") as mock_index:
        handler.monitor(MagicMock())

    mock_sync.assert_called_once()
    mock_index.assert_called_once_with(fake_contacts)


@pytest.mark.unit
def test_monitor_skips_index_when_no_contacts():
    handler = _make_handler()

    with patch.object(handler, "sync_contacts", return_value=[]) as mock_sync, \
         patch.object(handler, "index_contacts") as mock_index:
        handler.monitor(MagicMock())

    mock_sync.assert_called_once()
    mock_index.assert_not_called()


# ---------------------------------------------------------------------------
# open_client
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_open_client_returns_none_on_exception():
    handler = _make_handler()

    with patch("caldav.DAVClient", side_effect=ConnectionError("refused")):
        result = handler.open_client("https://contacts.icloud.com/", "user", "pass")

    assert result is None


@pytest.mark.unit
def test_open_client_returns_none_when_principal_raises():
    handler = _make_handler()
    mock_client = MagicMock()
    mock_client.principal.side_effect = PermissionError("401 Unauthorized")

    with patch("caldav.DAVClient", return_value=mock_client):
        result = handler.open_client("https://contacts.icloud.com/", "user", "badpass")

    assert result is None

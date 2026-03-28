"""Unit tests for ImapCapability — credential, connect, and ingest."""

from __future__ import annotations

from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

_CREDS = {
    "imap:host": "imap.gmail.com", "imap:port": "993",
    "imap:password": "pw", "imap:email": "u@gmail.com", "imap:tls": "1",
}


def _make():
    from capabilities.imap_capability.capability import ImapCapability
    return ImapCapability()


def _email_bytes(subject="Test", from_="John <john@example.com>",
                 to="jane@example.com", date="Mon, 28 Mar 2026 10:00:00 +0000",
                 msg_id="<123@example.com>"):
    msg = MIMEText("body", "plain")
    msg["From"] = from_
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = date
    msg["Message-ID"] = msg_id
    return msg.as_bytes()


# --- Identity & configure ---

@pytest.mark.unit
def test_identity():
    assert _make().get_id() == "imap"


@pytest.mark.unit
def test_configure_missing_fields():
    with pytest.raises(ValueError, match="email and password required"):
        _make().configure({"email": "u@gmail.com"})


@pytest.mark.unit
def test_configure_unsupported_domain():
    cap = _make()
    with patch.object(cap, "store_credential"):
        with pytest.raises(ValueError, match="Unsupported email provider"):
            cap.configure({"email": "u@unknown-xyz.com", "password": "pw"})


@pytest.mark.unit
def test_configure_known_provider():
    cap = _make()
    with patch.object(cap, "store_credential") as ms, \
         patch.object(cap, "connect", return_value=True):
        cap.configure({"email": "u@gmail.com", "password": "pw"})
    ms.assert_any_call("imap:host", "imap.gmail.com")


@pytest.mark.unit
def test_configure_connect_failure():
    cap = _make()
    with patch.object(cap, "store_credential"), \
         patch.object(cap, "connect", return_value=False), \
         patch.object(cap, "delete_credentials") as d:
        with pytest.raises(ValueError, match="Connection test failed"):
            cap.configure({"email": "u@gmail.com", "password": "bad"})
    d.assert_called_once()


@pytest.mark.unit
def test_connect_success():
    cap = _make()
    mc = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", return_value=mc):
        assert cap.connect() is True


@pytest.mark.unit
def test_connect_failure():
    cap = _make()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", side_effect=Exception("refused")):
        assert cap.connect() is False


# --- parse_headers ---

@pytest.mark.unit
def test_parse_headers():
    from capabilities.imap_capability.capability import parse_headers
    r = parse_headers(42, _email_bytes())
    assert (r["uid"], r["from_addr"], r["subject"]) == (42, "john@example.com", "Test")
    msg = MIMEText("body", "plain")
    msg["From"] = "news@example.com"
    msg["List-Unsubscribe"] = "<mailto:unsub@example.com>"
    assert parse_headers(1, msg.as_bytes())["has_unsubscribe"] is True


# --- ingest ---

@pytest.mark.unit
def test_ingest_not_connected():
    cap = _make()
    assert cap.ingest() == []


@pytest.mark.unit
def test_ingest_fresh():
    cap = _make()
    cap._connected = True
    mc = MagicMock()
    mc.search.return_value = [100, 101]
    mc.fetch.return_value = {
        100: {b"RFC822.HEADER": _email_bytes(subject="A")},
        101: {b"RFC822.HEADER": _email_bytes(subject="B")},
    }
    creds = {**_CREDS, "imap:last_uid": None}
    with patch.object(cap, "load_credential", side_effect=creds.get), \
         patch("imapclient.IMAPClient", return_value=mc), \
         patch.object(cap, "store_credential") as sc:
        results = cap.ingest()
    assert [r["subject"] for r in results] == ["A", "B"]
    sc.assert_called_with("imap:last_uid", "101")


@pytest.mark.unit
def test_ingest_incremental():
    cap = _make()
    cap._connected = True
    mc = MagicMock()
    mc.search.return_value = [200, 201]
    mc.fetch.return_value = {
        200: {b"RFC822.HEADER": _email_bytes(subject="Old")},
        201: {b"RFC822.HEADER": _email_bytes(subject="New")},
    }
    creds = {**_CREDS, "imap:last_uid": "200"}
    with patch.object(cap, "load_credential", side_effect=creds.get), \
         patch("imapclient.IMAPClient", return_value=mc), \
         patch.object(cap, "store_credential"):
        results = cap.ingest()
    assert len(results) == 1


# --- classify_email ---

def _item(addr="a@b.com", subj="Hi", unsub=False, reply_to=""):
    return {"has_unsubscribe": unsub, "from_addr": addr, "subject": subj, "in_reply_to": reply_to}


@pytest.mark.unit
@pytest.mark.parametrize("item,expected", [
    (_item(unsub=True), "noise"),
    (_item(addr="noreply@github.com"), "noise"),
    (_item(addr="newsletter@co.com"), "noise"),
    (_item(reply_to="<prev@ex>"), "actionable"),
    (_item(subj="Please confirm"), "actionable"),
    (_item(subj="Invoice #123"), "actionable"),
    (_item(addr="colleague@work.com", subj="Meeting notes"), "informational"),
])
def test_classify_email(item, expected):
    from capabilities.imap_capability.capability import classify_email
    assert classify_email(item) == expected


# --- understand ---

def _email_item(uid=1, msg_id="<m@ex>", subj="Hi", name="Bob", addr="bob@ex.com", unsub=False, reply=""):
    return {"uid": uid, "message_id": msg_id, "subject": subj, "from_name": name,
            "from_addr": addr, "has_unsubscribe": unsub, "in_reply_to": reply}


@pytest.mark.unit
def test_understand_empty_returns_empty():
    assert _make().understand([]) == []


@pytest.mark.unit
def test_understand_classifies_and_stores():
    cap = _make()
    items = [
        _email_item(),
        _email_item(uid=2, msg_id="<m2@ex>", unsub=True),
        _email_item(uid=3, msg_id="<m3@ex>", reply="<prev@ex>"),
    ]
    ks_mock = MagicMock()
    with patch("services.database_service.get_shared_db_service"), \
         patch("services.knowledge_service.KnowledgeService", return_value=ks_mock):
        result = cap.understand(items)
    assert result[0]["triage"] == "informational"
    assert result[1]["triage"] == "noise"
    assert result[2]["triage"] == "actionable"
    assert result[2]["is_thread"] is True
    assert ks_mock.store.call_count == 3
    assert ks_mock.store.call_args_list[0].kwargs["entity"] == "email"


@pytest.mark.unit
def test_understand_survives_exception():
    items = [_email_item()]
    with patch("services.database_service.get_shared_db_service", side_effect=Exception("db down")):
        assert _make().understand(items) == items


# --- monitor ---

@pytest.mark.unit
def test_monitor_calls_ingest_understand():
    cap = _make()
    with patch.object(cap, "ingest", return_value=[{"uid": 1}]) as ing, \
         patch.object(cap, "understand") as und:
        cap.monitor()
    ing.assert_called_once()
    und.assert_called_once()


@pytest.mark.unit
def test_monitor_skips_understand_when_empty():
    cap = _make()
    with patch.object(cap, "ingest", return_value=[]), \
         patch.object(cap, "understand") as und:
        cap.monitor()
    und.assert_not_called()

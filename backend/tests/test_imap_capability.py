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

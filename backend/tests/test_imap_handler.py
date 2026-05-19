"""Unit tests for ImapHandler.

Tests mock IMAP/SMTP connections and verify header parsing, body extraction,
and search criteria building.
"""

from __future__ import annotations

import email as _email_mod
import email.mime.multipart
import email.mime.text
from unittest.mock import MagicMock, patch

import pytest

from capabilities.mail_capability.imap_handler import (
    ImapHandler,
    _imap_date,
    extract_body,
    parse_headers,
)

MOCK_AUTH_TOKEN = "fake-token-for-test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def handler() -> ImapHandler:
    return ImapHandler()


def _make_header_bytes(
    subject: str = "Hello",
    from_addr: str = "Alice <alice@example.com>",
    to_addr: str = "bob@example.com",
    date: str = "Mon, 01 Jan 2024 12:00:00 +0000",
    message_id: str = "<abc123@mail.example.com>",
    in_reply_to: str = "",
    list_unsubscribe: str = "",
) -> bytes:
    """Build minimal raw header bytes."""
    lines = [
        f"Subject: {subject}",
        f"From: {from_addr}",
        f"To: {to_addr}",
        f"Date: {date}",
        f"Message-ID: {message_id}",
    ]
    if in_reply_to:
        lines.append(f"In-Reply-To: {in_reply_to}")
    if list_unsubscribe:
        lines.append(f"List-Unsubscribe: {list_unsubscribe}")
    return "\r\n".join(lines).encode()


def _make_raw_email(plain: str | None = None, html: str | None = None) -> bytes:
    """Build a raw RFC822 email with optional plain/html parts."""
    if plain and not html:
        msg = _email_mod.mime.text.MIMEText(plain, "plain", "utf-8")
    elif plain and html:
        msg = _email_mod.mime.multipart.MIMEMultipart("alternative")
        msg.attach(_email_mod.mime.text.MIMEText(plain, "plain", "utf-8"))
        msg.attach(_email_mod.mime.text.MIMEText(html, "html", "utf-8"))
    elif html:
        msg = _email_mod.mime.text.MIMEText(html, "html", "utf-8")
    else:
        msg = _email_mod.mime.text.MIMEText("", "plain", "utf-8")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    return msg.as_bytes()


# ---------------------------------------------------------------------------
# parse_headers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseHeaders:
    def test_basic_fields(self):
        raw = _make_header_bytes(
            subject="Meeting tomorrow",
            from_addr="Alice Smith <alice@example.com>",
            message_id="<msg1@example.com>",
        )
        result = parse_headers(42, raw)

        assert result["uid"] == 42
        assert result["subject"] == "Meeting tomorrow"
        assert result["from_name"] == "Alice Smith"
        assert result["from_addr"] == "alice@example.com"
        assert result["message_id"] == "<msg1@example.com>"
        assert result["has_unsubscribe"] is False


# ---------------------------------------------------------------------------
# extract_body
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractBody:
    def test_plain_text_returned(self):
        raw = _make_raw_email(plain="Hello, this is the body.")
        assert extract_body(raw) == "Hello, this is the body."

    def test_prefers_plain_over_html(self):
        raw = _make_raw_email(plain="Plain content", html="<p>HTML content</p>")
        result = extract_body(raw)
        assert "Plain content" in result
        assert "<p>" not in result




# ---------------------------------------------------------------------------
# _imap_date
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestImapDate:
    def test_valid_iso_date(self):
        assert _imap_date("2024-06-15") == "15-Jun-2024"




# ---------------------------------------------------------------------------
# open_client
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenClient:
    def test_returns_client_on_success(self, handler):
        mock_client = MagicMock()
        with patch("imapclient.IMAPClient", return_value=mock_client) as mock_cls:
            result = handler.open_client(
                host="imap.example.com", port=993, tls=True,
                email="user@example.com", password=MOCK_AUTH_TOKEN,
            )
        mock_cls.assert_called_once_with(
            "imap.example.com", port=993, ssl=True, timeout=30
        )
        mock_client.login.assert_called_once_with("user@example.com", MOCK_AUTH_TOKEN)
        assert result is mock_client


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngest:
    def _make_imap_client(self, uids, raw_map):
        client = MagicMock()
        client.search.return_value = uids
        client.fetch.return_value = raw_map
        return client

    def test_returns_items_and_new_watermark(self, handler):
        uid = 101
        raw = _make_header_bytes(subject="Invoice", from_addr="billing@co.com")
        client = self._make_imap_client(
            uids=[uid],
            raw_map={uid: {b"RFC822.HEADER": raw}},
        )

        items, new_wm = handler.ingest(client, watermark=0)

        assert len(items) == 1
        assert items[0]["uid"] == uid
        assert items[0]["subject"] == "Invoice"
        assert new_wm == uid



# ---------------------------------------------------------------------------
# search — criteria building
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSearch:
    def _make_imap_client(self, uids=None, raw_map=None):
        client = MagicMock()
        client.search.return_value = uids or []
        client.fetch.return_value = raw_map or {}
        return client

    def test_sender_criteria(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {"sender": "alice@example.com"})
        criteria = client.search.call_args[0][0]
        assert "FROM" in criteria
        assert "alice@example.com" in criteria

    def test_no_criteria_falls_back_to_since(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {})
        criteria = client.search.call_args[0][0]
        assert "SINCE" in criteria

    def test_triage_filter_applied(self, handler):
        uid = 10
        raw = _make_header_bytes(subject="Promo", from_addr="noreply@shop.com")
        client = self._make_imap_client(
            uids=[uid],
            raw_map={uid: {b"RFC822.HEADER": raw}},
        )
        # Stub out email_triage module so it returns "noise" for every email
        import types
        import sys
        stub_module = types.ModuleType("capabilities.mail_capability.email_triage")
        stub_module.classify_email = MagicMock(return_value="noise")
        sys.modules["capabilities.mail_capability.email_triage"] = stub_module
        try:
            result = handler.search(client, {"triage": "actionable"})
        finally:
            del sys.modules["capabilities.mail_capability.email_triage"]
        # noise emails filtered out when requesting actionable
        assert result.get("count", 0) == 0 or result.get("emails") == []



# ---------------------------------------------------------------------------
# read_email
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReadEmail:
    def test_returns_body_and_headers(self, handler):
        uid = 55
        raw = _make_raw_email(plain="This is the email body.")
        client = MagicMock()
        client.fetch.return_value = {uid: {b"RFC822": raw}}

        result = handler.read_email(client, {"uid": uid})

        assert result["uid"] == uid
        assert "This is the email body." in result["body"]



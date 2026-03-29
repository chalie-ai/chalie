"""Unit tests for ImapHandler.

Tests mock IMAP/SMTP connections and verify header parsing, body extraction,
and search criteria building.
"""

from __future__ import annotations

import email as _email_mod
import email.mime.multipart
import email.mime.text
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from capabilities.mail_capability.imap_handler import (
    ImapHandler,
    _imap_date,
    extract_body,
    parse_headers,
)


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

    def test_in_reply_to_populated(self):
        raw = _make_header_bytes(in_reply_to="<original@example.com>")
        result = parse_headers(1, raw)
        assert result["in_reply_to"] == "<original@example.com>"

    def test_has_unsubscribe_true(self):
        raw = _make_header_bytes(list_unsubscribe="<mailto:unsubscribe@list.com>")
        result = parse_headers(1, raw)
        assert result["has_unsubscribe"] is True

    def test_missing_optional_fields_are_empty_strings(self):
        # Minimal headers: no Message-ID, no In-Reply-To
        raw = b"Subject: Hi\r\nFrom: a@b.com\r\nTo: c@d.com\r\n"
        result = parse_headers(5, raw)
        assert result["message_id"] == ""
        assert result["in_reply_to"] == ""

    def test_date_is_iso_string(self):
        raw = _make_header_bytes(date="Mon, 01 Jan 2024 12:00:00 +0000")
        result = parse_headers(1, raw)
        # Should be a valid ISO string, not empty
        assert "2024" in result["date"]


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

    def test_html_fallback_strips_tags(self):
        raw = _make_raw_email(html="<p>Hello <b>world</b></p>")
        result = extract_body(raw)
        assert "<p>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_truncation(self):
        long_text = "x" * 5000
        raw = _make_raw_email(plain=long_text)
        result = extract_body(raw, max_chars=100)
        assert result.endswith("\n[truncated]")
        # Truncated portion is 100 chars + the suffix
        assert len(result) == 100 + len("\n[truncated]")

    def test_no_truncation_within_limit(self):
        short = "Short body."
        raw = _make_raw_email(plain=short)
        result = extract_body(raw)
        assert result == short
        assert "[truncated]" not in result

    def test_empty_body(self):
        raw = _make_raw_email()
        result = extract_body(raw)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _imap_date
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestImapDate:
    def test_valid_iso_date(self):
        assert _imap_date("2024-06-15") == "15-Jun-2024"

    def test_iso_datetime_truncated(self):
        # Should only use the date portion
        assert _imap_date("2024-01-01T10:00:00") == "01-Jan-2024"

    def test_invalid_passthrough(self):
        # Bad input returned as-is
        result = _imap_date("not-a-date")
        assert result == "not-a-date"


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
                email="user@example.com", password="secret",
            )
        mock_cls.assert_called_once_with(
            "imap.example.com", port=993, ssl=True, timeout=30
        )
        mock_client.login.assert_called_once_with("user@example.com", "secret")
        assert result is mock_client

    def test_returns_none_on_connection_error(self, handler):
        with patch("imapclient.IMAPClient", side_effect=ConnectionRefusedError("refused")):
            result = handler.open_client(
                host="bad.host", port=993, tls=True,
                email="u@example.com", password="pw",
            )
        assert result is None

    def test_starttls_used_when_no_ssl(self, handler):
        mock_client = MagicMock()
        with patch("imapclient.IMAPClient", return_value=mock_client) as mock_cls:
            handler.open_client(
                host="imap.example.com", port=143, tls=False,
                email="u@example.com", password="pw",
            )
            # ssl=False passed through — checked inside the patch context
            mock_cls.assert_called_with(
                "imap.example.com", port=143, ssl=False, timeout=30
            )


# ---------------------------------------------------------------------------
# open_smtp
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenSmtp:
    def test_smtp_ssl_on_tls_true(self, handler):
        mock_conn = MagicMock()
        with patch("smtplib.SMTP_SSL", return_value=mock_conn):
            result = handler.open_smtp(
                host="smtp.example.com", port=465, tls=True,
                email="u@example.com", password="pw",
            )
        mock_conn.login.assert_called_once_with("u@example.com", "pw")
        assert result is mock_conn

    def test_smtp_starttls_on_tls_false(self, handler):
        mock_conn = MagicMock()
        with patch("smtplib.SMTP", return_value=mock_conn):
            result = handler.open_smtp(
                host="smtp.example.com", port=587, tls=False,
                email="u@example.com", password="pw",
            )
        mock_conn.starttls.assert_called_once()
        mock_conn.login.assert_called_once_with("u@example.com", "pw")
        assert result is mock_conn

    def test_returns_none_on_error(self, handler):
        with patch("smtplib.SMTP_SSL", side_effect=OSError("refused")):
            result = handler.open_smtp(
                host="bad.host", port=465, tls=True,
                email="u@example.com", password="pw",
            )
        assert result is None


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

    def test_uses_since_when_no_watermark(self, handler):
        client = self._make_imap_client(uids=[], raw_map={})
        handler.ingest(client, watermark=None)
        # Should search by SINCE, not UID range
        args = client.search.call_args[0][0]
        assert args[0] == "SINCE"

    def test_uses_uid_range_when_watermark_set(self, handler):
        client = self._make_imap_client(uids=[], raw_map={})
        handler.ingest(client, watermark=50)
        args = client.search.call_args[0][0]
        assert args[0] == "UID"
        assert args[1] == "51:*"

    def test_empty_inbox_returns_empty_list(self, handler):
        client = self._make_imap_client(uids=[], raw_map={})
        items, wm = handler.ingest(client, watermark=None)
        assert items == []
        assert wm is None

    def test_exception_returns_empty(self, handler):
        client = MagicMock()
        client.select_folder.side_effect = OSError("connection lost")
        items, wm = handler.ingest(client, watermark=0)
        assert items == []


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

    def test_subject_criteria(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {"subject": "Invoice"})
        criteria = client.search.call_args[0][0]
        assert "SUBJECT" in criteria
        assert "Invoice" in criteria

    def test_keyword_criteria(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {"keyword": "payment"})
        criteria = client.search.call_args[0][0]
        assert "TEXT" in criteria
        assert "payment" in criteria

    def test_date_range_criteria(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {"date_from": "2024-01-01", "date_to": "2024-01-31"})
        criteria = client.search.call_args[0][0]
        assert "SINCE" in criteria
        assert "BEFORE" in criteria

    def test_unanswered_criteria(self, handler):
        client = self._make_imap_client()
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            handler.search(client, {"unanswered": True})
        criteria = client.search.call_args[0][0]
        assert "UNANSWERED" in criteria

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

    def test_empty_results(self, handler):
        client = self._make_imap_client(uids=[])
        with patch("capabilities.mail_capability.email_triage.classify_email"):
            result = handler.search(client, {})
        assert result == {"emails": [], "count": 0}


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

    def test_uid_not_found(self, handler):
        client = MagicMock()
        client.fetch.return_value = {}
        result = handler.read_email(client, {"uid": 999})
        assert "error" in result

    def test_missing_uid_param(self, handler):
        client = MagicMock()
        result = handler.read_email(client, {})
        assert result == {"error": "uid is required"}

    def test_non_integer_uid(self, handler):
        client = MagicMock()
        result = handler.read_email(client, {"uid": "abc"})
        assert result == {"error": "uid must be an integer"}

    def test_string_uid_cast(self, handler):
        uid = 7
        raw = _make_raw_email(plain="Body text")
        client = MagicMock()
        client.fetch.return_value = {uid: {b"RFC822": raw}}

        result = handler.read_email(client, {"uid": "7"})
        assert result["uid"] == 7


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSendEmail:
    _creds = dict(
        smtp_host="smtp.example.com",
        smtp_port=465,
        smtp_tls=True,
        email="sender@example.com",
        password="secret",
    )

    def test_success(self, handler):
        mock_conn = MagicMock()
        with patch.object(handler, "open_smtp", return_value=mock_conn):
            result = handler.send_email(
                **self._creds,
                params={"to": "bob@example.com", "subject": "Hi", "body": "Hello"},
            )
        mock_conn.send_message.assert_called_once()
        mock_conn.quit.assert_called_once()
        assert result == {
            "success": True,
            "to": "bob@example.com",
            "subject": "Hi",
        }

    def test_smtp_connect_failure(self, handler):
        with patch.object(handler, "open_smtp", return_value=None):
            result = handler.send_email(
                **self._creds,
                params={"to": "bob@example.com", "subject": "Hi", "body": "Hello"},
            )
        assert "error" in result

    def test_missing_required_params(self, handler):
        result = handler.send_email(
            **self._creds,
            params={"to": "bob@example.com"},  # missing subject + body
        )
        assert result == {"error": "to, subject, and body are required"}

    def test_in_reply_to_sets_headers(self, handler):
        mock_conn = MagicMock()
        sent_msg = {}

        def capture(msg):
            sent_msg["msg"] = msg

        mock_conn.send_message.side_effect = capture
        with patch.object(handler, "open_smtp", return_value=mock_conn):
            handler.send_email(
                **self._creds,
                params={
                    "to": "bob@example.com",
                    "subject": "Re: Hi",
                    "body": "Reply body",
                    "in_reply_to": "<orig@example.com>",
                },
            )
        msg = sent_msg["msg"]
        assert msg["In-Reply-To"] == "<orig@example.com>"
        assert msg["References"] == "<orig@example.com>"

    def test_smtp_error_returns_error_dict(self, handler):
        mock_conn = MagicMock()
        mock_conn.send_message.side_effect = OSError("SMTP error")
        with patch.object(handler, "open_smtp", return_value=mock_conn):
            result = handler.send_email(
                **self._creds,
                params={"to": "b@ex.com", "subject": "S", "body": "B"},
            )
        assert "error" in result
        assert "SMTP error" in result["error"]

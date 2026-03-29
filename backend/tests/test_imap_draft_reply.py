"""Unit tests for imap_draft_reply — Drafts folder discovery + draft save."""

from __future__ import annotations

from email.mime.text import MIMEText
from unittest.mock import MagicMock

import pytest


def _make():
    from capabilities.imap_capability.capability import ImapCapability
    return ImapCapability()


def _tools(cap):
    return {t["name"]: t["handler"] for t in cap.get_tools()}


# --- _find_drafts_folder ---

@pytest.mark.unit
def test_find_drafts_folder():
    """\\Drafts flag preferred; fallback to hardcoded 'Drafts'."""
    cap = _make()
    client = MagicMock()
    # Special-use flag returns actual folder name
    client.list_folders.return_value = [
        ([b"\\HasNoChildren"], b"/", "INBOX"),
        ([b"\\Drafts"], b"/", "[Gmail]/Drafts"),
    ]
    assert cap._find_drafts_folder(client) == "[Gmail]/Drafts"
    # No flag → falls back to "Drafts"
    client.list_folders.return_value = [([], b"/", "INBOX")]
    assert cap._find_drafts_folder(client) == "Drafts"


# --- _save_draft ---

@pytest.mark.unit
def test_save_draft():
    """Happy path + no-email error."""
    cap = _make()
    cap._find_drafts_folder = MagicMock(return_value="Drafts")
    client = MagicMock()
    # Happy path
    cap.load_credential = MagicMock(return_value="u@gmail.com")
    result = cap._save_draft(client, to="j@x.com", subject="Re: X",
                             body="Thanks!", in_reply_to="<abc@x.com>")
    assert result["success"] is True
    assert result["folder"] == "Drafts"
    client.append.assert_called_once()
    # No email configured
    cap.load_credential = MagicMock(return_value=None)
    assert "No email" in cap._save_draft(
        MagicMock(), to="j@x.com", subject="X", body="Y")["error"]


# --- imap_draft_reply tool ---

@pytest.mark.unit
def test_draft_reply_happy():
    cap = _make()
    hdr = MIMEText("body", "plain")
    hdr["From"] = "John <john@example.com>"
    hdr["Subject"] = "Budget Q2"
    hdr["Message-ID"] = "<orig@example.com>"
    hdr["Date"] = "Sat, 29 Mar 2026 10:00:00 +0000"
    client = MagicMock()
    client.fetch.return_value = {42: {b"RFC822.HEADER": hdr.as_bytes()}}
    cap._open_client = MagicMock(return_value=client)
    cap._save_draft = MagicMock(return_value={
        "success": True, "to": "john@example.com",
        "subject": "Re: Budget Q2", "folder": "Drafts"})
    result = _tools(cap)["imap_draft_reply"](None, {"uid": 42, "body": "OK!"})
    assert result["success"] is True
    cap._save_draft.assert_called_once()
    assert cap._save_draft.call_args[1]["in_reply_to"] == "<orig@example.com>"
    assert cap._save_draft.call_args[1]["subject"] == "Re: Budget Q2"


@pytest.mark.unit
def test_draft_reply_preserves_re_prefix():
    cap = _make()
    hdr = MIMEText("body", "plain")
    hdr["From"] = "John <john@example.com>"
    hdr["Subject"] = "Re: Budget Q2"
    hdr["Message-ID"] = "<o@x.com>"
    hdr["Date"] = "Sat, 29 Mar 2026 10:00:00 +0000"
    client = MagicMock()
    client.fetch.return_value = {7: {b"RFC822.HEADER": hdr.as_bytes()}}
    cap._open_client = MagicMock(return_value=client)
    cap._save_draft = MagicMock(return_value={"success": True})
    _tools(cap)["imap_draft_reply"](None, {"uid": 7, "body": "Got it"})
    assert cap._save_draft.call_args[1]["subject"] == "Re: Budget Q2"


@pytest.mark.unit
def test_draft_reply_errors():
    cap = _make()
    t = _tools(cap)["imap_draft_reply"]
    assert t(None, {"body": "Hi"})["error"] == "uid is required"
    assert t(None, {"uid": 42})["error"] == "body is required"
    cap._open_client = MagicMock(return_value=None)
    assert "Cannot connect" in t(None, {"uid": 1, "body": "Hi"})["error"]
    client = MagicMock()
    client.fetch.return_value = {}
    cap._open_client = MagicMock(return_value=client)
    assert "not found" in t(None, {"uid": 999, "body": "Hi"})["error"]

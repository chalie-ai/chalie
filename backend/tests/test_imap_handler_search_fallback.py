"""Unit tests for ImapHandler.search name-based fallback.

When IMAP FROM-criteria search returns 0 UIDs and the caller supplied a bare
display name (no "@"), the handler falls back to a date-windowed fetch and
post-filters in Python against From-name, From-addr, and the local-part of
the address.  These tests verify that path without a live IMAP server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from capabilities.mail_capability.imap_handler import ImapHandler, _IMAP_FETCH_HEADER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header_bytes(
    subject: str = "Hello",
    from_addr: str = "alice@example.com",
    date: str = "Mon, 01 Jan 2024 12:00:00 +0000",
    message_id: str = "<uid1@mail.example.com>",
) -> bytes:
    """Build minimal raw From-only header bytes (no display name)."""
    lines = [
        f"Subject: {subject}",
        f"From: {from_addr}",
        "To: bob@example.com",
        f"Date: {date}",
        f"Message-ID: {message_id}",
    ]
    return "\r\n".join(lines).encode()


def _make_client(
    *,
    initial_uids: list[int],
    fallback_uids: list[int] | None = None,
    raw_map: dict | None = None,
) -> MagicMock:
    """Build a fake IMAPClient with controlled search/fetch responses.

    search() returns initial_uids on the first call, then fallback_uids on the
    second call (if provided).  fetch() always returns raw_map.
    """
    client = MagicMock()
    search_responses: list[list[int]] = [initial_uids]
    if fallback_uids is not None:
        search_responses.append(fallback_uids)
    client.search.side_effect = search_responses
    client.fetch.return_value = raw_map or {}
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSearchNameFallback:
    """Verify the name post-filter fallback behaves correctly."""

    @pytest.fixture()
    def handler(self) -> ImapHandler:
        return ImapHandler()

    @pytest.fixture(autouse=True)
    def _stub_triage(self):
        """Stub classify_email so tests don't need the full ML stack."""
        with patch(
            "capabilities.mail_capability.email_triage.classify_email",
            return_value="actionable",
        ):
            yield

    # ------------------------------------------------------------------
    # Case 1: fallback fires and returns the bare-address message
    # ------------------------------------------------------------------

    def test_bare_address_matched_via_fallback(self, handler):
        uid = 7
        raw = _header_bytes(from_addr="alice@example.com", subject="Budget review")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid],
            raw_map={uid: {_IMAP_FETCH_HEADER: raw}},
        )

        result = handler.search(client, {"sender": "Alice"})

        assert result["count"] == 1
        assert result["emails"][0]["uid"] == uid
        assert result["emails"][0]["from_addr"] == "alice@example.com"
        # Two search calls: initial (FROM Alice) + fallback (SINCE …)
        assert client.search.call_count == 2

    # ------------------------------------------------------------------
    # Case 2: initial criteria search returns hits → fallback NOT taken
    # ------------------------------------------------------------------

    def test_no_fallback_when_initial_returns_results(self, handler):
        uid = 3
        raw = _header_bytes(from_addr="Alice <alice@example.com>")
        client = _make_client(
            initial_uids=[uid],
            raw_map={uid: {_IMAP_FETCH_HEADER: raw}},
        )

        result = handler.search(client, {"sender": "Alice"})

        assert result["count"] == 1
        # Only one search call — fallback was NOT triggered
        assert client.search.call_count == 1

    # ------------------------------------------------------------------
    # Case 3: sender contains "@" → fallback NOT taken even on 0 UIDs
    # ------------------------------------------------------------------

    def test_email_address_sender_skips_fallback(self, handler):
        client = _make_client(initial_uids=[])

        result = handler.search(client, {"sender": "alice@example.com"})

        assert result == {"emails": [], "count": 0}
        # Only one search call — no fallback for address-based senders
        assert client.search.call_count == 1

    # ------------------------------------------------------------------
    # Case 4: fallback + subject filter — only matching subject returned
    # ------------------------------------------------------------------

    def test_fallback_subject_filter_applied(self, handler):
        uid_match = 10
        uid_other = 11
        raw_match = _header_bytes(from_addr="alice@example.com", subject="Budget Q1", message_id="<m1@x>")
        raw_other = _header_bytes(from_addr="alice@example.com", subject="Unrelated topic", message_id="<m2@x>")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid_match, uid_other],
            raw_map={
                uid_match: {_IMAP_FETCH_HEADER: raw_match},
                uid_other: {_IMAP_FETCH_HEADER: raw_other},
            },
        )

        result = handler.search(client, {"sender": "Alice", "subject": "Budget"})

        assert result["count"] == 1
        assert result["emails"][0]["uid"] == uid_match

    # ------------------------------------------------------------------
    # Case 5: fallback candidate does NOT contain the sender substring
    # ------------------------------------------------------------------

    def test_fallback_filters_out_non_matching_sender(self, handler):
        uid = 20
        # From address has no part of "Alice" in it
        raw = _header_bytes(from_addr="bob@example.com", subject="Hello")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid],
            raw_map={uid: {_IMAP_FETCH_HEADER: raw}},
        )

        result = handler.search(client, {"sender": "Alice"})

        assert result == {"emails": [], "count": 0}

    # ------------------------------------------------------------------
    # Supplementary: local-part match (e.g. "alice" in "alice@company.com")
    # ------------------------------------------------------------------

    def test_local_part_match(self, handler):
        uid = 30
        raw = _header_bytes(from_addr="alice@company.com", subject="Proposal")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid],
            raw_map={uid: {_IMAP_FETCH_HEADER: raw}},
        )

        result = handler.search(client, {"sender": "alice"})

        assert result["count"] == 1
        assert result["emails"][0]["uid"] == uid

    # ------------------------------------------------------------------
    # Supplementary: keyword filter applied in fallback (subject match)
    # ------------------------------------------------------------------

    def test_fallback_keyword_filter_applied(self, handler):
        uid_match = 40
        uid_other = 41
        raw_match = _header_bytes(from_addr="supplier@supplierco.com", subject="Invoice Q2", message_id="<m3@x>")
        raw_other = _header_bytes(from_addr="supplier@supplierco.com", subject="Meeting notes", message_id="<m4@x>")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid_match, uid_other],
            raw_map={
                uid_match: {_IMAP_FETCH_HEADER: raw_match},
                uid_other: {_IMAP_FETCH_HEADER: raw_other},
            },
        )

        result = handler.search(client, {"sender": "SupplierCo", "keyword": "Invoice"})

        assert result["count"] == 1
        assert result["emails"][0]["uid"] == uid_match

    # ------------------------------------------------------------------
    # Supplementary: fetch is NOT called twice when fallback fires
    # ------------------------------------------------------------------

    def test_prefetched_raw_reused_no_double_fetch(self, handler):
        uid = 50
        raw = _header_bytes(from_addr="alice@example.com")
        client = _make_client(
            initial_uids=[],
            fallback_uids=[uid],
            raw_map={uid: {_IMAP_FETCH_HEADER: raw}},
        )

        handler.search(client, {"sender": "Alice"})

        # fetch() should be called exactly once (the fallback fetch is reused)
        assert client.fetch.call_count == 1

    # ------------------------------------------------------------------
    # Supplementary: limit clamp must apply on the prefetched path too.
    # Without this, fallback would silently exceed the caller's limit
    # because prefetched is built before the slice.
    # ------------------------------------------------------------------

    def test_fallback_respects_limit_clamp(self, handler):
        # 5 alice@example.com candidates, limit=2 → only 2 returned
        uids = [100, 101, 102, 103, 104]
        raw_map = {
            u: {_IMAP_FETCH_HEADER: _header_bytes(
                from_addr="alice@example.com",
                subject=f"msg-{u}",
                message_id=f"<m{u}@x>",
            )}
            for u in uids
        }
        client = _make_client(
            initial_uids=[],
            fallback_uids=uids,
            raw_map=raw_map,
        )

        result = handler.search(client, {"sender": "Alice", "limit": 2})

        assert result["count"] == 2

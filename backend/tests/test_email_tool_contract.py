"""Feature tests for the email tool's model-facing result contract.

Real-world assertions only — no mocks of in-process production code. The
shaping layer (``abilities.email._shape_result``) is exercised directly with
handler-shaped dicts, and the ``emails_sent`` ledger (``models.email_sent``)
runs against a fully-converged real SQLite database (the ``db`` fixture), so
every assertion covers the same code the dispatcher runs in production.

What the contract guarantees:
  1. search rows carry exactly id / from / subject / date — no ``snippet``
     (a dead field that was always empty) and no ``message_id`` leak.
  2. search echoes the effective query criteria; an empty result is a loud
     ``code=no-results`` error naming those criteria, never a quiet success.
  3. read returns the plaintext body ONLY, truncated at 20k with the shared
     ``truncated`` meta — no re-sent uid/from/subject/date metadata.
  4. send/reply/forward envelopes carry to/subject only; the stamped
     Message-ID is ledger bookkeeping, never part of the model-facing body.
  5. rows/reads whose Message-ID is in the ``emails_sent`` ledger are
     annotated as Chalie's own outgoing mail, so the model recognizes its own
     sends when they echo back into the inbox.
  6. download returns the files list (no uid echo) and a meta count when
     attachments are present; an empty files list is a loud
     ``code=no-results`` error whose hint names attachments.
  7. read surfaces attachments meta only when the handler body carries them;
     send echoes attachments only when the handler body carries them.
  8. search rows carry ``has_attachments`` only when truthy — a false or
     absent key stays out of the shaped row.
"""

import sqlite3
from typing import cast

import pytest

from abilities.email import _READ_BODY_LIMIT, _REAL_SENDER_NOTE, _record_sent, _shape_result
from models.email_sent import EmailSent

pytestmark = pytest.mark.unit


def _handler_row(uid: int, message_id: str = "") -> dict[str, object]:
    """A search row shaped exactly as ImapHandler.search emits it."""
    row: dict[str, object] = {
        "uid": uid,
        "from_addr": "owner@example.com",
        "from_name": "Alex",
        "subject": f"Subject {uid}",
        "date": "2026-08-01T11:33:00-07:00",
    }
    if message_id:
        row["message_id"] = message_id
    return row


# ---------------------------------------------------------------------------
# 1+2. Search shape — lean rows, criteria echo, loud empty
# ---------------------------------------------------------------------------


def test_search_rows_are_lean_and_echo_the_query(db: sqlite3.Connection) -> None:
    body: dict[str, object] = {
        "emails": [_handler_row(101), _handler_row(102)],
        "query": {"sender": "owner@example.com", "date_from": "2026-07-25"},
    }
    result = _shape_result("search", body)

    assert result.status == "success"
    assert result.meta["count"] == 2
    assert isinstance(result.body, dict)
    rows = cast(list[dict[str, object]], result.body["emails"])
    assert [set(r) for r in rows] == [{"id", "from", "subject", "date"}] * 2
    assert rows[0] == {
        "id": 101,
        "from": "Alex",
        "subject": "Subject 101",
        "date": "2026-08-01T11:33:00-07:00",
    }
    # The criteria that produced the rows travel with them.
    assert result.body["query"] == {
        "sender": "owner@example.com",
        "date_from": "2026-07-25",
    }


def test_search_empty_is_a_loud_no_results_error_naming_the_criteria(
    db: sqlite3.Connection,
) -> None:
    result = _shape_result("search", {"emails": [], "query": {"sender": "nobody@x.com"}})

    assert result.status == "error"
    assert result.code == "no-results"
    assert "nobody@x.com" in (result.hint or "")


# ---------------------------------------------------------------------------
# 3. Read shape — body only, 20k cap
# ---------------------------------------------------------------------------


def test_read_returns_body_only(db: sqlite3.Connection) -> None:
    body = {
        "uid": 101,
        "from_addr": "owner@example.com",
        "subject": "Subject 101",
        "date": "2026-08-01T11:33:00-07:00",
        "body": "plain text body",
        "message_id": "<not-in-ledger@example.com>",
    }
    result = _shape_result("read", body)

    assert result.status == "success"
    assert result.body == "plain text body"  # nothing re-sent around it
    assert result.meta == {"action": "read", "truncated": False}


def test_read_clips_at_20k_and_reports_the_truncation(db: sqlite3.Connection) -> None:
    result = _shape_result("read", {"body": "x" * (_READ_BODY_LIMIT + 1)})

    assert isinstance(result.body, str)
    assert len(result.body) == _READ_BODY_LIMIT == 20_000
    assert result.meta["truncated"] is True


# ---------------------------------------------------------------------------
# 4. Send/reply/forward envelope — the stamp stays internal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["send", "reply", "forward"])
def test_send_envelope_never_exposes_the_message_id(action: str) -> None:
    result = _shape_result(
        action,
        {
            "success": True,
            "to": "someone@example.com",
            "subject": "Hello",
            "message_id": "<stamp@local>",
        },
    )

    assert result.status == "success"
    assert result.body == {"to": "someone@example.com", "subject": "Hello"}


# ---------------------------------------------------------------------------
# 6+7. Download shape — files list, empty = loud error
# ---------------------------------------------------------------------------


def test_download_returns_files_and_counts(db: sqlite3.Connection) -> None:
    result = _shape_result(
        "download",
        {
            "files": [
                {
                    "path": "/tmp/chalie_email_5_ab12cd34/report.pdf",
                    "filename": "report.pdf",
                    "size": 3,
                    "mime": "application/pdf",
                }
            ],
            "uid": 5,
            "count": 1,
        },
    )

    assert result.status == "success"
    assert result.body == {
        "files": [
            {
                "path": "/tmp/chalie_email_5_ab12cd34/report.pdf",
                "filename": "report.pdf",
                "size": 3,
                "mime": "application/pdf",
            }
        ]
    }
    assert result.meta["count"] == 1
    # uid is ledger bookkeeping, never echoed.
    assert "uid" not in result.body


def test_download_empty_is_no_results_error(db: sqlite3.Connection) -> None:
    result = _shape_result("download", {"files": [], "uid": 5, "count": 0})

    assert result.status == "error"
    assert result.code == "no-results"
    assert "attachments" in (result.hint or "")


# ---------------------------------------------------------------------------
# 7. Send echo and read meta — attachments only when carried
# ---------------------------------------------------------------------------


def test_send_echo_includes_attachments_when_present(db: sqlite3.Connection) -> None:
    result = _shape_result(
        "send",
        {
            "success": True,
            "to": "someone@example.com",
            "subject": "Hello",
            "message_id": "<stamp@local>",
            "attachments": ["report.pdf"],
        },
    )

    assert result.status == "success"
    assert result.body == {
        "to": "someone@example.com",
        "subject": "Hello",
        "attachments": ["report.pdf"],
    }
    # Message-ID stays ledger-only even when attachments are present.
    assert "message_id" not in result.body


def test_read_surfaces_attachments_in_body_when_non_empty(db: sqlite3.Connection) -> None:
    result = _shape_result(
        "read",
        {
            "body": "hi",
            "attachments": [
                {"filename": "a.png", "size": 10, "mime": "image/png"}
            ],
        },
    )

    assert isinstance(result.body, dict)
    assert result.body["attachments"] == [
        {"filename": "a.png", "size": 10, "mime": "image/png"}
    ]
    assert result.body["body"] == "hi"
    assert result.meta["truncated"] is False


def test_read_body_is_plain_string_when_no_attachments(db: sqlite3.Connection) -> None:
    result = _shape_result("read", {"body": "hi"})

    assert result.body == "hi"
    assert "attachments" not in result.meta


# ---------------------------------------------------------------------------
# 8. Search rows — has_attachments only when truthy
# ---------------------------------------------------------------------------


def test_search_rows_carry_has_attachments_only_when_true(db: sqlite3.Connection) -> None:
    body: dict[str, object] = {
        "emails": [
            {
                "uid": 1,
                "from_addr": "a@b.com",
                "from_name": "A",
                "subject": "S1",
                "date": "2026-01-01",
                "has_attachments": True,
            },
            {
                "uid": 2,
                "from_addr": "a@b.com",
                "from_name": "A",
                "subject": "S2",
                "date": "2026-01-01",
                "has_attachments": False,
            },
            {
                "uid": 3,
                "from_addr": "a@b.com",
                "from_name": "A",
                "subject": "S3",
                "date": "2026-01-01",
            },
        ],
        "query": {},
    }
    result = _shape_result("search", body)

    assert isinstance(result.body, dict)
    rows = cast(list[dict[str, object]], result.body["emails"])
    assert set(rows[0]) == {"id", "from", "subject", "date", "has_attachments"}
    assert rows[0]["has_attachments"] is True
    assert set(rows[1]) == {"id", "from", "subject", "date"}
    assert set(rows[2]) == {"id", "from", "subject", "date"}


# ---------------------------------------------------------------------------
# 5. emails_sent ledger — record, match, annotate
# ---------------------------------------------------------------------------


def test_record_is_idempotent_and_stamps_the_send_time(db: sqlite3.Connection) -> None:
    EmailSent.record("<stamp@local>")
    EmailSent.record("<stamp@local>")

    rows = db.execute("SELECT key, value FROM emails_sent").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "<stamp@local>"
    assert rows[0][1]  # send timestamp filled by the column default


def test_sent_ids_returns_the_matching_subset(db: sqlite3.Connection) -> None:
    EmailSent.record("<a@local>")
    EmailSent.record("<b@local>")

    assert EmailSent.sent_ids(["<a@local>", "<c@local>"]) == {"<a@local>"}
    assert EmailSent.sent_ids([]) == set()
    # Generators are materialized before use, not consumed twice.
    assert EmailSent.sent_ids(m for m in ["<b@local>"]) == {"<b@local>"}


def test_record_sent_writes_the_ledger_from_a_send_body(db: sqlite3.Connection) -> None:
    _record_sent({"success": True, "to": "x@y.com", "subject": "s", "message_id": "<s@l>"})
    _record_sent({"success": True, "to": "x@y.com", "subject": "s"})  # no id → no row

    keys = [r[0] for r in db.execute("SELECT key FROM emails_sent").fetchall()]
    assert keys == ["<s@l>"]


def test_own_sends_are_annotated_in_search_rows(db: sqlite3.Connection) -> None:
    EmailSent.record("<mine@local>")
    body: dict[str, object] = {
        "emails": [
            _handler_row(101, message_id="<mine@local>"),
            _handler_row(102, message_id="<theirs@remote>"),
        ],
        "query": {"sender": "owner@example.com"},
    }
    result = _shape_result("search", body)

    assert isinstance(result.body, dict)
    mine, theirs = cast(list[dict[str, object]], result.body["emails"])
    assert mine["real-sender"] == _REAL_SENDER_NOTE
    assert "real-sender" not in theirs


def test_own_send_is_annotated_on_read(db: sqlite3.Connection) -> None:
    EmailSent.record("<mine@local>")

    annotated = _shape_result("read", {"body": "hi", "message_id": "<mine@local>"})
    plain = _shape_result("read", {"body": "hi", "message_id": "<theirs@remote>"})

    assert annotated.meta["real_sender"] == _REAL_SENDER_NOTE
    assert "real_sender" not in plain.meta

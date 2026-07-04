# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Calendar-specific business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
calendar ability's genuine behaviour tests (default window, fuzzy matching,
dtstart validation, ambiguous-title errors) that have no coverage elsewhere."""

import json
import sqlite3
from datetime import timedelta, datetime
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.time_utils import utc_now
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="what's on my calendar")


def _allow_calendar_writes(db: sqlite3.Connection, channel: str = "chat") -> None:
    """Set the REAL ``policy`` table so calendar write actions are ``allow`` on the
    channel — the same row the real ``PolicyManager`` gate reads.

    Write actions ship as ``ask`` by seed (mutating CalDAV is risky), which on a
    headless test parks the real gate waiting for a human POST. Flipping the real
    policy row to ``allow`` (exactly what a user does when they pick "always
    allow") lets the gate pass so the production ``run()`` — and its dtstart
    validation / fuzzy-match / not-connected paths — actually executes. No mock:
    this is the production policy table driving the production gate."""
    for action in ("create_event", "update_event", "delete_event"):
        allow_policy(db, f"calendar.{action}", channel=channel)


def _seed_event(
    db: sqlite3.Connection,
    *,
    uid: str,
    title: str,
    due_at: datetime,
    dtend: datetime | None = None,
    location: str | None = None,
    calendar_name: str = "Personal",
) -> None:
    """Insert a calendar event into ``scheduled_items`` exactly as the CalDAV
    ingest does: source='mail', item_type='event', hidden=1, external_uid keyed
    'caldav:<uid>', and the event detail JSON in ``metadata``."""
    import uuid as _uuid

    meta: dict[str, object] = {
        "uid": uid,
        "dtend": dtend.isoformat() if dtend else None,
        "location": location,
        "attendees": [],
        "all_day": False,
        "calendar_name": calendar_name,
    }
    db.execute(
        """
        INSERT INTO scheduled_items
          (id, item_type, message, due_at, status, source, external_uid,
           metadata, hidden, created_at)
        VALUES (?, 'event', ?, ?, 'pending', 'mail', ?, ?, 1, datetime('now'))
        """,
        (
            _uuid.uuid4().hex[:8],
            title,
            due_at.isoformat(),
            f"caldav:{uid}",
            json.dumps(meta),
        ),
    )
    db.commit()


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> _MP:
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and calendar write actions flipped to ``allow``
    in the real policy table so the gate passes through to the production run()."""
    _allow_calendar_writes(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "calendar") -> dict[str, object]:
    """Extract and JSON-parse the rich-card head between the open tag and
    ``[end:<tool>]`` (the JSON before the blank-line span trailer)."""
    return cast(dict[str, object], parse_body(rendered, tool, rich=True))


def test_list_events_default_window_is_today_plus_seven_days(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """``list_events`` with NO ``date_from``/``date_to`` applies the advertised
    default window (today → +7 days): an event 2 days out is returned, one 30 days
    out is excluded — the schema text no longer lies."""
    _seed_event(db, uid="ev-soon", title="Standup", due_at=utc_now() + timedelta(days=2))
    _seed_event(db, uid="ev-far", title="Conference", due_at=utc_now() + timedelta(days=30))

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "list_events", "act_summary": "x"}
    )

    assert "[calendar(status=success" in out
    body = _parse_body(out)
    titles = {ev["title"] for ev in cast(list[dict[str, object]], body["events"])}
    assert "Standup" in titles
    assert "Conference" not in titles


def test_list_events_today_includes_timed_events_with_bare_date_bounds(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """"What's on my calendar today?" sends a BARE-date upper bound
    (date_from == date_to == today's YYYY-MM-DD, exactly as the schema instructs).
    A stored event's ``due_at`` carries a time-of-day, so the lexical
    ``due_at <= '<today>'`` compare in query_items sorted the timestamped value
    AFTER the bare date and silently dropped every event on that day. list_events
    must widen the bare upper bound to end-of-day and return the event."""
    today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    _seed_event(db, uid="ev-today", title="Morning standup", due_at=today + timedelta(hours=8))

    bare = today.date().isoformat()
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar",
        {"action": "list_events", "date_from": bare, "date_to": bare, "act_summary": "x"},
    )

    assert "[calendar(status=success" in out
    titles = {ev["title"] for ev in cast(list[dict[str, object]], _parse_body(out)["events"])}
    assert "Morning standup" in titles


def test_get_event_by_uid_renders_the_event(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """``get_event`` by CalDAV uid renders the single event record."""
    _seed_event(
        db, uid="dentist-123", title="Dentist appointment",
        due_at=utc_now() + timedelta(days=3), location="Sliema Clinic",
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "uid": "dentist-123", "act_summary": "x"}
    )

    assert "[calendar(status=success" in out
    payload = _parse_body(out)
    event = cast(dict[str, object], payload["event"])
    assert event["uid"] == "dentist-123"
    assert event["title"] == "Dentist appointment"
    assert event["location"] == "Sliema Clinic"


def test_get_event_by_title_resolves_single_match(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """``get_event`` with a ``title`` fuzzy-matches a single event without forcing
    a list-then-act round-trip."""
    _seed_event(db, uid="yoga-1", title="Yoga class", due_at=utc_now() + timedelta(days=1))

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "title": "yoga", "act_summary": "x"}
    )

    assert "[calendar(status=success" in out
    payload = _parse_body(out)
    assert cast(dict[str, object], payload["event"])["uid"] == "yoga-1"


def test_get_event_unknown_title_errors_not_found(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A ``title`` matching nothing errors with ``code=not-found`` (NOT the
    placeholder) and a hint pointing at list_events."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "title": "nonexistent", "act_summary": "x"}
    )

    assert "[calendar(status=error, code=not-found" in out
    assert "code=error]" not in out
    assert "hint:" in out


def test_ambiguous_title_errors_with_candidate_ids(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """``delete_event title='meeting'`` with 2 matches returns
    ``code=ambiguous-match`` and lists the candidate uids in the body — never a
    silent first-hit pick, and NO CalDAV write."""
    _seed_event(db, uid="mtg-a", title="Team meeting", due_at=utc_now() + timedelta(days=1))
    _seed_event(db, uid="mtg-b", title="Budget meeting", due_at=utc_now() + timedelta(days=2))

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "delete_event", "title": "meeting", "act_summary": "x"}
    )

    assert "[calendar(status=error, code=ambiguous-match" in out
    assert "code=error]" not in out
    # Both candidate uids are surfaced so the model can pick one.
    assert "mtg-a" in out
    assert "mtg-b" in out


def test_update_with_malformed_dtstart_errors_invalid_time(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A malformed ``dtstart`` on ``update_event`` errors with ``code=invalid-time``
    and example forms — BEFORE the CalDAV write, so the year-0001 ``datetime.min``
    sentinel never reaches the server. The error fires regardless of whether the
    mail capability is connected (validation precedes the connected gate)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar",
        {"action": "update_event", "uid": "ev-x",
         "dtstart": "blorptdfgh nonsense", "act_summary": "x"},
    )

    assert "[calendar(status=error, code=invalid-time" in out
    assert "code=error]" not in out
    assert "hint:" in out
    # The year-0001 sentinel string never appears anywhere in the rendered output.
    assert "0001-" not in out


def test_create_with_malformed_dtstart_errors_invalid_time(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A malformed ``dtstart`` on ``create_event`` likewise errors with a stable
    ``code=invalid-time`` before any write."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar",
        {"action": "create_event", "summary": "New thing",
         "dtstart": "not a date at all", "dtend": "tomorrow 4pm",
         "act_summary": "x"},
    )

    assert "[calendar(status=error, code=invalid-time" in out
    assert "code=error]" not in out
    assert "0001-" not in out

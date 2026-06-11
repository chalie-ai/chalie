# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the calendar tool's ToolResult contract (TKT-888).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``CalendarAbility`` (now a ``CapabilityAbility`` subclass), the real
``PolicyManager.wrap`` gate, the real ``CalendarAbility.run``, the real
``scheduled_items`` SQLite reads (the ``db`` fixture binds the singletons to a
real SQLite database — calendar events are mirrored there by the CalDAV ingest),
and the real ``ActTrail`` write.

What TKT-888 changes, exercised end to end:

* **Data-corruption fix:** a malformed ``dtstart`` on a write action errors with a
  STABLE ``code=invalid-time`` BEFORE any CalDAV write — never the ``parse_utc``
  ``datetime.min`` year-0001 sentinel reaching the server. Validation runs ahead
  of the capability-connected gate, so the loud error fires regardless of whether
  CalDAV is wired up.
* **Schema honesty:** ``list_events`` with no ``date_from``/``date_to`` applies the
  advertised default window (today → +7 days) instead of silently ignoring the
  documented defaults.
* **Addressing automation:** write/read actions accept a ``title`` for fuzzy
  matching; >1 match returns ``code=ambiguous-match`` with the candidate ids in
  the body — never a silent first-hit pick.
* **Foundation:** the ability is a ``CapabilityAbility`` subclass — the
  not-connected / unknown-action / handler-dispatch flow comes from the base.
* **Contract + rich:** every action returns a ``ToolResult``; ``list_events`` /
  ``get_event`` return JSON rows and pair a rich card via ``ToolResult(rich=…)``
  (the dispatcher owns the ordinal + the single span instruction); errors carry
  stable kebab codes (NOT the ``code="error"`` placeholder) + hints.

Calendar events live in ``scheduled_items`` (source='mail', item_type='event',
hidden=1, external_uid='caldav:<uid>') — the same rows the CalDAV ingest writes.
Tests seed those rows the production way and read them back through the ability.
"""

import json
from datetime import timedelta

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.time_utils import utc_now
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="what's on my calendar")


def _allow_calendar_writes(db, channel: str = "chat") -> None:
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
    db,
    *,
    uid: str,
    title: str,
    due_at,
    dtend=None,
    location: str | None = None,
    calendar_name: str = "Personal",
) -> None:
    """Insert a calendar event into ``scheduled_items`` exactly as the CalDAV
    ingest does: source='mail', item_type='event', hidden=1, external_uid keyed
    'caldav:<uid>', and the event detail JSON in ``metadata``."""
    import uuid as _uuid

    meta = {
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
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and calendar write actions flipped to ``allow``
    in the real policy table so the gate passes through to the production run()."""
    _allow_calendar_writes(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "calendar") -> object:
    """Extract and JSON-parse the rich-card head between the open tag and
    ``[end:<tool>]`` (the JSON before the blank-line span trailer)."""
    return parse_body(rendered, tool, rich=True)


# ── Foundation: CalendarAbility is a CapabilityAbility ──────────────────────────


def test_calendar_is_a_capability_ability():
    """The ability is migrated onto the shared ``CapabilityAbility`` base — the
    copy-pasted capability-delegation block is gone."""
    from abilities._capability import CapabilityAbility
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("calendar")
    assert isinstance(ability, CapabilityAbility)
    assert ability.CAPABILITY_KEY == "mail"


# ── Schema honesty + trimmed surface ────────────────────────────────────────────


def test_param_surface_is_trimmed():
    """The 11-param surface is reduced — ``calendar_name`` (an unused filter) is
    gone and the new ``title`` fuzzy-match param is present."""
    from abilities._registry import AbilityRegistry

    schema = AbilityRegistry.get("calendar").get_parameters()
    props = schema["properties"]
    assert "title" in props
    assert "calendar_name" not in props
    assert len(props) <= 9


# ── Read: list_events applies the advertised default window ─────────────────────


def test_list_events_default_window_is_today_plus_seven_days(db, chat_mp):
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
    titles = {ev["title"] for ev in body["events"]}
    assert "Standup" in titles
    assert "Conference" not in titles


def test_list_events_renders_rich_card_on_user_broadcast(db, chat_mp):
    """On a user-broadcasting channel ``list_events`` pairs a rich card: the
    dispatcher injects the ordinal-keyed span instruction and the card payload
    (JSON head before the blank line) carries the events + action_performed."""
    assert getattr(chat_mp.config, "broadcast_to", None) == "user"
    _seed_event(db, uid="ev-1", title="Lunch", due_at=utc_now() + timedelta(days=1))

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "list_events", "act_summary": "x"}
    )

    assert "[calendar(status=success" in out
    assert "<span id='calendar_1'>" in out
    payload = _parse_body(out)
    assert payload["action_performed"] == "list_events"
    assert any(ev["title"] == "Lunch" for ev in payload["events"])


# ── Read: get_event by uid and by fuzzy title ───────────────────────────────────


def test_get_event_by_uid_renders_the_event(db, chat_mp):
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
    event = payload["event"]
    assert event["uid"] == "dentist-123"
    assert event["title"] == "Dentist appointment"
    assert event["location"] == "Sliema Clinic"


def test_get_event_by_title_resolves_single_match(db, chat_mp):
    """``get_event`` with a ``title`` fuzzy-matches a single event without forcing
    a list-then-act round-trip."""
    _seed_event(db, uid="yoga-1", title="Yoga class", due_at=utc_now() + timedelta(days=1))

    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "title": "yoga", "act_summary": "x"}
    )

    assert "[calendar(status=success" in out
    payload = _parse_body(out)
    assert payload["event"]["uid"] == "yoga-1"


def test_get_event_unknown_title_errors_not_found(db, chat_mp):
    """A ``title`` matching nothing errors with ``code=not-found`` (NOT the
    placeholder) and a hint pointing at list_events."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "title": "nonexistent", "act_summary": "x"}
    )

    assert "[calendar(status=error, code=not-found" in out
    assert "code=error]" not in out
    assert "hint:" in out


# ── Addressing automation: ambiguous title → disambiguation error ───────────────


def test_ambiguous_title_errors_with_candidate_ids(db, chat_mp):
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


# ── Data-corruption fix: malformed dtstart → invalid-time, never a sentinel ─────


def test_update_with_malformed_dtstart_errors_invalid_time(db, chat_mp):
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


def test_create_with_malformed_dtstart_errors_invalid_time(db, chat_mp):
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


# ── Write actions on a NOT-connected capability → stable not-connected error ────


def test_update_event_valid_dtstart_not_connected_errors_cleanly(db, chat_mp):
    """With a well-formed (natural-language) ``dtstart`` but no connected mail
    capability, ``update_event`` passes validation and then surfaces the base's
    stable ``code=not-connected`` error — NOT a sentinel write, NOT a placeholder.
    """
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar",
        {"action": "update_event", "uid": "ev-x",
         "dtstart": "tomorrow 3pm", "act_summary": "x"},
    )

    # mail is not connected in the test environment → base not-connected error.
    assert "[calendar(status=error, code=not-connected" in out
    assert "code=error]" not in out
    assert "0001-" not in out


# ── Error ladders for malformed input (ACTION_REQUIRED pre-gate) ────────────────


def test_unknown_action_lists_real_actions_in_valid(db, chat_mp):
    """An unknown action is pre-gated into ONE ``code=unknown-action`` error whose
    ``valid:`` line names the REAL actions."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "teleport", "act_summary": "x"}
    )

    assert "[calendar(status=error, code=unknown-action" in out
    assert "valid:" in out
    for action in ("list_events", "get_event", "create_event", "update_event", "delete_event"):
        assert action in out


def test_get_event_without_target_reports_missing_params(db, chat_mp):
    """``get_event`` with neither ``uid`` nor ``title`` errors with a stable code
    (NOT the placeholder) and a hint — never an empty or sentinel result."""
    out = ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "get_event", "act_summary": "x"}
    )

    assert "[calendar(status=error" in out
    assert "code=error]" not in out
    assert "hint:" in out


# ── Act-trail records the rendered envelope ─────────────────────────────────────


def test_act_trail_records_the_envelope(db, chat_mp):
    """The act-trail records the same non-empty calendar envelope against the
    transcript anchor — the cross-step write really lands."""
    _seed_event(db, uid="ev-trail", title="Sync", due_at=utc_now() + timedelta(days=1))
    ToolDispatcher(chat_mp).dispatch(
        "calendar", {"action": "list_events", "act_summary": "x"}
    )
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("[calendar(status=" in row["result"] for row in trail)

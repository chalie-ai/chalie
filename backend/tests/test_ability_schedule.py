# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""schedule-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975. Covers natural-language due_at resolution,
relative due_at, ISO due_at back-compat, unparseable due_at errors, past due_at
rejection, schema-declared limit, structured list rows, cancel by message,
cancel-without-target error, and invalid recurrence error ladder.
"""

import json
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.time_utils import utc_now
from tests._tool_result_harness import MP, parse_body, seed_transcript

pytestmark = pytest.mark.unit

_TZ = "Europe/Malta"


def _seed_timezone(db, tz_name: str = _TZ) -> None:
    """Write a real client heartbeat timezone into the telemetry table — the same
    store the production ``locale_service.get_timezone()`` reads. No mock."""
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.execute(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        ("timezone", json.dumps(tz_name)),
    )
    db.commit()
    heartbeat_service._ctx = None


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and a seeded user timezone."""
    _seed_timezone(db)
    return MP(seed_transcript(db, "chat", "remind me to do a thing"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "schedule") -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:<tool>]``.

    On a user-broadcasting channel the dispatcher pairs a rich card: the body is
    ``<card_json>\\n\\n<span instruction>``. Split on the blank line and parse the
    JSON head so the same helper works for both plain and card-paired bodies."""
    return parse_body(rendered, tool, rich=True)


def _row(db, item_id: str) -> dict | None:
    cur = db.execute(
        "SELECT id, message, due_at, status, recurrence FROM scheduled_items WHERE id = ?",
        (item_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {"id": r[0], "message": r[1], "due_at": r[2], "status": r[3], "recurrence": r[4]}


# ── Natural-language due_at: resolved in the user's tz, echoed back ─────────────


def test_create_with_natural_language_due_at_resolves_in_user_tz(db, chat_mp):
    """``due_at='tomorrow 9am'`` resolves to 09:00 in the user's seeded timezone,
    persists the UTC instant to the row, and echoes BOTH the resolved UTC and
    local strings in the success body so the model can confirm to the user."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Call the dentist",
         "due_at": "tomorrow 9am", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    assert "[end:schedule]" in out
    card = _parse_body(out)
    assert card["action_performed"] == "create"
    record = card["record"]

    # Resolved instants are echoed back, both UTC and local.
    assert "due_at_utc" in record
    assert "due_at_local" in record
    # 09:00 in the user's local tz.
    assert "09:00" in record["due_at_local"]

    # The persisted row carries a real future UTC instant — NOT datetime.min.
    persisted = _row(db, record["id"])
    assert persisted is not None
    assert persisted["status"] == "pending"
    assert not persisted["due_at"].startswith("0001-")
    from services.time_utils import parse_utc
    assert parse_utc(persisted["due_at"]) > utc_now()


def test_create_with_relative_due_at_in_two_hours(db, chat_mp):
    """``due_at='in 2 hours'`` resolves to a future instant ~2h out in the user's
    tz, persisted to the row."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Stretch break",
         "due_at": "in 2 hours", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    record = _parse_body(out)["record"]
    from services.time_utils import parse_utc
    due = parse_utc(_row(db, record["id"])["due_at"])
    delta = (due - utc_now()).total_seconds()
    assert 6900 < delta < 7500  # ~2h, allowing a little slack


def test_create_with_iso_due_at_still_accepted(db, chat_mp):
    """A hand-authored ISO 8601 with offset still parses (back-compat)."""
    future = (utc_now() + timedelta(days=3)).astimezone(ZoneInfo(_TZ))
    iso = future.replace(microsecond=0).isoformat()
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Quarterly report",
         "due_at": iso, "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    record = _parse_body(out)["record"]
    from services.time_utils import parse_utc
    persisted = parse_utc(_row(db, record["id"])["due_at"])
    assert abs((persisted - future).total_seconds()) < 2


def test_unparseable_due_at_errors_invalid_time_and_persists_nothing(db, chat_mp):
    """An unparseable ``due_at`` errors with ``code=invalid-time`` and a ``hint:``
    of example forms — and NEVER writes a row (no datetime.min sentinel)."""
    before = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]

    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Mystery",
         "due_at": "blorptdfgh nonsense", "act_summary": "x"},
    )

    assert "[schedule(status=error, code=invalid-time" in out
    assert "code=error]" not in out
    assert "hint:" in out
    # Never persisted, never a datetime.min sentinel row.
    after = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
    assert after == before
    assert db.execute(
        "SELECT COUNT(*) FROM scheduled_items WHERE due_at LIKE '0001-%'"
    ).fetchone()[0] == 0


def test_past_due_at_errors_due_in_past(db, chat_mp):
    """A clearly-past ``due_at`` (well beyond the grace window) errors with a
    stable ``code=due-in-past`` rather than silently bumping or persisting."""
    past = (utc_now() - timedelta(days=2)).astimezone(ZoneInfo(_TZ))
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Yesterday's thing",
         "due_at": past.replace(microsecond=0).isoformat(), "act_summary": "x"},
    )

    assert "[schedule(status=error, code=due-in-past" in out
    assert "code=error]" not in out


# ── Schema honesty: declared, clamped limit on search ──────────────────────────


def test_limit_is_declared_in_the_schema():
    """The ``limit`` param the code reads is now DECLARED in the tool schema —
    schema and code agree."""
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("schedule")
    schema = ability.get_parameters()
    assert "limit" in schema["properties"]
    assert schema["properties"]["limit"]["type"] == "integer"


# ── list / cancel happy paths render structured, parseable bodies ──────────────


def test_list_renders_structured_rows(db, chat_mp):
    """``list`` renders a success envelope whose JSON body carries the pending
    items (id + message + due_at + status)."""
    ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Water the plants",
         "due_at": "tomorrow 8am", "act_summary": "x"},
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "schedule", {"action": "list", "act_summary": "x"}
    )

    assert "[schedule(status=success" in out
    b = _parse_body(out)
    records = b["records"]
    assert any(r["message"] == "Water the plants" for r in records)
    assert all("due_at_utc" in r and "status" in r for r in records)


def test_cancel_by_message_removes_row(db, chat_mp):
    """``cancel`` with a fuzzy message match flips the row to cancelled and renders
    a success confirmation."""
    create = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Dentist appointment",
         "due_at": "tomorrow 3pm", "act_summary": "x"},
    )
    item_id = _parse_body(create)["record"]["id"]

    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "cancel", "message": "Dentist", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    assert _row(db, item_id)["status"] == "cancelled"


def test_cancel_without_target_errors_with_stable_code(db, chat_mp):
    """``cancel`` with neither item_id nor message errors with a stable kebab-case
    code (NOT the placeholder) and a hint."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule", {"action": "cancel", "act_summary": "x"}
    )

    assert "[schedule(status=error" in out
    assert "code=error]" not in out
    assert "hint:" in out


# ── Error ladders for malformed input ──────────────────────────────────────────


def test_invalid_recurrence_errors_with_valid_ladder(db, chat_mp):
    """A bad ``recurrence`` errors with a stable code and a ``valid:`` ladder of
    the accepted recurrence keywords."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Standup",
         "due_at": "tomorrow 9am", "recurrence": "fortnightly", "act_summary": "x"},
    )

    assert "[schedule(status=error" in out
    assert "code=error]" not in out
    assert "valid:" in out

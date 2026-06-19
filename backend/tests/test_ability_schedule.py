# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import sqlite3
from datetime import timedelta
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.time_utils import utc_now
from tests._tool_result_harness import MP, parse_body, seed_transcript

pytestmark = pytest.mark.unit

_TZ = "Europe/Malta"


def _seed_timezone(db: sqlite3.Connection, tz_name: str = _TZ) -> None:
    """Write a real heartbeat timezone into the telemetry table — the same store
    the production ``locale_service.get_timezone()`` reads."""
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
def chat_mp(db: sqlite3.Connection) -> MP:
    _seed_timezone(db)
    return MP(seed_transcript(db, "chat", "remind me to do a thing"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "schedule") -> dict[str, object]:
    return cast("dict[str, object]", parse_body(rendered, tool, rich=True))


def _row(db: sqlite3.Connection, item_id: str) -> "dict[str, object] | None":
    cur = db.execute(
        "SELECT id, message, due_at, status, recurrence FROM scheduled_items WHERE id = ?",
        (item_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {"id": r[0], "message": r[1], "due_at": r[2], "status": r[3], "recurrence": r[4]}


# ── Natural-language due_at: resolved in the user's tz, echoed back ─────────────


def test_create_with_natural_language_due_at_resolves_in_user_tz(db: sqlite3.Connection, chat_mp: MP) -> None:
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Call the dentist",
         "due_at": "tomorrow 9am", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    assert "[end:schedule]" in out
    card = _parse_body(out)
    assert card["action_performed"] == "create"
    record = cast("dict[str, object]", card["record"])

    # Resolved instants are echoed back, both UTC and local.
    assert "due_at_utc" in record
    assert "due_at_local" in record
    # 09:00 in the user's local tz.
    assert "09:00" in cast(str, record["due_at_local"])

    # The persisted row carries a real future UTC instant — NOT datetime.min.
    persisted = _row(db, cast(str, record["id"]))
    assert persisted is not None
    assert persisted["status"] == "pending"
    assert not cast(str, persisted["due_at"]).startswith("0001-")
    from services.time_utils import parse_utc
    assert parse_utc(cast(str, persisted["due_at"])) > utc_now()


def test_create_with_relative_due_at_in_two_hours(db: sqlite3.Connection, chat_mp: MP) -> None:
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Stretch break",
         "due_at": "in 2 hours", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    record = cast("dict[str, object]", _parse_body(out)["record"])
    from services.time_utils import parse_utc
    due = parse_utc(cast(str, cast("dict[str, object]", _row(db, cast(str, record["id"])))["due_at"]))
    delta = (due - utc_now()).total_seconds()
    assert 6900 < delta < 7500  # ~2h, allowing a little slack


def test_create_with_iso_due_at_still_accepted(db: sqlite3.Connection, chat_mp: MP) -> None:
    future = (utc_now() + timedelta(days=3)).astimezone(ZoneInfo(_TZ))
    iso = future.replace(microsecond=0).isoformat()
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Quarterly report",
         "due_at": iso, "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    record = cast("dict[str, object]", _parse_body(out)["record"])
    from services.time_utils import parse_utc
    persisted = parse_utc(cast(str, cast("dict[str, object]", _row(db, cast(str, record["id"])))["due_at"]))
    assert abs((persisted - future).total_seconds()) < 2


def test_unparseable_due_at_errors_invalid_time_and_persists_nothing(db: sqlite3.Connection, chat_mp: MP) -> None:
    before = cast(int, db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0])

    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Mystery",
         "due_at": "blorptdfgh nonsense", "act_summary": "x"},
    )

    assert "[schedule(status=error, code=invalid-time" in out
    assert "code=error]" not in out
    assert "hint:" in out
    # Never persisted, never a datetime.min sentinel row.
    after = cast(int, db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0])
    assert after == before
    assert db.execute(
        "SELECT COUNT(*) FROM scheduled_items WHERE due_at LIKE '0001-%'"
    ).fetchone()[0] == 0


def test_past_due_at_errors_due_in_past(db: sqlite3.Connection, chat_mp: MP) -> None:
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


def test_limit_is_declared_in_the_schema() -> None:
    """The ``limit`` param the code reads is now DECLARED in the tool schema —
    schema and code agree."""
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("schedule")
    schema = ability.get_parameters()
    assert "limit" in cast("dict[str, object]", schema["properties"])
    assert cast("dict[str, object]", cast("dict[str, object]", schema["properties"])["limit"])["type"] == "integer"


# ── list / cancel happy paths render structured, parseable bodies ──────────────


def test_list_renders_structured_rows(db: sqlite3.Connection, chat_mp: MP) -> None:
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
    records = cast("list[dict[str, object]]", b["records"])
    assert any(r["message"] == "Water the plants" for r in records)
    assert all("due_at_utc" in r and "status" in r for r in records)


def test_cancel_by_message_removes_row(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``cancel`` with a fuzzy message match flips the row to cancelled and renders
    a success confirmation."""
    create = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Dentist appointment",
         "due_at": "tomorrow 3pm", "act_summary": "x"},
    )
    item_id = cast(str, cast("dict[str, object]", _parse_body(create)["record"])["id"])

    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "cancel", "message": "Dentist", "act_summary": "x"},
    )

    assert "[schedule(status=success" in out
    assert cast("dict[str, object]", _row(db, item_id))["status"] == "cancelled"


def test_cancel_without_target_errors_with_cancel_target_required(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``cancel`` needs ``item_id`` OR ``message`` — an either/or the pre-gate map
    cannot express (ACTION_REQUIRED["cancel"] == ()), so the guard lives in run()
    and surfaces the tool-unique ``code=cancel-target-required`` (schedule.py)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule", {"action": "cancel", "act_summary": "x"}
    )

    assert "[schedule(status=error, code=cancel-target-required" in out
    assert "hint:" in out


# ── Error ladders for malformed input ──────────────────────────────────────────


def test_invalid_recurrence_errors_with_valid_ladder(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A bad ``recurrence`` errors with the tool-unique ``code=invalid-recurrence``
    and a ``valid:`` ladder naming the real accepted keywords (schedule.py
    ``_VALID_RECURRENCES`` + ``interval:N``)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "schedule",
        {"action": "create", "message": "Standup",
         "due_at": "tomorrow 9am", "recurrence": "fortnightly", "act_summary": "x"},
    )

    assert "[schedule(status=error, code=invalid-recurrence" in out
    for keyword in ("daily", "weekly", "monthly", "weekdays", "hourly", "interval:N"):
        assert keyword in out, keyword

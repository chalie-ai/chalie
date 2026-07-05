# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the ``schedule`` ability — the cron recurrence model
(TKT-1434): ``cron_dom``/``cron_hour``/``cron_minute`` (NULL = "every") plus a
``start_at`` UTC activation floor, replacing the old ``recurrence`` keyword
string and ``is_prompt`` notification/prompt toggle (both columns are gone
from schema.sql — there is no backwards compatibility).

Calls ``ScheduleAbility().run(params)`` directly against a real, fully-migrated
SQLite database (the ``db`` fixture). The old ``ToolDispatcher``-based harness
(``tests._tool_result_harness``) drove business logic through a dispatcher that
no longer exists in this tree (deleted as part of the in-flight MP-spine
rewrite, replaced by ``services.dispatch_service.DispatchService``); calling
``run()`` directly exercises the exact same business logic without the
dispatcher's envelope rendering, matching the precedent in
``test_timer_ability.py``. ``self.mp`` is left ``None`` — ``_create``/``_cancel``
read ``self.mp.config.channel`` via a safe ``getattr`` chain that degrades to
``""`` on ``None``, so no fake/mock collaborator is needed.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from abilities.schedule import ScheduleAbility
from services.time_utils import utc_now

pytestmark = pytest.mark.unit

_TZ = "Europe/Malta"


def _seed_timezone(db: sqlite3.Connection, tz_name: str = _TZ) -> None:
    """Write a real heartbeat timezone into the telemetry table — the same
    store the production ``locale_service.get_timezone()`` reads."""
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.execute(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        ("timezone", json.dumps(tz_name)),
    )
    db.commit()
    heartbeat_service._ctx = None


def _row(db: sqlite3.Connection, item_id: str) -> "dict[str, object] | None":
    cur = db.execute(
        "SELECT id, message, start_at, due_at, cron_dom, cron_hour, cron_minute, status "
        "FROM scheduled_items WHERE id = ?",
        (item_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {
        "id": r[0], "message": r[1], "start_at": r[2], "due_at": r[3],
        "cron_dom": r[4], "cron_hour": r[5], "cron_minute": r[6], "status": r[7],
    }


def _count(db: sqlite3.Connection) -> int:
    return cast(int, db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0])


# ── Create persists UTC start_at/due_at + local cron fields verbatim ──────────


def test_create_persists_utc_instants_and_local_cron_fields_verbatim(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    tr = ScheduleAbility().run({
        "action": "create", "message": "Water the plants",
        "day": None, "hour": 14, "minute": 30,  # E F F — every day at 14:30
    })

    assert tr.status == "success"
    record = cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])
    item_id = cast(str, record["id"])

    persisted = _row(db, item_id)
    assert persisted is not None
    assert persisted["status"] == "pending"
    # cron fields land on the row exactly as passed — no timezone conversion
    # is ever applied to them (they are already local by contract).
    assert persisted["cron_dom"] is None
    assert persisted["cron_hour"] == 14
    assert persisted["cron_minute"] == 30
    # start_at/due_at are real, parseable, future UTC instants.
    start_at = datetime.fromisoformat(cast(str, persisted["start_at"]))
    due_at = datetime.fromisoformat(cast(str, persisted["due_at"]))
    assert due_at.tzinfo is not None and due_at.utcoffset() == timedelta(0)
    assert due_at >= utc_now()
    assert start_at <= utc_now() + timedelta(seconds=5)


# ── TZ round-trip: local cron fires materialize the correct UTC instant ────────


def test_every_day_cron_materializes_local_fire_time_correctly_in_utc(db: sqlite3.Connection) -> None:
    """``day: every, hour: 3, minute: 0`` in Europe/Malta must materialize
    ``due_at`` at the true UTC instant of the next local 03:00 — verified
    independently against the real IANA tz database (``zoneinfo``), not by
    re-deriving the production cron-walk algorithm."""
    _seed_timezone(db)

    # Anchor on a concrete future local day so `next_due_from`'s "floor to now"
    # guard never engages, and the target 03:00 fire hasn't passed yet that day.
    target_local_date = (datetime.now(ZoneInfo(_TZ)) + timedelta(days=2)).date()
    start_at_local = f"{target_local_date.isoformat()}T00:00:00"
    expected_fire_local = datetime.combine(target_local_date, datetime.min.time()).replace(
        hour=3, minute=0, tzinfo=ZoneInfo(_TZ)
    )
    expected_fire_utc = expected_fire_local.astimezone(timezone.utc)

    tr = ScheduleAbility().run({
        "action": "create", "message": "Morning briefing",
        "start_at": start_at_local,
        "day": None, "hour": 3, "minute": 0,  # E F F — every day at 03:00
    })

    assert tr.status == "success"
    record = cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])
    item_id = cast(str, record["id"])

    persisted = _row(db, item_id)
    assert persisted is not None
    persisted_due_utc = datetime.fromisoformat(cast(str, persisted["due_at"]))
    assert abs((persisted_due_utc - expected_fire_utc).total_seconds()) < 1

    # Read back through the ability's own local-render path (_format_record):
    # it must localize the same UTC instant back to 03:00 in Europe/Malta.
    assert "03:00" in cast(str, record["due_at"])


# ── start_at omitted defaults to "now" (UTC) ───────────────────────────────────


def test_create_without_start_at_defaults_to_now(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    before = utc_now()
    tr = ScheduleAbility().run({
        "action": "create", "message": "Every-minute ping",
        # day/hour/minute all omitted — E E E, every minute; start_at omitted too.
    })
    after = utc_now()

    assert tr.status == "success"
    record = cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])
    persisted = _row(db, cast(str, record["id"]))
    assert persisted is not None
    start_at = datetime.fromisoformat(cast(str, persisted["start_at"]))
    assert before - timedelta(seconds=2) <= start_at <= after + timedelta(seconds=2)


# ── Every-prefix violations are rejected with code=invalid-cron, nothing persists ──


@pytest.mark.parametrize(
    ("day", "hour", "minute"),
    [
        (None, 3, None),   # E F E — minute cannot be 'every' when hour is fixed
        (15, None, None),  # F E E — day fixed forces hour+minute fixed
        (15, None, 30),    # F E F — day fixed forces hour fixed too
        (15, 9, None),     # F F E — day fixed forces minute fixed too
    ],
)
def test_create_rejects_illegal_every_prefix_shapes(
    db: sqlite3.Connection, day: int | None, hour: int | None, minute: int | None
) -> None:
    _seed_timezone(db)
    before = _count(db)

    tr = ScheduleAbility().run({
        "action": "create", "message": "Illegal shape",
        "day": day, "hour": hour, "minute": minute,
    })

    assert tr.status == "error"
    assert tr.code == "invalid-cron"
    assert _count(db) == before, "an illegal cron shape must persist nothing"


# ── cancel / update — still real, user-facing scheduler behavior ──────────────


def test_cancel_by_message_marks_pending_row_cancelled(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = ScheduleAbility().run({
        "action": "create", "message": "Dentist appointment",
        "day": None, "hour": 15, "minute": 0,
    })
    item_id = cast(str, cast("dict[str, object]", cast("dict[str, object]", create.body)["record"])["id"])

    tr = ScheduleAbility().run({"action": "cancel", "message": "Dentist"})

    assert tr.status == "success"
    assert cast("dict[str, object]", _row(db, item_id))["status"] == "cancelled"


def test_update_cancels_target_and_recreates_with_new_values(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = ScheduleAbility().run({
        "action": "create", "message": "Call the plumber",
        "day": None, "hour": 15, "minute": 0,
    })
    old_id = cast(str, cast("dict[str, object]", cast("dict[str, object]", create.body)["record"])["id"])

    tr = ScheduleAbility().run({
        "action": "update", "item_id": old_id,
        "message": "Call the electrician", "day": None, "hour": 17, "minute": 0,
    })

    assert tr.status == "success"
    new_id = cast(str, cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])["id"])
    assert new_id != old_id
    assert cast("dict[str, object]", _row(db, old_id))["status"] == "cancelled"
    new_row = cast("dict[str, object]", _row(db, new_id))
    assert new_row["status"] == "pending"
    assert new_row["message"] == "Call the electrician"
    assert new_row["cron_hour"] == 17


def test_cancel_without_target_errors_cancel_target_required(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    tr = ScheduleAbility().run({"action": "cancel"})

    assert tr.status == "error"
    assert tr.code == "cancel-target-required"

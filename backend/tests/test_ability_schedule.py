# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the ``schedule`` ability — the prompt-only dumb-cron model
(TKT-1434): ``scheduled_items`` is one row per schedule, forever
(``id`` INTEGER PRIMARY KEY AUTOINCREMENT — also the thread's ``turn_id`` on
the ``'schedule'`` channel). There is no ``item_type``/``due_at``/``status``/
``group_id``/``turn_id`` column any more — a stateless poller
(``services.scheduler_service``) fires any enabled row whose ``start_at``
floor has passed and whose cron fields match the current wall-clock minute.
Cancel is a hard ``DELETE`` (no soft-cancel state to assert on).

Calls ``ScheduleAbility().run(params)`` directly against a real, fully-migrated
SQLite database (the ``db`` fixture). ``self.mp`` is left ``None`` —
``_create``/``_cancel`` read ``self.mp.config.channel`` via a safe ``getattr``
chain that degrades to ``""`` on ``None``, so no fake/mock collaborator is
needed.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import cast

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


def _row(db: sqlite3.Connection, item_id: object) -> "dict[str, object] | None":
    cur = db.execute(
        "SELECT id, message, start_at, cron_dom, cron_hour, cron_minute, enabled "
        "FROM scheduled_items WHERE id = ?",
        (item_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {
        "id": r[0], "message": r[1], "start_at": r[2],
        "cron_dom": r[3], "cron_hour": r[4], "cron_minute": r[5], "enabled": r[6],
    }


def _count(db: sqlite3.Connection) -> int:
    return cast(int, db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0])


def _record_of(tr: object) -> dict[str, object]:
    return cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])  # type: ignore[attr-defined]


# ── Create persists the local cron fields + UTC start_at verbatim ─────────────


def test_create_persists_prompt_only_row_with_local_cron_fields_verbatim(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    tr = ScheduleAbility().run({
        "action": "create", "message": "Water the plants",
        "day": None, "hour": 14, "minute": 30,  # E F F — every day at 14:30
    })

    assert tr.status == "success"
    record = _record_of(tr)
    item_id = record["id"]
    assert isinstance(item_id, int)

    persisted = _row(db, item_id)
    assert persisted is not None
    assert persisted["message"] == "Water the plants"
    assert persisted["enabled"] == 1
    # cron fields land on the row exactly as passed — no timezone conversion
    # is ever applied to them (they are already local by contract).
    assert persisted["cron_dom"] is None
    assert persisted["cron_hour"] == 14
    assert persisted["cron_minute"] == 30
    # start_at is a real, parseable UTC instant, close to "now".
    start_at = datetime.fromisoformat(cast(str, persisted["start_at"]))
    assert start_at <= utc_now() + timedelta(seconds=5)


def test_create_without_start_at_defaults_to_now(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    before = utc_now()
    tr = ScheduleAbility().run({
        "action": "create", "message": "Every-minute ping",
        # day/hour/minute all omitted — E E E, every minute; start_at omitted too.
    })
    after = utc_now()

    assert tr.status == "success"
    persisted = _row(db, _record_of(tr)["id"])
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


# ── cancel = hard DELETE; id never reissued (AUTOINCREMENT) ───────────────────


def test_cancel_by_message_hard_deletes_the_row(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = ScheduleAbility().run({
        "action": "create", "message": "Dentist appointment",
        "day": None, "hour": 15, "minute": 0,
    })
    item_id = _record_of(create)["id"]

    tr = ScheduleAbility().run({"action": "cancel", "message": "Dentist"})

    assert tr.status == "success"
    assert _row(db, item_id) is None, "cancel must hard-delete the row, not soft-cancel it"


def test_cancelled_id_is_never_reissued_by_a_later_create(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    first = ScheduleAbility().run({
        "action": "create", "message": "First schedule",
        "day": None, "hour": 10, "minute": 0,
    })
    first_id = cast(int, _record_of(first)["id"])

    cancel = ScheduleAbility().run({"action": "cancel", "item_id": str(first_id)})
    assert cancel.status == "success"

    second = ScheduleAbility().run({
        "action": "create", "message": "Second schedule",
        "day": None, "hour": 11, "minute": 0,
    })
    second_id = cast(int, _record_of(second)["id"])

    assert second_id != first_id
    assert second_id > first_id
    assert _row(db, first_id) is None


def test_cancel_without_target_errors_cancel_target_required(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    tr = ScheduleAbility().run({"action": "cancel"})

    assert tr.status == "error"
    assert tr.code == "cancel-target-required"


# ── update composes cancel + create; the old row is gone, the new one is live ──


def test_update_hard_deletes_target_and_creates_fresh_row_with_new_values(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = ScheduleAbility().run({
        "action": "create", "message": "Call the plumber",
        "day": None, "hour": 15, "minute": 0,
    })
    old_id = cast(int, _record_of(create)["id"])

    tr = ScheduleAbility().run({
        "action": "update", "item_id": str(old_id),
        "message": "Call the electrician", "day": None, "hour": 17, "minute": 0,
    })

    assert tr.status == "success"
    new_id = _record_of(tr)["id"]
    assert new_id != old_id
    assert _row(db, old_id) is None
    new_row = _row(db, new_id)
    assert new_row is not None
    assert new_row["message"] == "Call the electrician"
    assert new_row["cron_hour"] == 17


# ── enable/disable toggle the poller-visible enabled flag ─────────────────────


def test_disable_then_enable_toggles_the_enabled_flag(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = ScheduleAbility().run({
        "action": "create", "message": "Weekly check-in",
        "day": None, "hour": 9, "minute": 0,
    })
    item_id = cast(int, _record_of(create)["id"])
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 1

    disable = ScheduleAbility().run({"action": "disable", "item_id": str(item_id)})
    assert disable.status == "success"
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 0

    enable = ScheduleAbility().run({"action": "enable", "item_id": str(item_id)})
    assert enable.status == "success"
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 1

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the ``schedule`` ability — the real 5-field crontab
engine: ``scheduled_items`` is one row per schedule, forever (``id`` INTEGER
PRIMARY KEY AUTOINCREMENT — also the thread's ``turn_id`` on the ``'schedule'``
channel), and its five ``cron_minute``/``cron_hour``/``cron_dom``/``cron_month``/
``cron_dow`` columns are TEXT crontab expressions (``'*'`` = every) validated by
``services.cron_schedule.validate_cron``. There is no every-prefix invariant any
more — any combination of fields is legal standard crontab; only a malformed or
out-of-range expression is rejected. Cancel is a hard ``DELETE`` (no soft-cancel
state to assert on).

Calls ``run()`` directly with bags built via ``ScheduleParamsBag.from_params``
— the same factory the dispatch seam uses — against a real, fully-migrated
SQLite database (the ``db`` fixture), on an ability bound to a real inert
``MessageProcessor`` under ``ScheduledConfig`` — the same collaborator the
dispatcher binds in production (construction is side-effect-free, §6.13/I2).
``run()`` raises on an unbound instance by contract, so no fake/mock
collaborator is possible.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import cast

import pytest

from abilities._result import ToolResult
from abilities.schedule import ScheduleAbility
from configs.channels.scheduled import ScheduledConfig
from contracts.params.schedule_params_bag import ScheduleParamsBag
from controllers.message_processor import MessageProcessor
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
        "SELECT id, message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow, enabled "
        "FROM scheduled_items WHERE id = ?",
        (item_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {
        "id": r[0], "message": r[1], "start_at": r[2],
        "cron_minute": r[3], "cron_hour": r[4], "cron_dom": r[5],
        "cron_month": r[6], "cron_dow": r[7], "enabled": r[8],
    }


def _ability() -> ScheduleAbility:
    """A schedule ability bound the way the dispatcher binds it in production:
    to a real, inert ``MessageProcessor`` on the ``schedule`` channel."""
    return ScheduleAbility(MessageProcessor(ScheduledConfig()))


def _run(params: dict[str, object]) -> ToolResult:
    """Dispatch through the same seam production uses: the router bag is built
    from the raw dict first, then handed to ``run()``."""
    return _ability().run(ScheduleParamsBag.from_params(params))


def _record_of(tr: ToolResult) -> dict[str, object]:
    return cast("dict[str, object]", cast("dict[str, object]", tr.body)["record"])


# ── Create persists the local cron fields + UTC start_at verbatim ─────────────


def test_create_persists_prompt_only_row_with_local_cron_fields_verbatim(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    tr = _run({
        "action": "create", "message": "Water the plants",
        "hour": "14", "minute": "30",  # day/month/weekday omitted -> "*" (every)
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
    assert persisted["cron_hour"] == "14"
    assert persisted["cron_minute"] == "30"
    assert persisted["cron_dom"] == "*"
    assert persisted["cron_month"] == "*"
    assert persisted["cron_dow"] == "*"
    # start_at is a real, parseable UTC instant, close to "now".
    start_at = datetime.fromisoformat(cast(str, persisted["start_at"]))
    assert start_at <= utc_now() + timedelta(seconds=5)


def test_create_without_start_at_defaults_to_now(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    before = utc_now()
    tr = _run({
        "action": "create", "message": "Every-minute ping",
        # every cron field omitted -> all "*" (every minute); start_at omitted too.
    })
    after = utc_now()

    assert tr.status == "success"
    persisted = _row(db, _record_of(tr)["id"])
    assert persisted is not None
    assert (
        persisted["cron_minute"], persisted["cron_hour"], persisted["cron_dom"],
        persisted["cron_month"], persisted["cron_dow"],
    ) == ("*", "*", "*", "*", "*")
    start_at = datetime.fromisoformat(cast(str, persisted["start_at"]))
    assert before - timedelta(seconds=2) <= start_at <= after + timedelta(seconds=2)


# ── New capability: any combination is now legal, incl. step/comma/Vixie OR ──


def test_create_persists_new_crontab_capabilities_verbatim(db: sqlite3.Connection) -> None:
    """The every-prefix invariant is gone — any combination of crontab shapes
    is now legal, including ones the old dumb-cron model could never express:
    a step (``hour``), a comma-union (``minute``), and BOTH day-of-month and
    day-of-week restricted at once (the Vixie OR quirk, ``day`` + ``weekday``)."""
    _seed_timezone(db)
    tr = _run({
        "action": "create", "message": "Multi-shape schedule",
        "minute": "0,15,30,45", "hour": "*/2", "day": "13", "month": "*", "weekday": "1-5",
    })

    assert tr.status == "success"
    persisted = _row(db, _record_of(tr)["id"])
    assert persisted is not None
    assert persisted["cron_minute"] == "0,15,30,45"
    assert persisted["cron_hour"] == "*/2"
    assert persisted["cron_dom"] == "13"
    assert persisted["cron_month"] == "*"
    assert persisted["cron_dow"] == "1-5"


# ── cancel = hard DELETE; id never reissued (AUTOINCREMENT) ───────────────────


def test_cancel_by_message_hard_deletes_the_row(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = _run({
        "action": "create", "message": "Dentist appointment",
        "hour": "15", "minute": "0",
    })
    item_id = _record_of(create)["id"]

    tr = _run({"action": "cancel", "message": "Dentist"})

    assert tr.status == "success"
    assert _row(db, item_id) is None, "cancel must hard-delete the row, not soft-cancel it"


def test_cancelled_id_is_never_reissued_by_a_later_create(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    first = _run({
        "action": "create", "message": "First schedule",
        "hour": "10", "minute": "0",
    })
    first_id = cast(int, _record_of(first)["id"])

    cancel = _run({"action": "cancel", "item_id": str(first_id)})
    assert cancel.status == "success"

    second = _run({
        "action": "create", "message": "Second schedule",
        "hour": "11", "minute": "0",
    })
    second_id = cast(int, _record_of(second)["id"])

    assert second_id != first_id
    assert second_id > first_id
    assert _row(db, first_id) is None


# ── update composes cancel + create; the old row is gone, the new one is live ──


def test_update_hard_deletes_target_and_creates_fresh_row_with_new_values(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = _run({
        "action": "create", "message": "Call the plumber",
        "hour": "15", "minute": "0",
    })
    old_id = cast(int, _record_of(create)["id"])

    tr = _run({
        "action": "update", "item_id": str(old_id),
        "message": "Call the electrician", "hour": "17", "minute": "0",
    })

    assert tr.status == "success"
    new_id = _record_of(tr)["id"]
    assert new_id != old_id
    assert _row(db, old_id) is None
    new_row = _row(db, new_id)
    assert new_row is not None
    assert new_row["message"] == "Call the electrician"
    assert new_row["cron_hour"] == "17"


# ── enable/disable toggle the poller-visible enabled flag ─────────────────────


def test_disable_then_enable_toggles_the_enabled_flag(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    create = _run({
        "action": "create", "message": "Weekly check-in",
        "hour": "9", "minute": "0",
    })
    item_id = cast(int, _record_of(create)["id"])
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 1

    disable = _run({"action": "disable", "item_id": str(item_id)})
    assert disable.status == "success"
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 0

    enable = _run({"action": "enable", "item_id": str(item_id)})
    assert enable.status == "success"
    assert cast("dict[str, object]", _row(db, item_id))["enabled"] == 1

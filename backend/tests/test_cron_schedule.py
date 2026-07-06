# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for ``services.cron_schedule.matches`` — the dumb-cron
poller's stateless yes/no question (TKT-1434): does ``now_utc``, converted to
the user's real local wall clock via ``services.locale_service.get_timezone``,
satisfy the (dom, hour, minute) cron triple? A NULL field always matches
("every"). No mocking of the timezone lookup — the real ``telemetry`` table
is seeded through the same path the frontend heartbeat uses, exactly as
``services.locale_service.get_timezone()`` reads it in production.
"""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from services.cron_schedule import matches

pytestmark = pytest.mark.unit

_TZ = "Europe/Malta"  # UTC+1 (winter) / UTC+2 (summer, CEST)


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


def test_null_fields_match_every_instant(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    # Arbitrary instant — every field is NULL ("every minute").
    now_utc = datetime(2026, 4, 1, 13, 47, tzinfo=timezone.utc)

    assert matches(now_utc, dom=None, hour=None, minute=None) is True


def test_exact_minute_hour_dom_match_in_local_time(db: sqlite3.Connection) -> None:
    _seed_timezone(db, "Europe/Malta")
    # 2026-04-01 is before EU DST switch this year context — use a date/time
    # combo and derive the expected local fields the same way production does,
    # from a real ZoneInfo conversion, not a hand-guessed offset.
    from zoneinfo import ZoneInfo

    now_utc = datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo("Europe/Malta"))

    assert matches(now_utc, dom=local.day, hour=local.hour, minute=local.minute) is True


def test_non_matching_minute_fails(db: sqlite3.Connection) -> None:
    _seed_timezone(db, "Europe/Malta")
    from zoneinfo import ZoneInfo

    now_utc = datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo("Europe/Malta"))
    wrong_minute = (local.minute + 1) % 60

    assert matches(now_utc, dom=None, hour=local.hour, minute=wrong_minute) is False


def test_non_matching_hour_fails(db: sqlite3.Connection) -> None:
    _seed_timezone(db, "Europe/Malta")
    from zoneinfo import ZoneInfo

    now_utc = datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo("Europe/Malta"))
    wrong_hour = (local.hour + 1) % 24

    assert matches(now_utc, dom=None, hour=wrong_hour, minute=local.minute) is False


def test_timezone_conversion_crosses_local_day_boundary(db: sqlite3.Connection) -> None:
    """A cron fixed to a local day-of-month must be evaluated against the
    LOCAL day, not the UTC day — the whole reason `matches` converts via
    `get_timezone()` before comparing fields."""
    # 23:30 UTC on 2026-06-15 is 01:30 local the *next* day in Europe/Malta
    # (UTC+2 in June, DST) — a real day-boundary crossing, not a guess.
    from zoneinfo import ZoneInfo

    _seed_timezone(db, "Europe/Malta")
    now_utc = datetime(2026, 6, 15, 23, 30, tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo("Europe/Malta"))
    assert local.day != now_utc.day, "fixture must actually cross a day boundary"

    # Matching against the UTC day must fail...
    assert matches(now_utc, dom=now_utc.day, hour=local.hour, minute=local.minute) is False
    # ...while matching against the real local day succeeds.
    assert matches(now_utc, dom=local.day, hour=local.hour, minute=local.minute) is True

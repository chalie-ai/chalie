# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for ``services.cron_schedule`` — the real 5-field crontab
engine the scheduler poller asks a stateless yes/no question each minute.

Two layers are exercised:

* ``parse_field`` / ``validate_cron`` — pure crontab-expression parsing and
  range validation (no timezone, no DB).
* ``matches`` — does ``now_utc``, converted to the user's real local wall clock
  via ``LocaleService.get_timezone``, satisfy all five fields
  (minute, hour, day-of-month, month, day-of-week)? No mocking of the timezone
  lookup — the real ``telemetry`` table is seeded through the same path the
  frontend heartbeat uses, exactly as ``get_timezone()`` reads it in
  production. Every local-time expectation is derived from a real ``ZoneInfo``
  conversion (or an independently verified weekday), never a hand-guessed
  offset.
"""

import json
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from services.cron_schedule import matches, parse_field, validate_cron

pytestmark = pytest.mark.unit

_TZ = "Europe/Malta"  # UTC+1 (winter) / UTC+2 (summer, CEST)
_MT = ZoneInfo(_TZ)


def _seed_timezone(db: sqlite3.Connection, tz_name: str = _TZ) -> None:
    """Write a real heartbeat timezone into the telemetry table — the same
    store the production ``LocaleService.get_timezone()`` reads."""
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.execute(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        ("timezone", json.dumps(tz_name)),
    )
    db.commit()
    heartbeat_service._ctx = None


def _utc(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# ── parse_field: pure expression → set of ints (no tz / DB) ───────────────────


def test_parse_field_star_is_full_range() -> None:
    assert parse_field("*", 0, 59) == frozenset(range(0, 60))
    assert parse_field("*", 1, 12) == frozenset(range(1, 13))


def test_parse_field_step_every_5() -> None:
    assert parse_field("*/5", 0, 59) == frozenset(range(0, 60, 5))


def test_parse_field_comma_list() -> None:
    assert parse_field("0,15,30,45", 0, 59) == frozenset({0, 15, 30, 45})


def test_parse_field_inclusive_range() -> None:
    assert parse_field("9-17", 0, 23) == frozenset(range(9, 18))


def test_parse_field_range_with_step() -> None:
    assert parse_field("0-30/10", 0, 59) == frozenset({0, 10, 20, 30})


def test_parse_field_single_value() -> None:
    assert parse_field("7", 0, 23) == frozenset({7})


def test_parse_field_union_of_shapes() -> None:
    # A comma-union mixing a value, a range and a step.
    assert parse_field("0,10-12,*/20", 0, 59) == frozenset({0, 10, 11, 12, 20, 40})


@pytest.mark.parametrize("expr", ["", "  ", "*/0", "*/-1", "60", "8-3", "mon", "5-", "1,,2", "1-2-3"])
def test_parse_field_rejects_malformed(expr: str) -> None:
    with pytest.raises(ValueError):
        parse_field(expr, 0, 59)


# ── validate_cron: five fields, per-field range + name-prefixed errors ────────


def test_validate_cron_accepts_any_combination() -> None:
    # No every-prefix invariant any more — a fixed minute with an "every" hour
    # is perfectly legal crontab.
    validate_cron("*/5", "*", "*", "*", "*")
    validate_cron("0", "9", "*", "*", "1-5")
    validate_cron("30", "*", "1,15", "*", "*")
    validate_cron("0", "0", "1", "1", "0")


@pytest.mark.parametrize(
    ("field", "args", "prefix"),
    [
        ("minute", ("60", "*", "*", "*", "*"), "minute:"),
        ("minute", ("", "*", "*", "*", "*"), "minute:"),
        ("hour", ("*", "24", "*", "*", "*"), "hour:"),
        ("dom", ("*", "*", "0", "*", "*"), "dom:"),      # dom lower bound is 1
        ("dom", ("*", "*", "8-3", "*", "*"), "dom:"),    # start > end
        ("month", ("*", "*", "*", "13", "*"), "month:"),
        ("dow", ("*", "*", "*", "*", "8"), "dow:"),      # dow upper bound is 7
        ("dow", ("*", "*", "*", "*", "mon"), "dow:"),    # numeric only
    ],
)
def test_validate_cron_rejects_with_field_prefix(
    field: str, args: tuple[str, str, str, str, str], prefix: str
) -> None:
    with pytest.raises(ValueError) as exc:
        validate_cron(*args)
    assert str(exc.value).startswith(prefix), f"{field} error must name its field"


# ── matches: every field "*" ─────────────────────────────────────────────────


def test_all_star_matches_every_instant(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    now_utc = _utc(2026, 4, 1, 13, 47)
    assert matches(now_utc, "*", "*", "*", "*", "*") is True


# ── matches: exact fixed time in LOCAL wall clock ─────────────────────────────


def test_exact_local_time_matches(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 10, 30)
    local = now_utc.astimezone(_MT)  # 12:30 local (CEST, UTC+2), Monday the 15th
    assert matches(
        now_utc,
        str(local.minute),
        str(local.hour),
        str(local.day),
        str(local.month),
        "*",
    ) is True


def test_wrong_minute_fails(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 10, 30)
    local = now_utc.astimezone(_MT)
    wrong_minute = (local.minute + 1) % 60
    assert matches(now_utc, str(wrong_minute), str(local.hour), "*", "*", "*") is False


def test_wrong_hour_fails(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 10, 30)
    local = now_utc.astimezone(_MT)
    wrong_hour = (local.hour + 1) % 24
    assert matches(now_utc, str(local.minute), str(wrong_hour), "*", "*", "*") is False


# ── matches: interval / list / range shapes ───────────────────────────────────


def test_every_5_minutes(db: sqlite3.Connection) -> None:
    """The headline capability: minute='*/5' fires on :30 (a multiple of 5) but
    not on :32. Malta is a whole-hour offset, so UTC minute == local minute."""
    _seed_timezone(db)
    assert matches(_utc(2026, 6, 15, 10, 30), "*/5", "*", "*", "*", "*") is True
    assert matches(_utc(2026, 6, 15, 10, 32), "*/5", "*", "*", "*", "*") is False


def test_minute_comma_list(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    assert matches(_utc(2026, 6, 15, 10, 30), "0,15,30,45", "*", "*", "*", "*") is True
    assert matches(_utc(2026, 6, 15, 10, 32), "0,15,30,45", "*", "*", "*", "*") is False


def test_hour_range(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 10, 30)  # 12:30 local
    assert matches(now_utc, "*", "9-17", "*", "*", "*") is True
    assert matches(now_utc, "*", "0-6", "*", "*", "*") is False


# ── matches: day-of-week (numbering + Sunday fold) ────────────────────────────


def test_weekday_range_matches_only_on_weekdays(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    # dow='1-5' = Mon..Fri (cron Sun=0). Independently verified local weekdays:
    monday = _utc(2026, 6, 15, 10, 30)      # local Mon 15th  (cron_dow 1)
    saturday = _utc(2026, 6, 13, 10, 30)    # local Sat 13th  (cron_dow 6)
    sunday = _utc(2026, 6, 14, 10, 30)      # local Sun 14th  (cron_dow 0)
    assert matches(monday, "*", "*", "*", "*", "1-5") is True
    assert matches(saturday, "*", "*", "*", "*", "1-5") is False
    assert matches(sunday, "*", "*", "*", "*", "1-5") is False


def test_sunday_matches_both_0_and_7(db: sqlite3.Connection) -> None:
    """Standard crontab folds 7 -> Sunday, so dow='0' and dow='7' both match a
    Sunday (2026-06-14 local is a Sunday)."""
    _seed_timezone(db)
    sunday = _utc(2026, 6, 14, 10, 30)
    assert matches(sunday, "*", "*", "*", "*", "0") is True
    assert matches(sunday, "*", "*", "*", "*", "7") is True
    # ...and a Monday matches neither.
    monday = _utc(2026, 6, 15, 10, 30)
    assert matches(monday, "*", "*", "*", "*", "0") is False
    assert matches(monday, "*", "*", "*", "*", "7") is False


# ── matches: the Vixie DOM/DOW OR quirk ───────────────────────────────────────


def test_dom_and_dow_both_restricted_is_an_or(db: sqlite3.Connection) -> None:
    """When BOTH day-of-month and day-of-week are restricted, a match on EITHER
    fires (the Vixie crond default). Expr: dom='13', dow='5' (Friday)."""
    _seed_timezone(db)
    friday_the_13th = _utc(2026, 11, 13, 10, 30)  # local Fri the 13th — both match
    dom13_not_friday = _utc(2026, 6, 13, 10, 30)  # local Sat the 13th — dom matches
    friday_not_13th = _utc(2026, 6, 19, 10, 30)   # local Fri the 19th — dow matches
    neither = _utc(2026, 6, 15, 10, 30)           # local Mon the 15th — neither

    assert matches(friday_the_13th, "*", "*", "13", "*", "5") is True
    assert matches(dom13_not_friday, "*", "*", "13", "*", "5") is True
    assert matches(friday_not_13th, "*", "*", "13", "*", "5") is True
    assert matches(neither, "*", "*", "13", "*", "5") is False


def test_only_dom_restricted_governs_alone(db: sqlite3.Connection) -> None:
    """dow='*' (unrestricted) → the day is governed by dom alone (AND, not OR)."""
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 10, 30)  # local day 15
    assert matches(now_utc, "*", "*", "15", "*", "*") is True
    assert matches(now_utc, "*", "*", "16", "*", "*") is False


# ── matches: short months + month field ───────────────────────────────────────


def test_dom_31_never_fires_in_february(db: sqlite3.Connection) -> None:
    """dom is a literal day number with no calendar arithmetic — day 31 simply
    never occurs in February, so a dom='31' schedule cannot fire that month."""
    _seed_timezone(db)
    feb_instant = _utc(2026, 2, 15, 10, 30)  # local Feb, day 15
    assert matches(feb_instant, "*", "*", "31", "*", "*") is False


def test_month_field_restricts(db: sqlite3.Connection) -> None:
    _seed_timezone(db)
    june = _utc(2026, 6, 15, 10, 30)  # local month 6
    assert matches(june, "*", "*", "*", "6", "*") is True
    assert matches(june, "*", "*", "*", "7", "*") is False
    assert matches(june, "*", "*", "*", "*/3", "*") is False  # Jan/Apr/Jul/Oct, not Jun


# ── matches: timezone correctness (local day, not UTC day) ────────────────────


def test_local_day_boundary_crossing(db: sqlite3.Connection) -> None:
    """A day-of-month cron must be evaluated against the LOCAL day, not the UTC
    day — the whole reason ``matches`` converts via ``get_timezone()`` first.
    23:30 UTC on 2026-06-15 is 01:30 local the NEXT day in Europe/Malta."""
    _seed_timezone(db)
    now_utc = _utc(2026, 6, 15, 23, 30)
    local = now_utc.astimezone(_MT)
    assert local.day != now_utc.day, "fixture must actually cross a day boundary"

    # Matching against the UTC day must fail...
    assert matches(
        now_utc, str(local.minute), str(local.hour), str(now_utc.day), "*", "*"
    ) is False
    # ...while matching against the real local day succeeds.
    assert matches(
        now_utc, str(local.minute), str(local.hour), str(local.day), "*", "*"
    ) is True

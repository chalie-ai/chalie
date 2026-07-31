# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract guard for the ``day`` key every projected message carries.

``timestamp`` is a *display* string (``CHAT_TIMESTAMP_FMT`` = ``%d %b %H:%M``)
and deliberately carries no year — it is rendered beside a bubble, not parsed.
The conversation spine also groups turns into day dividers, and it used to get
that day by parsing ``timestamp`` with JavaScript's ``new Date()``. A year-less
string leaves the year to the engine (WebKit invents 2000, V8 invents 2001), so
dividers rendered a wrong weekday and the "Today" divider could never match.

``day`` closes that hole at the source: the backend, which alone knows the
user's configured timezone, emits the calendar day as ``%Y-%m-%d`` so the
frontend never parses a date. These tests pin the two properties the frontend
depends on — a real 4-digit year, and agreement with the ``timestamp`` shown
next to it — by driving the real serializer over real transcript rows.

No mocks: a real inert ``MessageProcessor`` appends real rows to the real,
fully-migrated SQLite database, exactly as production does.
"""

import json
import re
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from services.turn_serializer_service import get_service as _serializer
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_CHANNEL = "user"
_DAY_KEY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def _messages(turn_id: int) -> list[dict[str, object]]:
    result = _serializer().serialize(_CHANNEL, turn_id)
    return cast("list[dict[str, object]]", result["messages"])


def _seed_timezone(db: sqlite3.Connection, tz_name: str) -> None:
    """Put a real timezone in the real telemetry table, the way a heartbeat does.

    Without this the table is empty, ``get_timezone()`` falls back to UTC
    (``locale_service.py``), and every assertion below would hold just as well
    if the timezone conversion were removed entirely.
    """
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.execute("INSERT INTO telemetry (key, value) VALUES (?, ?)", ("timezone", json.dumps(tz_name)))
    db.commit()


def test_every_projected_message_carries_a_full_calendar_day(db: sqlite3.Connection) -> None:
    """Every row the serializer projects must carry ``day`` as ``YYYY-MM-DD``
    with a real 4-digit year. The frontend stamps this verbatim onto the turn
    host, so a missing or malformed key silently drops the turn into the day
    group above it."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9401
    mp = MessageProcessor(UserConfig(), turn_id, "What day is it?")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.transcript_service.append_assistant("It is Wednesday.")

    messages = _messages(turn_id)

    assert messages, "sanity: the real appends produced projectable rows"
    for m in messages:
        day = cast("str", m["day"])
        assert _DAY_KEY.match(day), f"day must be YYYY-MM-DD, got {day!r}"
        # The invented years are what the old frontend parse produced. A real
        # row can never legitimately land on them, so they stay a hard guard.
        assert not day.startswith(("2000-", "2001-"))


def test_day_and_display_timestamp_describe_the_same_instant(db: sqlite3.Connection) -> None:
    """``day`` and ``timestamp`` are two renderings of one ``created_at``, both
    resolved through the user's configured timezone. If they ever disagreed,
    a turn would file under one date while displaying another — so the
    day-of-month and month in ``timestamp`` must match ``day``'s tail."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9402
    mp = MessageProcessor(UserConfig(), turn_id, "Group me correctly.")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)

    message = _messages(turn_id)[0]
    day = cast("str", message["day"])
    timestamp = cast("str", message["timestamp"])

    # timestamp is "%d %b %H:%M" — e.g. "29 Jul 10:35"; it has no year at all,
    # which is precisely why `day` has to exist.
    dd, mon, _ = timestamp.split(" ", 2)
    assert day.endswith(f"-{_MONTHS[mon]}-{dd}"), f"{day!r} disagrees with {timestamp!r}"


def test_day_is_stable_across_every_row_of_one_turn(db: sqlite3.Connection) -> None:
    """A turn's rows are written within moments of each other, so they share a
    calendar day (barring a real midnight crossing). The divider reconciler
    reads only ``messages[0]``, so a per-row disagreement would mean the
    divider silently contradicts the rows beneath it."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9403
    mp = MessageProcessor(UserConfig(), turn_id, "First.")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.transcript_service.append_assistant("Second.")
    mp.transcript_service.append_assistant("Third.")

    days = {cast("str", m["day"]) for m in _messages(turn_id)}

    assert len(days) == 1, f"one turn projected rows across multiple days: {days}"


def test_day_is_the_users_local_calendar_day_not_the_utc_one(db: sqlite3.Connection) -> None:
    """``day`` must be resolved through the user's configured timezone, which is
    the whole reason the backend owns this key — the browser cannot know that
    timezone. Pinning a row to 23:40 UTC puts the two calendars on *different
    dates*: still 27 Jul in UTC, already 28 Jul in Malta (UTC+2 in July). A turn
    the user sent just after their midnight has to file under the day they
    experienced, and this is the only test here that can tell the difference —
    under the default UTC fallback the local and UTC renderings are identical,
    so dropping the conversion would otherwise go unnoticed.
    """
    turn_id = 9405
    mp = MessageProcessor(UserConfig(), turn_id, "Just past midnight, my time.")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    # Rewrite the stored instant rather than the clock: created_at is UTC text
    # ("2026-07-29 15:44:24"), so this is exactly a row written at 23:40 UTC.
    db.execute("UPDATE transcript SET created_at = ? WHERE turn_id = ?", ("2026-07-27 23:40:00", turn_id))
    db.commit()
    _seed_timezone(db, "Europe/Malta")

    message = _messages(turn_id)[0]

    # 2026-07-27 would be the UTC date — seeing it means the conversion is gone.
    assert message["day"] == "2026-07-28", "day must follow the user's calendar, not UTC's"
    # The displayed time moves with it; the two renderings never disagree.
    assert message["timestamp"] == "28 Jul 01:40"


def test_day_survives_the_rest_contract(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The serializer emitting ``day`` is not enough — ``GET /api/threads/<id>``
    marshals through the ``Message`` DTO, which drops any key it does not
    declare. This drives the real HTTP read so the field is proven to reach the
    client, not merely to exist one layer below it."""
    client, _db, _store = authed_client
    turn_id = 9404
    mp = MessageProcessor(UserConfig(), turn_id, "Does the day key survive marshalling?")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid
    mp.transcript_service.append_assistant("It must.")

    resp = client.get(f"/api/threads/{turn_id}")

    assert resp.status_code == 200
    messages = resp.get_json()["result"]["messages"]
    assert messages, "sanity: the real appends are visible over HTTP"
    for m in messages:
        assert _DAY_KEY.match(m["day"]), f"day missing or malformed over HTTP: {m.get('day')!r}"

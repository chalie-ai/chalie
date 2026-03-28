"""Unit tests for ``caldav_find_free_slots`` tool handler."""

from __future__ import annotations

import datetime
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_FIXED_NOW = datetime.datetime(2026, 3, 30, 6, 0, 0, tzinfo=_UTC)
_DAY = {"date_from": "2026-03-30T00:00:00+00:00",
        "date_to": "2026-03-31T00:00:00+00:00",
        "working_hours_start": 9, "working_hours_end": 17}


def _evt(uid, ds, de, all_day=False):
    return {"uid": uid, "summary": uid, "dtstart": ds.isoformat(),
            "dtend": de.isoformat(), "all_day": all_day,
            "location": "", "attendees": [], "calendar_name": "T"}


def _handler(connected=True):
    from capabilities.caldav_capability.capability import CaldavCapability
    cap = CaldavCapability()
    cap._connected = connected
    tools = {t["name"]: t["handler"] for t in cap.get_tools()}
    return tools["caldav_find_free_slots"]


def _setup_db(events):
    """Create an in-memory SQLite with scheduled_items populated from events."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE scheduled_items (
        id TEXT PRIMARY KEY, item_type TEXT, message TEXT, due_at TEXT,
        status TEXT, topic TEXT, source TEXT, external_uid TEXT,
        hidden INTEGER, created_at TEXT, recurrence TEXT, is_prompt INTEGER,
        metadata TEXT
    )""")
    for i, e in enumerate(events):
        meta = json.dumps({"dtend": e["dtend"], "all_day": e.get("all_day", False)})
        conn.execute(
            "INSERT INTO scheduled_items (id, item_type, due_at, status, source, metadata) "
            "VALUES (?, 'event', ?, 'pending', 'caldav', ?)",
            (f"e{i}", e["dtstart"], meta),
        )
    conn.commit()
    return conn


def _run(events, params):
    conn = _setup_db(events)
    db_mock = MagicMock()
    db_mock.connection.return_value.__enter__ = lambda s: conn
    db_mock.connection.return_value.__exit__ = MagicMock(return_value=False)
    with patch("services.database_service.get_shared_db_service", return_value=db_mock), \
         patch("capabilities.caldav_capability.capability._get_user_tz", return_value=None), \
         patch("capabilities.caldav_capability.capability.utc_now", return_value=_FIXED_NOW):
        result = _handler()(topic="", params=params)
    conn.close()
    return result


class TestFindFreeSlots:

    @pytest.mark.unit
    def test_not_connected(self):
        assert "error" in _handler(connected=False)(topic="", params={})

    @pytest.mark.unit
    def test_single_event_two_gaps(self):
        ev = _evt("e1", datetime.datetime(2026, 3, 30, 12, 0, tzinfo=_UTC),
                       datetime.datetime(2026, 3, 30, 13, 0, tzinfo=_UTC))
        r = _run([ev], _DAY)
        assert r["count"] == 2
        assert r["free_slots"][0]["duration_minutes"] == 180
        assert r["free_slots"][1]["duration_minutes"] == 240

    @pytest.mark.unit
    def test_fully_booked_and_min_duration(self):
        # Fully booked: event covers entire working hours
        ev = _evt("e1", datetime.datetime(2026, 3, 30, 8, 0, tzinfo=_UTC),
                       datetime.datetime(2026, 3, 30, 18, 0, tzinfo=_UTC))
        r = _run([ev], {**_DAY, "working_hours_start": 8, "working_hours_end": 18})
        assert r["count"] == 0
        # Min duration filter: 10-min gap too short
        evs = [_evt("e1", datetime.datetime(2026, 3, 30, 9, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 30, 9, 50, tzinfo=_UTC)),
               _evt("e2", datetime.datetime(2026, 3, 30, 10, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 30, 17, 0, tzinfo=_UTC))]
        r = _run(evs, {**_DAY, "min_duration_minutes": 15})
        assert r["count"] == 0

    @pytest.mark.unit
    def test_multi_day(self):
        evs = [_evt("e1", datetime.datetime(2026, 3, 30, 10, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 30, 11, 0, tzinfo=_UTC)),
               _evt("e2", datetime.datetime(2026, 3, 31, 14, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 31, 15, 0, tzinfo=_UTC))]
        r = _run(evs, {**_DAY, "date_to": "2026-04-01T00:00:00+00:00"})
        assert r["count"] == 4

    @pytest.mark.unit
    def test_overlapping_events(self):
        evs = [_evt("e1", datetime.datetime(2026, 3, 30, 10, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 30, 12, 0, tzinfo=_UTC)),
               _evt("e2", datetime.datetime(2026, 3, 30, 11, 0, tzinfo=_UTC),
                          datetime.datetime(2026, 3, 30, 13, 0, tzinfo=_UTC))]
        r = _run(evs, _DAY)
        assert r["count"] == 2
        assert r["free_slots"][0]["duration_minutes"] == 60
        assert r["free_slots"][1]["duration_minutes"] == 240

    @pytest.mark.unit
    def test_no_events_allday_ignored_custom_hours(self):
        # No events — full working day free
        r = _run([], _DAY)
        assert r["count"] == 1 and r["free_slots"][0]["duration_minutes"] == 480
        # All-day events should not block timed slots
        ev = _evt("e1", datetime.datetime(2026, 3, 30, 0, 0, tzinfo=_UTC),
                       datetime.datetime(2026, 3, 31, 0, 0, tzinfo=_UTC), all_day=True)
        r = _run([ev], _DAY)
        assert r["count"] == 1 and r["free_slots"][0]["duration_minutes"] == 480
        # Custom working hours respected
        r = _run([], {**_DAY, "working_hours_start": 10, "working_hours_end": 14})
        assert r["free_slots"][0]["duration_minutes"] == 240

    @pytest.mark.unit
    def test_timezone_aware_working_hours(self):
        """Working hours clamp uses user timezone when available."""
        from zoneinfo import ZoneInfo
        tz_ny = ZoneInfo("America/New_York")
        # No events — working hours 9-17 ET = 13:00-21:00 UTC (EDT, -4)
        conn = _setup_db([])
        db_mock = MagicMock()
        db_mock.connection.return_value.__enter__ = lambda s: conn
        db_mock.connection.return_value.__exit__ = MagicMock(return_value=False)
        with patch("services.database_service.get_shared_db_service", return_value=db_mock), \
             patch("capabilities.caldav_capability.capability._get_user_tz", return_value=tz_ny), \
             patch("capabilities.caldav_capability.capability.utc_now", return_value=_FIXED_NOW):
            r = _handler()(topic="", params=_DAY)
        conn.close()
        # 9am-5pm ET = 13:00-21:00 UTC, but window ends at midnight UTC
        # So working window is 13:00 - 21:00 UTC (clamped to window_end 00:00+1d)
        assert r["count"] == 1
        assert r["free_slots"][0]["duration_minutes"] == 480

"""Unit tests for ``caldav_find_free_slots`` tool handler."""

from __future__ import annotations

import datetime
import json
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


def _run(events, params):
    facts = [{"key": "event:" + e["uid"], "data": json.dumps(e)} for e in events]
    ks = MagicMock()
    ks.search.return_value = facts
    with patch("services.knowledge_service.KnowledgeService", MagicMock(return_value=ks)), \
         patch("services.database_service.get_shared_db_service", return_value=MagicMock()), \
         patch("capabilities.caldav_capability.capability.utc_now", return_value=_FIXED_NOW):
        return _handler()(topic="", params=params)


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
        assert r["slots"][0]["duration_minutes"] == 180
        assert r["slots"][1]["duration_minutes"] == 240

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
        assert r["slots"][0]["duration_minutes"] == 60
        assert r["slots"][1]["duration_minutes"] == 240

    @pytest.mark.unit
    def test_no_events_allday_ignored_custom_hours(self):
        # No events — full working day free
        r = _run([], _DAY)
        assert r["count"] == 1 and r["slots"][0]["duration_minutes"] == 480
        # All-day events should not block timed slots
        ev = _evt("e1", datetime.datetime(2026, 3, 30, 0, 0, tzinfo=_UTC),
                       datetime.datetime(2026, 3, 31, 0, 0, tzinfo=_UTC), all_day=True)
        r = _run([ev], _DAY)
        assert r["count"] == 1 and r["slots"][0]["duration_minutes"] == 480
        # Custom working hours respected
        r = _run([], {**_DAY, "working_hours_start": 10, "working_hours_end": 14})
        assert r["slots"][0]["duration_minutes"] == 240

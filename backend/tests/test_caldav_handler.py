"""
Unit tests for :class:`capabilities.mail_capability.caldav_handler.CaldavHandler`.

Coverage:
    1.  ``parse_event`` — timed event fields are populated correctly.
    2.  ``parse_event`` — all-day event sets ``all_day=True``.
    3.  ``parse_event`` — DURATION used when DTEND absent.
    4.  ``parse_event`` — multiple attendees returned as list.
    5.  ``parse_event`` — returns empty list when VEVENT has no DTSTART.
    6.  ``parse_event`` — returns empty list when ical instance unavailable.
    7.  ``_find_overlap_pairs`` — detects overlapping timed events.
    8.  ``_find_overlap_pairs`` — all-day events excluded.
    9.  ``_find_overlap_pairs`` — non-overlapping events produce empty list.
    10. ``_find_back_to_back_pairs`` — detects gap < 5 min.
    11. ``_find_back_to_back_pairs`` — gap >= 5 min is not flagged.
    12. ``upsert_events`` — events written to scheduled_items with source='mail'.
    13. ``upsert_events`` — stale events are cancelled.
    14. ``upsert_events`` — 15-min alert created for upcoming event.
    15. ``upsert_events`` — alert NOT created for all-day events.
    16. ``upsert_events`` — conflict notification created for overlapping events.
    17. ``upsert_events`` — daily digest inserted exactly once.
    18. ``upsert_events`` — greeting inserted exactly once.
    19. ``list_events`` — returns events filtered by date range.
    20. ``list_events`` — filters by calendar_name (case-insensitive).
    21. ``get_event`` — returns error when uid missing.
    22. ``get_event`` — returns event dict when found.
    23. ``get_event`` — returns error when not found.
    24. ``find_free_slots`` — gap found between two events.
    25. ``find_free_slots`` — no slots when fully booked.
    26. ``create_event`` — returns error when summary missing.
    27. ``create_event`` — returns error when dtstart missing.
    28. ``open_client`` — raises RuntimeError when caldav unavailable.

All tests are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler():
    from capabilities.mail_capability.caldav_handler import CaldavHandler
    return CaldavHandler()


def _dt(year, month, day, hour=0, minute=0) -> datetime.datetime:
    return datetime.datetime(year, month, day, hour, minute, tzinfo=_UTC)


def _make_event(
    uid="uid-1",
    summary="Meeting",
    dtstart=None,
    dtend=None,
    all_day=False,
    cal_name="Work",
    attendees=None,
    recurrence=None,
    location=None,
) -> dict:
    start = dtstart or _dt(2026, 4, 1, 9)
    end = dtend or _dt(2026, 4, 1, 10)
    return {
        "uid": uid,
        "summary": summary,
        "dtstart": start,
        "dtend": end,
        "all_day": all_day,
        "calendar_name": cal_name,
        "attendees": attendees or [],
        "recurrence": recurrence,
        "location": location,
    }


def _build_ical_resource(uid, summary, dtstart, dtend=None, attendees=None):
    """Build a mock caldav resource wrapping a real icalendar object."""
    import icalendar

    cal = icalendar.Calendar()
    evt = icalendar.Event()
    evt.add("uid", uid)
    evt.add("summary", summary)
    evt.add("dtstart", dtstart)
    if dtend is not None:
        evt.add("dtend", dtend)
    if attendees:
        for a in attendees:
            evt.add("attendee", f"mailto:{a}")
    cal.add_component(evt)

    resource = MagicMock()
    resource.icalendar_instance = cal
    return resource


def _build_duration_resource(uid, summary, dtstart, duration):
    """Build a mock resource with DURATION instead of DTEND."""
    import icalendar

    cal = icalendar.Calendar()
    evt = icalendar.Event()
    evt.add("uid", uid)
    evt.add("summary", summary)
    evt.add("dtstart", dtstart)
    evt.add("duration", duration)
    cal.add_component(evt)

    resource = MagicMock()
    resource.icalendar_instance = cal
    return resource


# ---------------------------------------------------------------------------
# parse_event tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseEvent:
    def test_timed_event_fields(self):
        handler = _make_handler()
        dtstart = _dt(2026, 4, 1, 9)
        dtend = _dt(2026, 4, 1, 10)
        resource = _build_ical_resource("uid-1", "Standup", dtstart, dtend)

        result = handler.parse_event(resource, "Work")

        assert len(result) == 1
        ev = result[0]
        assert ev["uid"] == "uid-1"
        assert ev["summary"] == "Standup"
        assert ev["dtstart"] == dtstart
        assert ev["dtend"] == dtend
        assert ev["all_day"] is False
        assert ev["calendar_name"] == "Work"

    def test_all_day_event(self):
        handler = _make_handler()
        dtstart = datetime.date(2026, 4, 1)
        resource = _build_ical_resource("uid-2", "Birthday", dtstart)

        result = handler.parse_event(resource, "Personal")

        assert len(result) == 1
        assert result[0]["all_day"] is True

    def test_duration_fallback(self):
        handler = _make_handler()
        dtstart = _dt(2026, 4, 1, 10)
        duration = datetime.timedelta(hours=2)
        resource = _build_duration_resource("uid-3", "Workshop", dtstart, duration)

        result = handler.parse_event(resource, "Work")

        assert len(result) == 1
        expected_end = dtstart + duration
        assert result[0]["dtend"] == expected_end

    def test_multiple_attendees(self):
        handler = _make_handler()
        dtstart = _dt(2026, 4, 2, 14)
        dtend = _dt(2026, 4, 2, 15)
        resource = _build_ical_resource(
            "uid-4", "Team Sync", dtstart, dtend,
            attendees=["alice@example.com", "bob@example.com"],
        )

        result = handler.parse_event(resource, "Work")

        assert len(result) == 1
        attendees = result[0]["attendees"]
        assert "alice@example.com" in attendees
        assert "bob@example.com" in attendees
        assert len(attendees) == 2

    def test_no_dtstart_returns_empty(self):
        import icalendar

        handler = _make_handler()
        cal = icalendar.Calendar()
        evt = icalendar.Event()
        evt.add("uid", "uid-5")
        evt.add("summary", "No start")
        # DTSTART deliberately omitted
        cal.add_component(evt)
        resource = MagicMock()
        resource.icalendar_instance = cal

        result = handler.parse_event(resource, "Work")

        assert result == []

    def test_ical_instance_unavailable_returns_empty(self):
        handler = _make_handler()
        resource = MagicMock(spec=[])  # no icalendar_instance attribute

        result = handler.parse_event(resource, "Work")

        assert result == []


# ---------------------------------------------------------------------------
# _find_overlap_pairs tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindOverlapPairs:
    def test_overlapping_events_detected(self):
        from capabilities.mail_capability.caldav_handler import _find_overlap_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event("a", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11))
        ev_b = _make_event("b", dtstart=_dt(2026, 4, 1, 10, 30), dtend=_dt(2026, 4, 1, 11, 30))

        pairs = _find_overlap_pairs([ev_a, ev_b], now)

        assert len(pairs) == 1
        canon = pairs[0][2]
        assert "a" in canon and "b" in canon

    def test_all_day_excluded(self):
        from capabilities.mail_capability.caldav_handler import _find_overlap_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event(
            "a", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11), all_day=True
        )
        ev_b = _make_event("b", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11))

        pairs = _find_overlap_pairs([ev_a, ev_b], now)

        assert pairs == []

    def test_non_overlapping_events(self):
        from capabilities.mail_capability.caldav_handler import _find_overlap_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event("a", dtstart=_dt(2026, 4, 1, 9), dtend=_dt(2026, 4, 1, 10))
        ev_b = _make_event("b", dtstart=_dt(2026, 4, 1, 11), dtend=_dt(2026, 4, 1, 12))

        pairs = _find_overlap_pairs([ev_a, ev_b], now)

        assert pairs == []


# ---------------------------------------------------------------------------
# _find_back_to_back_pairs tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindBackToBackPairs:
    def test_tight_gap_detected(self):
        from capabilities.mail_capability.caldav_handler import _find_back_to_back_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event("a", dtstart=_dt(2026, 4, 1, 9), dtend=_dt(2026, 4, 1, 10))
        # 3-minute gap — below 5-minute threshold
        ev_b = _make_event(
            "b",
            dtstart=_dt(2026, 4, 1, 10, 3),
            dtend=_dt(2026, 4, 1, 11),
        )

        pairs = _find_back_to_back_pairs([ev_a, ev_b], now)

        assert len(pairs) == 1
        assert pairs[0][2] == 3  # gap_minutes

    def test_large_gap_not_flagged(self):
        from capabilities.mail_capability.caldav_handler import _find_back_to_back_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event("a", dtstart=_dt(2026, 4, 1, 9), dtend=_dt(2026, 4, 1, 10))
        ev_b = _make_event("b", dtstart=_dt(2026, 4, 1, 11), dtend=_dt(2026, 4, 1, 12))

        pairs = _find_back_to_back_pairs([ev_a, ev_b], now)

        assert pairs == []


# ---------------------------------------------------------------------------
# upsert_events tests (use real in-memory DB via conftest `db` fixture)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpsertEvents:
    def test_events_written_with_mail_source(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-10", "Sprint Review")]

        handler.upsert_events(events, now)

        row = db.execute(
            "SELECT source FROM scheduled_items WHERE external_uid='caldav:uid-10'"
        ).fetchone()
        assert row is not None
        assert row[0] == "mail"

    def test_stale_events_cancelled(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        # Insert an event that will not appear in the next sync
        db.execute(
            """INSERT INTO scheduled_items
               (id, item_type, message, due_at, status, source, external_uid, created_at)
               VALUES ('old1', 'event', 'Old event', '2026-04-01T09:00:00+00:00',
                       'pending', 'mail', 'caldav:stale-uid', '2026-04-01T08:00:00+00:00')"""
        )
        db.commit()

        # Sync with zero events
        handler.upsert_events([], now)

        row = db.execute(
            "SELECT status FROM scheduled_items WHERE external_uid='caldav:stale-uid'"
        ).fetchone()
        assert row[0] == "cancelled"

    def test_alert_created_for_upcoming_event(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        # Event starts in 1 hour (within 24h window), not all-day
        events = [
            _make_event(
                "uid-alert",
                "Dentist",
                dtstart=_dt(2026, 4, 1, 9),
                dtend=_dt(2026, 4, 1, 10),
            )
        ]

        handler.upsert_events(events, now)

        row = db.execute(
            "SELECT item_type FROM scheduled_items WHERE external_uid='caldav:uid-alert:alert'"
        ).fetchone()
        assert row is not None
        assert row[0] == "notification"

    def test_alert_not_created_for_all_day(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-allday", "Holiday", all_day=True)]

        handler.upsert_events(events, now)

        row = db.execute(
            "SELECT id FROM scheduled_items WHERE external_uid='caldav:uid-allday:alert'"
        ).fetchone()
        assert row is None

    def test_conflict_notification_created(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [
            _make_event("uid-x", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11)),
            _make_event("uid-y", dtstart=_dt(2026, 4, 1, 10, 30), dtend=_dt(2026, 4, 1, 11, 30)),
        ]

        with patch("capabilities.signal_bridge.emit_capability_signal"):
            handler.upsert_events(events, now)

        canon = ":".join(sorted(["uid-x", "uid-y"]))
        row = db.execute(
            "SELECT item_type FROM scheduled_items WHERE external_uid=?",
            (f"caldav:conflict:{canon}",),
        ).fetchone()
        assert row is not None
        assert row[0] == "notification"

    def test_daily_digest_inserted_once(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)

        handler.upsert_events([], now)
        handler.upsert_events([], now)  # second call must not duplicate

        count = db.execute(
            "SELECT COUNT(*) FROM scheduled_items WHERE external_uid='caldav:daily-digest'"
        ).fetchone()[0]
        assert count == 1

    def test_greeting_inserted_once(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)

        handler.upsert_events([], now)
        handler.upsert_events([], now)

        count = db.execute(
            "SELECT COUNT(*) FROM scheduled_items WHERE external_uid='caldav:greeting'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# list_events tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListEvents:
    def test_returns_events_within_date_range(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-list-1", dtstart=_dt(2026, 4, 2, 9))]
        handler.upsert_events(events, now)

        result = handler.list_events({
            "date_from": "2026-04-01T00:00:00+00:00",
            "date_to": "2026-04-30T00:00:00+00:00",
        })

        assert result.get("count", 0) >= 1
        uids = [e["uid"] for e in result["events"]]
        assert "uid-list-1" in uids

    def test_filters_by_calendar_name(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [
            _make_event("uid-work", cal_name="Work", dtstart=_dt(2026, 4, 3, 9)),
            _make_event("uid-personal", cal_name="Personal", dtstart=_dt(2026, 4, 3, 10)),
        ]
        handler.upsert_events(events, now)

        result = handler.list_events({"calendar_name": "work"})  # lower-case

        uids = [e["uid"] for e in result["events"]]
        assert "uid-work" in uids
        assert "uid-personal" not in uids


# ---------------------------------------------------------------------------
# get_event tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetEvent:
    def test_returns_error_when_uid_missing(self):
        handler = _make_handler()
        result = handler.get_event({})
        assert "error" in result

    def test_returns_event_when_found(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-get-1")]
        handler.upsert_events(events, now)

        result = handler.get_event({"uid": "uid-get-1"})

        assert "event" in result
        assert result["event"]["uid"] == "uid-get-1"

    def test_returns_error_when_not_found(self, db):
        handler = _make_handler()
        result = handler.get_event({"uid": "nonexistent-uid"})
        assert "error" in result


# ---------------------------------------------------------------------------
# find_free_slots tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindFreeSlots:
    def test_gap_between_two_events(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 1, 0)
        events = [
            _make_event("uid-s1", dtstart=_dt(2026, 4, 1, 9), dtend=_dt(2026, 4, 1, 10)),
            _make_event("uid-s2", dtstart=_dt(2026, 4, 1, 11), dtend=_dt(2026, 4, 1, 12)),
        ]
        handler.upsert_events(events, now)

        # Patch user tz to UTC for determinism
        with patch(
            "capabilities.mail_capability.caldav_handler._get_user_tz",
            return_value=None,
        ):
            result = handler.find_free_slots({
                "date_from": "2026-04-01T00:00:00+00:00",
                "date_to": "2026-04-01T23:59:59+00:00",
                "working_hours_start": 8,
                "working_hours_end": 18,
                "min_duration_minutes": 30,
            })

        assert result.get("count", 0) > 0
        # The 10:00-11:00 gap should appear
        starts = [s["start"] for s in result["free_slots"]]
        assert any("10:00" in s for s in starts)

    def test_fully_booked_returns_no_slots(self, db):
        handler = _make_handler()
        now = _dt(2026, 4, 2, 0)
        # One event that covers the entire working day
        events = [
            _make_event(
                "uid-full",
                dtstart=_dt(2026, 4, 2, 8),
                dtend=_dt(2026, 4, 2, 18),
            )
        ]
        handler.upsert_events(events, now)

        with patch(
            "capabilities.mail_capability.caldav_handler._get_user_tz",
            return_value=None,
        ):
            result = handler.find_free_slots({
                "date_from": "2026-04-02T00:00:00+00:00",
                "date_to": "2026-04-02T23:59:59+00:00",
                "working_hours_start": 8,
                "working_hours_end": 18,
                "min_duration_minutes": 30,
            })

        assert result.get("count", 0) == 0


# ---------------------------------------------------------------------------
# create_event validation tests (no real server)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateEvent:
    def test_returns_error_when_summary_missing(self):
        handler = _make_handler()
        client = MagicMock()
        result = handler.create_event(client, {
            "dtstart": "2026-04-01T09:00:00+00:00",
            "dtend": "2026-04-01T10:00:00+00:00",
        })
        assert "error" in result
        assert "summary" in result["error"].lower()

    def test_returns_error_when_dtstart_missing(self):
        handler = _make_handler()
        client = MagicMock()
        result = handler.create_event(client, {
            "summary": "Test",
            "dtend": "2026-04-01T10:00:00+00:00",
        })
        assert "error" in result
        assert "dtstart" in result["error"].lower()


# ---------------------------------------------------------------------------
# open_client test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenClient:
    def test_raises_when_caldav_unavailable(self):
        handler = _make_handler()
        with patch(
            "capabilities.mail_capability.caldav_handler._CALDAV_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError, match="caldav"):
                handler.open_client("https://example.com", "user", "pass")

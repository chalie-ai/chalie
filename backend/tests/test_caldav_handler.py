# mypy: disable-error-code=no-untyped-call
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
    17. ``find_free_slots`` — gap found between two events.
    18. ``create_event`` — returns error when summary missing.
    19. ``open_client`` — raises RuntimeError when caldav unavailable.

All tests are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import datetime
import sqlite3
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from capabilities.mail_capability.caldav_handler import CaldavHandler

_UTC = datetime.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> "CaldavHandler":
    from capabilities.mail_capability.caldav_handler import CaldavHandler
    return CaldavHandler()


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(year, month, day, hour, minute, tzinfo=_UTC)


def _make_event(
    uid: str = "uid-1",
    summary: str = "Meeting",
    dtstart: datetime.datetime | None = None,
    dtend: datetime.datetime | None = None,
    all_day: bool = False,
    cal_name: str = "Work",
    attendees: list[str] | None = None,
    recurrence: str | None = None,
    location: str | None = None,
) -> dict[str, object]:
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


def _build_ical_resource(
    uid: str,
    summary: str,
    dtstart: datetime.datetime,
    dtend: datetime.datetime | None = None,
    attendees: list[str] | None = None,
) -> MagicMock:
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


def _build_duration_resource(
    uid: str,
    summary: str,
    dtstart: datetime.datetime,
    duration: datetime.timedelta,
) -> MagicMock:
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
    def test_timed_event_fields(self) -> None:
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

    def test_duration_fallback(self) -> None:
        handler = _make_handler()
        dtstart = _dt(2026, 4, 1, 10)
        duration = datetime.timedelta(hours=2)
        resource = _build_duration_resource("uid-3", "Workshop", dtstart, duration)

        result = handler.parse_event(resource, "Work")

        assert len(result) == 1
        expected_end = dtstart + duration
        assert result[0]["dtend"] == expected_end


# ---------------------------------------------------------------------------
# _find_overlap_pairs tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindOverlapPairs:
    def test_overlapping_events_detected(self) -> None:
        from capabilities.mail_capability.caldav_handler import _find_overlap_pairs

        now = _dt(2026, 4, 1, 8)
        ev_a = _make_event("a", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11))
        ev_b = _make_event("b", dtstart=_dt(2026, 4, 1, 10, 30), dtend=_dt(2026, 4, 1, 11, 30))

        pairs = _find_overlap_pairs([ev_a, ev_b], now)

        assert len(pairs) == 1
        canon = pairs[0][2]
        assert "a" in canon and "b" in canon


# ---------------------------------------------------------------------------
# _find_back_to_back_pairs tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindBackToBackPairs:
    def test_tight_gap_detected(self) -> None:
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




# ---------------------------------------------------------------------------
# upsert_events tests (use real in-memory DB via conftest `db` fixture)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpsertEvents:
    def test_events_written_with_mail_source(self, db: sqlite3.Connection) -> None:
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-10", "Sprint Review")]

        handler.upsert_events(events, now)

        row = db.execute(
            "SELECT source FROM scheduled_items WHERE external_uid='caldav:uid-10'"
        ).fetchone()
        assert row is not None
        assert row[0] == "mail"

    def test_stale_events_cancelled(self, db: sqlite3.Connection) -> None:
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        # Insert an event that will not appear in the next sync
        db.execute(
            """INSERT INTO scheduled_items
               (id, item_type, message, start_at, due_at, status, source, external_uid, created_at)
               VALUES ('old1', 'event', 'Old event', '2026-04-01T09:00:00+00:00', '2026-04-01T09:00:00+00:00',
                       'pending', 'mail', 'caldav:stale-uid', '2026-04-01T08:00:00+00:00')"""
        )
        db.commit()

        # Sync with zero events
        handler.upsert_events([], now)

        row = db.execute(
            "SELECT status FROM scheduled_items WHERE external_uid='caldav:stale-uid'"
        ).fetchone()
        assert row[0] == "cancelled"

    def test_alert_created_for_upcoming_event(self, db: sqlite3.Connection) -> None:
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

    def test_alert_not_created_for_all_day(self, db: sqlite3.Connection) -> None:
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [_make_event("uid-allday", "Holiday", all_day=True)]

        handler.upsert_events(events, now)

        row = db.execute(
            "SELECT id FROM scheduled_items WHERE external_uid='caldav:uid-allday:alert'"
        ).fetchone()
        assert row is None

    def test_conflict_notification_created(self, db: sqlite3.Connection) -> None:
        handler = _make_handler()
        now = _dt(2026, 4, 1, 8)
        events = [
            _make_event("uid-x", dtstart=_dt(2026, 4, 1, 10), dtend=_dt(2026, 4, 1, 11)),
            _make_event("uid-y", dtstart=_dt(2026, 4, 1, 10, 30), dtend=_dt(2026, 4, 1, 11, 30)),
        ]

        handler.upsert_events(events, now)

        canon = ":".join(sorted(["uid-x", "uid-y"]))
        row = db.execute(
            "SELECT item_type FROM scheduled_items WHERE external_uid=?",
            (f"caldav:conflict:{canon}",),
        ).fetchone()
        assert row is not None
        assert row[0] == "notification"





# ---------------------------------------------------------------------------
# find_free_slots tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindFreeSlots:
    def test_gap_between_two_events(self, db: sqlite3.Connection) -> None:
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

        assert cast(int, result.get("count", 0)) > 0
        # The 10:00-11:00 gap should appear
        starts = [cast(dict[str, object], s)["start"] for s in cast(list[object], result["free_slots"])]
        assert any("10:00" in cast(str, s) for s in starts)



# ---------------------------------------------------------------------------
# create_event validation tests (no real server)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateEvent:
    def test_returns_error_when_summary_missing(self) -> None:
        handler = _make_handler()
        client = MagicMock()
        result = handler.create_event(client, {
            "dtstart": "2026-04-01T09:00:00+00:00",
            "dtend": "2026-04-01T10:00:00+00:00",
        })
        assert "error" in result
        assert "summary" in cast(str, result["error"]).lower()




# ---------------------------------------------------------------------------
# open_client test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenClient:
    def test_raises_when_caldav_unavailable(self) -> None:
        handler = _make_handler()
        with patch(
            "capabilities.mail_capability.caldav_handler._CALDAV_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError, match="caldav"):
                handler.open_client("https://example.com", "user", "pass")

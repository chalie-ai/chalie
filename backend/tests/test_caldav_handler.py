# mypy: disable-error-code=no-untyped-call
"""
Feature tests for :class:`capabilities.mail_capability.caldav_handler.CaldavHandler`.

TKT-1434 ripped the CalDAV-to-``scheduled_items`` mirror out entirely (the
dumb-cron rewrite made ``scheduled_items`` a prompt-only table with zero
CalDAV columns). ``upsert_events``, ``_find_overlap_pairs``, and
``_find_back_to_back_pairs`` — the mirror-sync machinery and its overlap/
back-to-back conflict-notification helpers — no longer exist; the tests that
covered them are deleted rather than adapted, per project policy (behavior
that no longer exists gets no test, however "adapted"). ``find_free_slots``
and ``get_attendees`` are decommissioned reads that now return one clean,
deliberate error instead of querying the dropped mirror.

Coverage:
    1.  ``parse_event`` — timed event fields are populated correctly.
    2.  ``parse_event`` — DURATION used when DTEND absent.
    3.  ``find_free_slots`` — returns the migrating-away error (mirror gone).
    4.  ``get_attendees`` — returns the same migrating-away error.
    5.  ``create_event`` — returns error when summary missing.
    6.  ``open_client`` — raises RuntimeError when caldav unavailable.

All tests are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import datetime
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
# find_free_slots / get_attendees — decommissioned reads (TKT-1434)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecommissionedReads:
    def test_find_free_slots_returns_migrating_away_error(self) -> None:
        from capabilities.mail_capability.caldav_handler import (
            _ERR_CALENDAR_READS_MIGRATING,
        )

        handler = _make_handler()

        result = handler.find_free_slots({
            "date_from": "2026-04-01T00:00:00+00:00",
            "date_to": "2026-04-01T23:59:59+00:00",
            "working_hours_start": 8,
            "working_hours_end": 18,
            "min_duration_minutes": 30,
        })

        assert result.get("error") == _ERR_CALENDAR_READS_MIGRATING

    def test_get_attendees_returns_migrating_away_error(self) -> None:
        from capabilities.mail_capability.caldav_handler import (
            _ERR_CALENDAR_READS_MIGRATING,
        )

        handler = _make_handler()

        result = handler.get_attendees({"uid": "some-event-uid"})

        assert result.get("error") == _ERR_CALENDAR_READS_MIGRATING


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

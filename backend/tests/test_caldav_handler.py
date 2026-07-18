# mypy: disable-error-code=no-untyped-call
"""
Feature tests for :class:`capabilities.mail_capability.caldav_handler.CaldavHandler`.

The CalDAV-to-``scheduled_items`` mirror was ripped out entirely (the
dumb-cron rewrite made ``scheduled_items`` a prompt-only table with zero
CalDAV columns). Calendar reads (``list_events`` / ``get_event``) have since
been re-homed onto direct, live CalDAV queries — their behavioral proof runs
against a live CalDAV server in the feature-test slice, not here.
``find_free_slots`` and ``get_attendees`` computed over the removed mirror and
have no live replacement in scope; they stay registered but return one honest
"unsupported" error.

Coverage (parsing + error-shape units; no network):
    1.  ``parse_event`` — timed event fields are populated correctly.
    2.  ``parse_event`` — DURATION used when DTEND absent.
    3.  ``find_free_slots`` — returns the unsupported-operation error.
    4.  ``get_attendees`` — returns the same unsupported-operation error.
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
# find_free_slots / get_attendees — out-of-scope stubs (no live replacement)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnsupportedOps:
    def test_find_free_slots_returns_unsupported_error(self) -> None:
        from capabilities.mail_capability.caldav_handler import _ERR_OP_UNSUPPORTED

        handler = _make_handler()

        result = handler.find_free_slots({
            "date_from": "2026-04-01T00:00:00+00:00",
            "date_to": "2026-04-01T23:59:59+00:00",
            "working_hours_start": 8,
            "working_hours_end": 18,
            "min_duration_minutes": 30,
        })

        assert result.get("error") == _ERR_OP_UNSUPPORTED

    def test_get_attendees_returns_unsupported_error(self) -> None:
        from capabilities.mail_capability.caldav_handler import _ERR_OP_UNSUPPORTED

        handler = _make_handler()

        result = handler.get_attendees({"uid": "some-event-uid"})

        assert result.get("error") == _ERR_OP_UNSUPPORTED


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


# ---------------------------------------------------------------------------
# _resolve_window — list window bounds (whole-day upper bound + inversion guard)
# ---------------------------------------------------------------------------

_NOW = _dt(2026, 7, 12, 12, 0)


def _ev(uid: str, summary: str, dtstart: datetime.datetime) -> dict[str, object]:
    return {"uid": uid, "summary": summary, "dtstart": dtstart, "dtend": dtstart}


@pytest.mark.unit
class TestResolveWindow:
    def _window(self, params: dict[str, object]) -> tuple[datetime.datetime, datetime.datetime]:
        handler = _make_handler()
        with patch("capabilities.mail_capability.caldav_handler.utc_now", return_value=_NOW):
            return handler._resolve_window(params)

    def test_date_only_upper_bound_covers_whole_day(self) -> None:
        # CalendarAbility resolves a bare date / day-name to midnight; the window
        # must extend through the whole day, else that day's afternoon is missed.
        _start, end = self._window({"date_to": "2026-08-01T00:00:00+00:00"})
        assert end == _dt(2026, 8, 2, 0, 0)

    def test_explicit_time_upper_bound_is_preserved(self) -> None:
        _start, end = self._window({"date_to": "2026-08-01T15:30:00+00:00"})
        assert end == _dt(2026, 8, 1, 15, 30)

    def test_lone_past_upper_bound_is_swapped_not_inverted(self) -> None:
        # Only date_to, in the past → start defaults to now; the window would
        # invert and be handed to date_search. It must be swapped to stay valid.
        start, end = self._window({"date_to": "2026-06-01T00:00:00+00:00"})
        assert start < end
        assert start == _dt(2026, 6, 2, 0, 0)  # whole-day applied before the swap
        assert end == _NOW

    def test_default_window_is_now_plus_seven_days(self) -> None:
        start, end = self._window({})
        assert start == _NOW
        assert end == _NOW + datetime.timedelta(days=7)

    def test_non_utc_user_same_day_window_extended(self) -> None:
        # Non-UTC user: bare date '2026-07-19' in Europe/Malta (+02:00) parses to
        # 2026-07-18T22:00:00Z. Local-midnight must still trigger the extension so
        # the window covers the whole local day.
        with patch(
            "services.locale_service.get_timezone_name", return_value="Europe/Malta"
        ):
            _start, end = self._window(
                {"date_from": "2026-07-18T22:00:00+00:00", "date_to": "2026-07-18T22:00:00+00:00"},
            )
        # '2026-07-18T22:00:00Z' is 2026-07-19 00:00 in Europe/Malta → extended by 1 day
        expected_end = _dt(2026, 7, 19, 22, 0)
        assert end == expected_end

    def test_non_utc_non_midnight_not_extended(self) -> None:
        # Non-midnight explicit time (08:30 UTC) must NOT be extended, even for a
        # non-UTC user — the user explicitly set a time of day.
        with patch(
            "services.locale_service.get_timezone_name", return_value="Europe/Malta"
        ):
            _start, end = self._window({"date_to": "2026-07-19T08:30:00+00:00"})
        assert end == _dt(2026, 7, 19, 8, 30)

# ---------------------------------------------------------------------------
# _representative_per_uid — recurrence occurrences collapse to one occurrence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRepresentativePerUid:
    def test_recurring_series_collapses_to_soonest_upcoming(self) -> None:
        handler = _make_handler()
        soon = _dt(2026, 7, 13, 9, 0)
        events = [
            _ev("series", "Standup", _dt(2026, 7, 2, 9, 0)),   # past occurrence
            _ev("series", "Standup", soon),                    # soonest upcoming
            _ev("series", "Standup", _dt(2026, 7, 20, 9, 0)),  # later upcoming
        ]
        result = handler._representative_per_uid(events, _NOW)
        assert len(result) == 1
        assert result[0]["dtstart"] == soon

    def test_all_past_series_returns_most_recent(self) -> None:
        handler = _make_handler()
        recent = _dt(2026, 7, 5, 9, 0)
        events = [
            _ev("gone", "Old", _dt(2026, 6, 20, 9, 0)),
            _ev("gone", "Old", recent),
        ]
        result = handler._representative_per_uid(events, _NOW)
        assert len(result) == 1
        assert result[0]["dtstart"] == recent

    def test_distinct_uids_are_each_kept(self) -> None:
        handler = _make_handler()
        events = [
            _ev("a", "Lunch", _dt(2026, 7, 13, 12, 0)),
            _ev("b", "Review", _dt(2026, 7, 14, 15, 0)),
        ]
        result = handler._representative_per_uid(events, _NOW)
        assert {str(e["uid"]) for e in result} == {"a", "b"}

    def test_uidless_events_pass_through(self) -> None:
        handler = _make_handler()
        events = [
            _ev("", "One", _dt(2026, 7, 13, 9, 0)),
            _ev("", "Two", _dt(2026, 7, 14, 9, 0)),
        ]
        result = handler._representative_per_uid(events, _NOW)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _ambiguous_title_error — the message carries every candidate uid
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAmbiguousTitleError:
    def test_message_lists_every_candidate_uid(self) -> None:
        handler = _make_handler()
        matches = [
            _ev("uid-1", "Standup", _dt(2026, 7, 13, 9, 0)),
            _ev("uid-2", "Standup review", _dt(2026, 7, 14, 9, 0)),
        ]
        msg = handler._ambiguous_title_error("standup", matches)
        assert "2 events match" in msg
        assert "uid-1" in msg and "uid-2" in msg

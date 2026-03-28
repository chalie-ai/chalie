"""
Unit tests for :class:`capabilities.caldav_capability.capability.CaldavCapability`.

No real CalDAV connections, database access, or external service calls are
made — all external interactions are replaced with
:class:`unittest.mock.MagicMock` or :func:`unittest.mock.patch`.

Coverage:
    1.  ``get_id()`` returns ``"caldav"``.
    2.  ``configure()`` raises ``ValueError`` when required credential fields are missing.
    3.  ``configure()`` raises ``ValueError`` when ``connect()`` returns ``False``.
    4.  ``connect()`` returns ``False`` when ``caldav.DAVClient`` raises an exception.
    5.  ``ingest()`` returns ``[]`` when the capability is not connected.
    6.  ``ingest()`` returns dicts containing all 9 required event keys.
    7.  ``ingest()`` sets ``all_day=True`` for events with date-only ``DTSTART``.
    8.  ``get_tools()`` returns exactly 7 dicts each with ``name`` and ``handler`` keys.
    9.  Every tool handler returns a dict with an ``error`` key when not connected.
    10. ``ingest()`` calls ``KnowledgeService.store`` with ``kind='fact'``,
        ``entity='calendar'``.

All tests are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_capability():
    """Instantiate and return a fresh :class:`CaldavCapability`.

    Uses a deferred import so that the module is not loaded at collection
    time before any test-level patches are in place.

    Returns:
        capabilities.caldav_capability.capability.CaldavCapability:
            A newly created instance with ``_connected=False``.
    """
    from capabilities.caldav_capability.capability import CaldavCapability

    return CaldavCapability()


def _build_ical_calendar(uid: str, summary: str, dtstart, dtend=None):
    """Build a minimal :class:`icalendar.Calendar` wrapping one VEVENT.

    The ``DTSTART`` value controls whether the event is treated as all-day:
    pass a :class:`datetime.date` (non-datetime) for all-day; pass a
    timezone-aware :class:`datetime.datetime` for timed events.

    Args:
        uid:     iCalendar UID string, e.g. ``"event-001"``.
        summary: Human-readable event title.
        dtstart: ``DTSTART`` value — :class:`datetime.date` or
                 :class:`datetime.datetime`.
        dtend:   Optional ``DTEND`` value; omitted when ``None``.

    Returns:
        icalendar.Calendar: Calendar object containing exactly one VEVENT.
    """
    import icalendar

    cal = icalendar.Calendar()
    evt = icalendar.Event()
    evt.add("uid", uid)
    evt.add("summary", summary)
    evt.add("dtstart", dtstart)
    if dtend is not None:
        evt.add("dtend", dtend)
    cal.add_component(evt)
    return cal


def _make_mock_caldav_setup(ical_calendar, cal_name: str = "Test Calendar"):
    """Build a mock CalDAV client hierarchy for a single calendar.

    Creates a chain of mocks:
    ``DAVClient`` → ``principal()`` → ``calendars()`` → ``[mock_calendar]``
    → ``date_search()`` → ``[mock_raw_event]``
    whose ``icalendar_instance`` attribute is *ical_calendar*.

    Args:
        ical_calendar: :class:`icalendar.Calendar` with at least one VEVENT.
        cal_name:      Human-readable name assigned to the mock calendar.

    Returns:
        tuple[MagicMock, MagicMock]: ``(mock_dav_client_class,
        mock_client_instance)``.  Pass ``mock_dav_client_class`` as the
        replacement for ``caldav.DAVClient`` in tests.
    """
    mock_raw_event = MagicMock()
    mock_raw_event.icalendar_instance = ical_calendar

    mock_calendar = MagicMock()
    mock_calendar.name = cal_name
    mock_calendar.date_search.return_value = [mock_raw_event]

    mock_principal = MagicMock()
    mock_principal.calendars.return_value = [mock_calendar]

    mock_client = MagicMock()
    mock_client.principal.return_value = mock_principal

    mock_dav_client_class = MagicMock(return_value=mock_client)
    return mock_dav_client_class, mock_client


# ---------------------------------------------------------------------------
# Context manager stack helpers
# ---------------------------------------------------------------------------

_NOW = datetime.datetime(2026, 3, 27, 10, 0, 0, tzinfo=_UTC)
_DTSTART = datetime.datetime(2026, 3, 27, 14, 0, 0, tzinfo=_UTC)
_DTEND = datetime.datetime(2026, 3, 27, 15, 0, 0, tzinfo=_UTC)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaldavCapability:
    """Unit tests for :class:`CaldavCapability`."""

    # ------------------------------------------------------------------
    # Test 1 — get_id
    # ------------------------------------------------------------------

    def test_get_id_returns_caldav(self):
        """``get_id()`` must return the string ``"caldav"``.

        Verifies that the unique capability identifier matches the value
        declared in ``manifest.yaml`` and required by the capability
        framework.
        """
        cap = _make_capability()
        assert cap.get_id() == "caldav"

    # ------------------------------------------------------------------
    # Test 2 — configure missing fields
    # ------------------------------------------------------------------

    def test_configure_raises_on_invalid_credentials(self):
        """``configure()`` must raise ``ValueError`` when required fields are absent.

        Passes a credentials dict that is missing ``username`` and
        ``password`` and asserts that a ``ValueError`` containing
        ``"missing required credential fields"`` is raised before any
        attempt to store credentials.
        """
        cap = _make_capability()
        with pytest.raises(ValueError, match="missing required credential fields"):
            cap.configure({"provider": "google"})

    # ------------------------------------------------------------------
    # Test 3 — configure raises when connect fails
    # ------------------------------------------------------------------

    def test_configure_raises_when_connect_fails(self):
        """``configure()`` must raise ``ValueError`` when ``connect()`` returns ``False``.

        Patches ``store_credential``, ``connect``, and ``delete_credentials``
        on the capability instance so that no real encryption or network
        activity occurs.  Verifies that ``ValueError`` containing
        ``"Could not connect"`` is raised when the connection test fails.
        """
        cap = _make_capability()
        with patch.object(cap, "store_credential"), \
             patch.object(cap, "connect", return_value=False), \
             patch.object(cap, "delete_credentials"):
            with pytest.raises(ValueError, match="Could not connect"):
                cap.configure(
                    {
                        "provider": "google",
                        "username": "user@example.com",
                        "password": "app-secret",
                    }
                )

    # ------------------------------------------------------------------
    # Test 4 — connect returns False on network error
    # ------------------------------------------------------------------

    def test_connect_returns_false_on_network_error(self):
        """``connect()`` must return ``False`` when the CalDAV client raises.

        Patches ``load_credential`` to supply valid-looking credentials and
        replaces the module-level ``_caldav_lib`` so that ``DAVClient(...)``
        raises ``Exception("Connection refused")``.  Asserts that
        ``connect()`` returns ``False`` and leaves ``is_connected()``
        as ``False``.
        """
        cap = _make_capability()

        with patch.object(
            cap,
            "load_credential",
            side_effect=["google", "user@test.com", "pass123"],
        ), patch(
            "capabilities.caldav_capability.capability._caldav_lib"
        ) as mock_lib, patch(
            "capabilities.caldav_capability.capability._CALDAV_AVAILABLE", True
        ):
            mock_lib.DAVClient.side_effect = Exception("Connection refused")
            result = cap.connect()

        assert result is False
        assert cap.is_connected() is False

    # ------------------------------------------------------------------
    # Test 5 — ingest returns [] when not connected
    # ------------------------------------------------------------------

    def test_ingest_returns_empty_when_not_connected(self):
        """``ingest()`` must return ``[]`` immediately when not connected.

        No mocking of external services is required because the method
        returns before reaching any network or database call when
        ``is_connected()`` is ``False``.
        """
        cap = _make_capability()
        assert cap.is_connected() is False
        result = cap.ingest()
        assert result == []

    # ------------------------------------------------------------------
    # Test 6 — ingest parses all 9 event fields
    # ------------------------------------------------------------------

    def test_ingest_parses_event_fields(self):
        """``ingest()`` must return dicts with all 9 required event keys.

        Constructs a minimal iCalendar ``VEVENT`` and wires it through a
        mock CalDAV client chain.  After ``ingest()``, asserts that the
        returned event dict contains the keys ``uid``, ``summary``,
        ``dtstart``, ``dtend``, ``location``, ``attendees``, ``recurrence``,
        ``all_day``, and ``calendar_name``, and that key values match what
        was provided in the VEVENT.
        """
        cap = _make_capability()
        cap._connected = True  # bypass is_connected() guard

        ical = _build_ical_calendar("uid-001", "Team Meeting", _DTSTART, _DTEND)
        mock_dav_client_class, _ = _make_mock_caldav_setup(ical, "Work")

        with patch.object(
            cap,
            "load_credential",
            side_effect=["google", "user@test.com", "mypass"],
        ), patch.object(
            cap, "_load_config_raw", return_value=None
        ), patch(
            "capabilities.caldav_capability.capability._caldav_lib"
        ) as mock_lib, patch(
            "capabilities.caldav_capability.capability._CALDAV_AVAILABLE", True
        ), patch(
            "capabilities.caldav_capability.capability._ICALENDAR_AVAILABLE", True
        ), patch(
            "services.database_service.get_shared_db_service",
            return_value=MagicMock(),
        ), patch(
            "services.knowledge_service.KnowledgeService",
            return_value=MagicMock(),
        ), patch(
            "services.world_state_service.WorldStateService",
            return_value=MagicMock(),
        ), patch(
            "capabilities.caldav_capability.capability.utc_now",
            return_value=_NOW,
        ):
            mock_lib.DAVClient = mock_dav_client_class
            events = cap.ingest()

        assert len(events) >= 1
        event = events[0]
        required_keys = {
            "uid",
            "summary",
            "dtstart",
            "dtend",
            "location",
            "attendees",
            "recurrence",
            "all_day",
            "calendar_name",
        }
        assert required_keys.issubset(set(event.keys())), (
            f"Missing keys: {required_keys - set(event.keys())}"
        )
        assert event["uid"] == "uid-001"
        assert event["summary"] == "Team Meeting"
        assert event["all_day"] is False
        assert event["calendar_name"] == "Work"

    # ------------------------------------------------------------------
    # Test 7 — ingest marks all-day events
    # ------------------------------------------------------------------

    def test_ingest_marks_all_day_events(self):
        """``ingest()`` must set ``all_day=True`` for date-only ``DTSTART`` events.

        Constructs a ``VEVENT`` whose ``DTSTART`` is a :class:`datetime.date`
        object (not a :class:`datetime.datetime`), which iCalendar represents
        as a "floating date".  Asserts that ``all_day`` is ``True`` in the
        parsed event dict.
        """
        cap = _make_capability()
        cap._connected = True

        # Date-only (not datetime) triggers all_day=True
        dtstart_date = datetime.date(2026, 3, 28)
        ical = _build_ical_calendar("uid-allday", "Holiday", dtstart_date)
        mock_dav_client_class, _ = _make_mock_caldav_setup(ical, "Personal")

        with patch.object(
            cap,
            "load_credential",
            side_effect=["google", "user@test.com", "mypass"],
        ), patch.object(
            cap, "_load_config_raw", return_value=None
        ), patch(
            "capabilities.caldav_capability.capability._caldav_lib"
        ) as mock_lib, patch(
            "capabilities.caldav_capability.capability._CALDAV_AVAILABLE", True
        ), patch(
            "capabilities.caldav_capability.capability._ICALENDAR_AVAILABLE", True
        ), patch(
            "services.database_service.get_shared_db_service",
            return_value=MagicMock(),
        ), patch(
            "services.knowledge_service.KnowledgeService",
            return_value=MagicMock(),
        ), patch(
            "services.world_state_service.WorldStateService",
            return_value=MagicMock(),
        ), patch(
            "capabilities.caldav_capability.capability.utc_now",
            return_value=_NOW,
        ):
            mock_lib.DAVClient = mock_dav_client_class
            events = cap.ingest()

        assert len(events) >= 1, "Expected at least one parsed event"
        assert events[0]["all_day"] is True

    # ------------------------------------------------------------------
    # Test 8 — get_tools returns 6 tools
    # ------------------------------------------------------------------

    def test_get_tools_returns_seven_tools(self):
        """``get_tools()`` must return a list of exactly 7 tool definition dicts.

        Each dict must have at minimum the keys ``name`` (str) and
        ``handler`` (callable), confirming that the tool definitions are
        ready for registration via ``register_tool()``.
        """
        cap = _make_capability()
        tools = cap.get_tools()

        assert isinstance(tools, list), "get_tools() must return a list"
        assert len(tools) == 7, f"Expected 7 tools, got {len(tools)}"
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name' key: {tool}"
            assert "handler" in tool, f"Tool missing 'handler' key: {tool}"
            assert callable(tool["handler"]), (
                f"Tool handler for '{tool['name']}' is not callable"
            )

    # ------------------------------------------------------------------
    # Test 9 — tool handlers return error when not connected
    # ------------------------------------------------------------------

    def test_tool_handler_returns_error_when_not_connected(self):
        """Every tool handler must return a dict with an ``error`` key when not connected.

        Iterates all 6 handlers returned by ``get_tools()`` and calls each
        with minimal arguments (``topic=""``, ``params={}``).  The
        capability is in its default disconnected state.  Asserts that every
        result is a dict containing the ``"error"`` key.
        """
        cap = _make_capability()
        assert cap.is_connected() is False

        tools = cap.get_tools()
        assert len(tools) == 7, "Prerequisite: get_tools() must return 7 tools"

        for tool in tools:
            result = tool["handler"](topic="", params={})
            assert isinstance(result, dict), (
                f"Handler '{tool['name']}' returned non-dict: {type(result)}"
            )
            assert "error" in result, (
                f"Handler '{tool['name']}' returned no 'error' key when disconnected: {result}"
            )

    # ------------------------------------------------------------------
    # Test 10 — ingest calls _upsert_events_to_scheduler
    # ------------------------------------------------------------------

    def test_ingest_calls_upsert_events_to_scheduler(self):
        """``ingest()`` must call ``_upsert_events_to_scheduler()`` to persist events."""
        cap = _make_capability()
        cap._connected = True

        ical = _build_ical_calendar("uid-store-test", "Stand-up", _DTSTART, _DTEND)
        mock_dav_client_class, _ = _make_mock_caldav_setup(ical, "Work")

        with patch.object(
            cap,
            "load_credential",
            side_effect=["google", "user@test.com", "mypass"],
        ), patch.object(
            cap, "_load_config_raw", return_value=None
        ), patch(
            "capabilities.caldav_capability.capability._caldav_lib"
        ) as mock_lib, patch(
            "capabilities.caldav_capability.capability._CALDAV_AVAILABLE", True
        ), patch(
            "capabilities.caldav_capability.capability._ICALENDAR_AVAILABLE", True
        ), patch.object(
            cap, "_upsert_events_to_scheduler"
        ) as mock_upsert, patch(
            "capabilities.caldav_capability.capability.utc_now",
            return_value=_NOW,
        ):
            mock_lib.DAVClient = mock_dav_client_class
            events = cap.ingest()

        assert len(events) >= 1, "Expected at least one parsed event"
        mock_upsert.assert_called_once()
        # First arg should be the events list, second should be the now timestamp
        call_args = mock_upsert.call_args
        assert len(call_args[0][0]) >= 1, "Events list should be non-empty"

    # ------------------------------------------------------------------
    # Test 11 — connect calls _ensure_sync_registration
    # ------------------------------------------------------------------

    def test_connect_calls_ensure_sync_registration(self):
        """``connect()`` must call ``_ensure_sync_registration()`` on success."""
        cap = _make_capability()

        with patch.object(
            cap,
            "load_credential",
            side_effect=["google", "user@test.com", "pass123"],
        ), patch(
            "capabilities.caldav_capability.capability._caldav_lib"
        ) as mock_lib, patch(
            "capabilities.caldav_capability.capability._CALDAV_AVAILABLE", True
        ), patch.object(
            cap, "_ensure_sync_registration"
        ) as mock_reg:
            mock_lib.DAVClient.return_value.principal.return_value = MagicMock()
            result = cap.connect()

        assert result is True
        mock_reg.assert_called_once()

    # ------------------------------------------------------------------
    # Test 12 — disconnect cancels caldav scheduled_items
    # ------------------------------------------------------------------

    def test_disconnect_cancels_scheduled_items(self):
        """``disconnect()`` must cancel all source='caldav' pending items."""
        cap = _make_capability()
        cap._connected = True

        mock_conn = MagicMock()
        mock_db = MagicMock()
        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(cap, "delete_credentials"), \
             patch("services.database_service.get_shared_db_service", return_value=mock_db):
            cap.disconnect()

        assert cap.is_connected() is False
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "cancelled" in sql and "caldav" in sql

    # ------------------------------------------------------------------
    # Test 13 — monitor auto-reconnects
    # ------------------------------------------------------------------

    def test_monitor_auto_reconnects(self):
        """``monitor()`` must call ``connect()`` when not connected."""
        cap = _make_capability()
        assert cap.is_connected() is False

        with patch.object(cap, "connect", return_value=False) as mock_connect, \
             patch.object(cap, "ingest") as mock_ingest:
            cap.monitor()

        mock_connect.assert_called_once()
        mock_ingest.assert_not_called()  # connect failed, so ingest should not run

    def test_monitor_calls_ingest_when_connected(self):
        """``monitor()`` must call ``ingest()`` when connected."""
        cap = _make_capability()
        cap._connected = True

        with patch.object(cap, "ingest", return_value=[]) as mock_ingest:
            cap.monitor()

        mock_ingest.assert_called_once()


# -----------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------


class TestModuleLevelHelpers:
    """Tests for module-level helper functions."""

    @staticmethod
    def _make_event(**overrides):
        base = {
            'uid': 'evt-1', 'summary': 'Team standup',
            'dtstart': datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
            'dtend': datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
            'location': None, 'attendees': [], 'recurrence': None,
            'all_day': False, 'calendar_name': 'Work',
        }
        base.update(overrides)
        return base

    @pytest.mark.unit
    def test_event_line_includes_time_title_calendar(self):
        from capabilities.caldav_capability.capability import _format_event_line

        line = _format_event_line(self._make_event())
        assert "09:00" in line and "09:30" in line
        assert "Team standup" in line and "[Work]" in line
        assert "all-day" in _format_event_line(self._make_event(all_day=True))

    @pytest.mark.unit
    def test_event_line_optional_fields(self):
        from capabilities.caldav_capability.capability import _format_event_line

        full = _format_event_line(self._make_event(
            location="Room 42", attendees=["a@b.com", "c@d.com"],
        ))
        assert "@ Room 42" in full and "(2 attendees)" in full

    @pytest.mark.unit
    def test_safe_int(self):
        from capabilities.caldav_capability.capability import _safe_int
        assert _safe_int("42", 0) == 42
        assert _safe_int(None, 10) == 10
        assert _safe_int("not-a-number", 5) == 5

    @pytest.mark.unit
    def test_events_overlap(self):
        from capabilities.caldav_capability.capability import _events_overlap
        ev_a = {'dtstart': datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                'dtend': datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC)}
        ev_b = {'dtstart': datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
                'dtend': datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC)}
        assert _events_overlap(ev_a, ev_b) is True

        ev_c = {'dtstart': datetime.datetime(2026, 3, 28, 16, 0, tzinfo=_UTC),
                'dtend': datetime.datetime(2026, 3, 28, 17, 0, tzinfo=_UTC)}
        assert _events_overlap(ev_a, ev_c) is False


# --- _next_morning_8am timezone tests ---

class TestNextMorning8am:

    @pytest.mark.unit
    def test_fallback_utc_when_no_timezone(self):
        """Without user timezone, returns 08:00 UTC next day."""
        from capabilities.caldav_capability.capability import _next_morning_8am
        fixed = datetime.datetime(2026, 3, 28, 10, 0, tzinfo=_UTC)
        with patch("capabilities.caldav_capability.capability.utc_now", return_value=fixed), \
             patch("capabilities.caldav_capability.capability._get_user_tz", return_value=None):
            result = _next_morning_8am()
        assert result.hour == 8
        assert result.day == 29

    @pytest.mark.unit
    def test_uses_user_timezone(self):
        """With user timezone, returns 08:00 local → converted to UTC."""
        from zoneinfo import ZoneInfo
        from capabilities.caldav_capability.capability import _next_morning_8am
        tz_ny = ZoneInfo("America/New_York")
        # 2026-03-28 10:00 UTC = 2026-03-28 06:00 ET (before 8am local)
        fixed = datetime.datetime(2026, 3, 28, 10, 0, tzinfo=_UTC)
        with patch("capabilities.caldav_capability.capability.utc_now", return_value=fixed), \
             patch("capabilities.caldav_capability.capability._get_user_tz", return_value=tz_ny):
            result = _next_morning_8am()
        # 8am ET (EDT, -4) = 12:00 UTC, same day
        assert result.hour == 12
        assert result.day == 28

    @pytest.mark.unit
    def test_next_day_when_past_8am_local(self):
        """If it's past 8am local, schedule for next day."""
        from zoneinfo import ZoneInfo
        from capabilities.caldav_capability.capability import _next_morning_8am
        tz_ny = ZoneInfo("America/New_York")
        # 2026-03-28 18:00 UTC = 2026-03-28 14:00 ET (past 8am local)
        fixed = datetime.datetime(2026, 3, 28, 18, 0, tzinfo=_UTC)
        with patch("capabilities.caldav_capability.capability.utc_now", return_value=fixed), \
             patch("capabilities.caldav_capability.capability._get_user_tz", return_value=tz_ny):
            result = _next_morning_8am()
        # Next 8am ET = 2026-03-29 08:00 ET = 12:00 UTC
        assert result.hour == 12
        assert result.day == 29


# --- Back-to-back detection tests ---

class TestBackToBackPairs:

    @pytest.mark.unit
    def test_detects_tight_and_zero_gap(self):
        """Tight gap (3min) and zero gap both detected."""
        from capabilities.caldav_capability.capability import _find_back_to_back_pairs
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)
        events = [
            {'uid': 'a', 'summary': 'Standup',
             'dtstart': datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
             'all_day': False},
            {'uid': 'b', 'summary': 'Design Review',
             'dtstart': datetime.datetime(2026, 3, 28, 9, 33, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 28, 10, 0, tzinfo=_UTC),
             'all_day': False},
            {'uid': 'c', 'summary': 'Sync',
             'dtstart': datetime.datetime(2026, 3, 28, 10, 0, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 28, 10, 30, tzinfo=_UTC),
             'all_day': False},
        ]
        pairs = _find_back_to_back_pairs(events, now)
        assert len(pairs) == 2
        assert pairs[0][2] == 3  # 3 min gap a→b
        assert pairs[1][2] == 0  # zero gap b→c

    @pytest.mark.unit
    def test_ignores_wide_gap_and_all_day(self):
        """Wide gaps (>5min) and all-day events are excluded."""
        from capabilities.caldav_capability.capability import _find_back_to_back_pairs
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)
        events = [
            {'uid': 'x', 'summary': 'Holiday',
             'dtstart': datetime.datetime(2026, 3, 28, 0, 0, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 29, 0, 0, tzinfo=_UTC),
             'all_day': True},
            {'uid': 'a', 'summary': 'Standup',
             'dtstart': datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
             'all_day': False},
            {'uid': 'b', 'summary': 'Lunch',
             'dtstart': datetime.datetime(2026, 3, 28, 12, 0, tzinfo=_UTC),
             'dtend': datetime.datetime(2026, 3, 28, 13, 0, tzinfo=_UTC),
             'all_day': False},
        ]
        pairs = _find_back_to_back_pairs(events, now)
        assert len(pairs) == 0



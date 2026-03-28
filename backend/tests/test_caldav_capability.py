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
    8.  ``get_tools()`` returns exactly 6 dicts each with ``name`` and ``handler`` keys.
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

    def test_get_tools_returns_five_tools(self):
        """``get_tools()`` must return a list of exactly 6 tool definition dicts..

        Each dict must have at minimum the keys ``name`` (str) and
        ``handler`` (callable), confirming that the tool definitions are
        ready for registration via ``register_tool()``.
        """
        cap = _make_capability()
        tools = cap.get_tools()

        assert isinstance(tools, list), "get_tools() must return a list"
        assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"
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
        assert len(tools) == 6, "Prerequisite: get_tools() must return 6 tools"

        for tool in tools:
            result = tool["handler"](topic="", params={})
            assert isinstance(result, dict), (
                f"Handler '{tool['name']}' returned non-dict: {type(result)}"
            )
            assert "error" in result, (
                f"Handler '{tool['name']}' returned no 'error' key when disconnected: {result}"
            )

    # ------------------------------------------------------------------
    # Test 10 — ingest stores a knowledge fact
    # ------------------------------------------------------------------

    def test_ingest_stores_knowledge_fact(self):
        """``ingest()`` must call ``KnowledgeService.store`` with ``kind='fact'``, ``entity='calendar'``.

        Wires a single VEVENT through the mock CalDAV pipeline and captures
        all calls to ``KnowledgeService.store``.  Asserts that at least one
        call was made with ``kind='fact'`` and ``entity='calendar'``,
        confirming that event data is persisted to the knowledge store.
        """
        cap = _make_capability()
        cap._connected = True

        ical = _build_ical_calendar("uid-store-test", "Stand-up", _DTSTART, _DTEND)
        mock_dav_client_class, _ = _make_mock_caldav_setup(ical, "Work")

        mock_ks_instance = MagicMock()
        mock_ks_class = MagicMock(return_value=mock_ks_instance)

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
            "services.knowledge_service.KnowledgeService", mock_ks_class
        ), patch(
            "services.world_state_service.WorldStateService",
            return_value=MagicMock(),
        ), patch(
            "capabilities.caldav_capability.capability.utc_now",
            return_value=_NOW,
        ):
            mock_lib.DAVClient = mock_dav_client_class
            cap.ingest()

        assert mock_ks_instance.store.called, (
            "KnowledgeService.store was never called during ingest()"
        )

        # Inspect the first store() call — all arguments are keyword-only in
        # the implementation so check call_args.kwargs; fall back to
        # positional args for robustness.
        first_call = mock_ks_instance.store.call_args_list[0]
        kwargs = first_call.kwargs if first_call.kwargs else {}
        args = first_call.args if first_call.args else ()

        kind_val = kwargs.get("kind") or (args[0] if args else None)
        entity_val = kwargs.get("entity") or (args[1] if len(args) > 1 else None)

        assert kind_val == "fact", (
            f"Expected kind='fact', got kind={kind_val!r}"
        )
        assert entity_val == "calendar", (
            f"Expected entity='calendar', got entity={entity_val!r}"
        )


# -----------------------------------------------------------------------
# Schedule hint formatting
# -----------------------------------------------------------------------


class TestFormatScheduleHint:
    """Tests for :func:`_format_event_line` and :func:`_format_schedule_hint`."""

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
        # all-day replaces time range
        assert "all-day" in _format_event_line(self._make_event(all_day=True))

    @pytest.mark.unit
    def test_event_line_optional_fields(self):
        from capabilities.caldav_capability.capability import _format_event_line

        full = _format_event_line(self._make_event(
            location="Room 42", attendees=["a@b.com", "c@d.com"],
        ))
        assert "@ Room 42" in full and "(2 attendees)" in full
        # single attendee still uses plural
        assert "(1 attendees)" in _format_event_line(
            self._make_event(attendees=["a@b.com"]))
        # no optional fields
        bare = _format_event_line(
            self._make_event(location=None, attendees=[], calendar_name=None))
        assert "@" not in bare and "attendee" not in bare and "[" not in bare

    @pytest.mark.unit
    def test_hint_empty_and_single(self):
        from capabilities.caldav_capability.capability import _format_schedule_hint

        assert _format_schedule_hint([]) == "No upcoming events in the next 24 hours."
        hint = _format_schedule_hint([self._make_event()])
        assert "Next 24h (1 events):" in hint and "Team standup" in hint

    @pytest.mark.unit
    def test_hint_caps_at_max_events(self):
        from capabilities.caldav_capability.capability import _format_schedule_hint

        events = [
            self._make_event(
                uid=f"evt-{i}", summary=f"Meeting {i}",
                dtstart=datetime.datetime(2026, 3, 28, 8 + i, 0, tzinfo=_UTC),
                dtend=datetime.datetime(2026, 3, 28, 8 + i, 30, tzinfo=_UTC),
            ) for i in range(8)
        ]
        hint = _format_schedule_hint(events, max_events=5)
        assert "(8 events)" in hint and "Meeting 4" in hint
        assert "Meeting 5" not in hint and "(+3 more)" in hint

    @pytest.mark.unit
    def test_hint_rich_event(self):
        from capabilities.caldav_capability.capability import _format_schedule_hint

        hint = _format_schedule_hint([self._make_event(
            location="HQ", attendees=["a@b.com", "c@d.com"],
        )])
        assert "[Work]" in hint and "@ HQ" in hint and "(2 attendees)" in hint


# -----------------------------------------------------------------------
# First-connect greeting
# -----------------------------------------------------------------------


class TestFirstConnectGreeting:
    """Tests for :meth:`CaldavCapability._maybe_send_greeting`."""

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
    @patch('services.memory_client.MemoryClientService')
    def test_first_ingest_sends_greeting(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None  # no flag yet
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)
        events = [self._make_event()]

        result = cap._maybe_send_greeting(events, now)
        assert result is True
        store.rpush.assert_called_once()
        # Verify the prompt-queue push contains schedule data
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "CALENDAR CONNECTED" in payload['prompt']
        assert "Team standup" in payload['prompt']
        assert payload['metadata']['source'] == 'capability_greeting'
        # Flag was set
        store.set.assert_called_once_with('caldav:greeting_sent', '1')

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_second_ingest_skips_greeting(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = "1"  # flag already set
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)

        result = cap._maybe_send_greeting([self._make_event()], now)
        assert result is False
        store.rpush.assert_not_called()

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_empty_schedule_still_sends_greeting(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)

        result = cap._maybe_send_greeting([], now)
        assert result is True
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "No upcoming events" in payload['prompt']


# -----------------------------------------------------------------------
# Upcoming-event alerts
# -----------------------------------------------------------------------


class TestUpcomingEventAlerts:
    """Tests for :meth:`CaldavCapability._maybe_send_upcoming_alerts`."""

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
    @patch('services.memory_client.MemoryClientService')
    def test_alert_fires_for_imminent_event(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None  # not alerted yet
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 40, tzinfo=_UTC)
        # Event at 09:00 is 20 min away — within 15-30 window
        result = cap._maybe_send_upcoming_alerts([self._make_event()], now)
        assert result == 1
        store.rpush.assert_called_once()
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "CALENDAR ALERT" in payload['prompt']
        assert "20 minutes" in payload['prompt']
        store.setex.assert_called_once()

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_already_alerted_uid_skips(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = "1"  # already alerted
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 40, tzinfo=_UTC)
        result = cap._maybe_send_upcoming_alerts([self._make_event()], now)
        assert result == 0
        store.rpush.assert_not_called()

    @pytest.mark.unit
    def test_no_imminent_events_returns_zero(self):
        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 7, 0, tzinfo=_UTC)
        # Event at 09:00 is 120 min away — outside window
        result = cap._maybe_send_upcoming_alerts(
            [self._make_event()], now)
        assert result == 0


class TestAttendeeContext:
    """Tests for :meth:`CaldavCapability._build_attendee_context`."""

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
    def test_no_attendees_returns_empty(self):
        cap = _make_capability()
        result = cap._build_attendee_context(self._make_event(attendees=[]))
        assert result == ""

    @pytest.mark.unit
    @patch('services.database_service.get_shared_db_service')
    @patch('services.knowledge_service.KnowledgeService')
    def test_attendee_with_knowledge(self, mock_ks_cls, mock_db):
        mock_ks = MagicMock()
        mock_ks.recall.return_value = [
            {"value": "prefers morning meetings"},
            {"value": "works on backend team"},
        ]
        mock_ks_cls.return_value = mock_ks

        cap = _make_capability()
        event = self._make_event(attendees=["sarah.chen@example.com"])
        result = cap._build_attendee_context(event)
        assert "Sarah Chen" in result
        assert "prefers morning meetings" in result
        assert "Attendee context:" in result

    @pytest.mark.unit
    @patch('services.database_service.get_shared_db_service')
    @patch('services.knowledge_service.KnowledgeService')
    def test_attendee_no_knowledge_returns_empty(self, mock_ks_cls, mock_db):
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_ks_cls.return_value = mock_ks

        cap = _make_capability()
        event = self._make_event(attendees=["unknown@example.com"])
        result = cap._build_attendee_context(event)
        assert result == ""

    @pytest.mark.unit
    @patch('services.database_service.get_shared_db_service')
    @patch('services.knowledge_service.KnowledgeService')
    def test_limits_attendees_to_max(self, mock_ks_cls, mock_db):
        mock_ks = MagicMock()
        mock_ks.recall.return_value = [{"value": "context"}]
        mock_ks_cls.return_value = mock_ks

        cap = _make_capability()
        many = [f"person{i}@example.com" for i in range(10)]
        event = self._make_event(attendees=many)
        cap._build_attendee_context(event)
        # Should only query for first 3 attendees
        assert mock_ks.recall.call_count == 3

    @pytest.mark.unit
    def test_survives_exception(self):
        cap = _make_capability()
        event = self._make_event(attendees=["a@b.com"])
        # No db service available — should return empty, not raise
        result = cap._build_attendee_context(event)
        assert result == ""

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_alert_includes_attendee_context(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 40, tzinfo=_UTC)
        event = self._make_event(attendees=["sarah@example.com"])
        with patch.object(
            cap, '_build_attendee_context',
            return_value="Attendee context:\n- Sarah: prefers concise updates",
        ):
            cap._maybe_send_upcoming_alerts([event], now)

        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "Attendee context:" in payload['prompt']
        assert "Sarah" in payload['prompt']


class TestDailyDigest:
    """Tests for :meth:`CaldavCapability._maybe_send_daily_digest`."""

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
    @patch('services.memory_client.MemoryClientService')
    def test_digest_fires_first_time_today(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None  # no digest yet today
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 6, 0, tzinfo=_UTC)
        result = cap._maybe_send_daily_digest([self._make_event()], now)
        assert result is True
        store.rpush.assert_called_once()
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "DAILY CALENDAR DIGEST" in payload['prompt']
        assert "Team standup" in payload['prompt']
        store.setex.assert_called_once_with("caldav:digest:2026-03-28", 86400, "1")

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_digest_skips_if_already_sent(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = "1"  # digest already sent today
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 12, 0, tzinfo=_UTC)
        result = cap._maybe_send_daily_digest([self._make_event()], now)
        assert result is False
        store.rpush.assert_not_called()

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_digest_with_empty_day(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 6, 0, tzinfo=_UTC)
        # No events today — digest should still fire with "no events" message
        result = cap._maybe_send_daily_digest([], now)
        assert result is True
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "No upcoming events" in payload['prompt']


class TestConflictAlert:
    """Tests for :meth:`CaldavCapability._maybe_send_conflict_alert`."""

    @staticmethod
    def _make_event(**overrides):
        base = {
            'uid': 'evt-1', 'summary': 'Meeting A',
            'dtstart': datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
            'dtend': datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC),
            'location': None, 'attendees': [], 'recurrence': None,
            'all_day': False, 'calendar_name': 'Work',
        }
        base.update(overrides)
        return base

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_conflict_alert_fires_for_overlapping_events(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = None  # not alerted yet
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 12, 0, tzinfo=_UTC)
        ev_a = self._make_event(uid='a', summary='Design Review',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))
        ev_b = self._make_event(uid='b', summary='Sprint Planning',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC))
        result = cap._maybe_send_conflict_alert([ev_a, ev_b], now)
        assert result == 1
        store.rpush.assert_called_once()
        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert "CALENDAR CONFLICT" in payload['prompt']
        assert "Design Review" in payload['prompt']
        assert "Sprint Planning" in payload['prompt']
        store.setex.assert_called_once()

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_conflict_alert_deduped(self, mock_mcs):
        store = MagicMock()
        store.get.return_value = "1"  # already alerted
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 12, 0, tzinfo=_UTC)
        ev_a = self._make_event(uid='a',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))
        ev_b = self._make_event(uid='b',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC))
        result = cap._maybe_send_conflict_alert([ev_a, ev_b], now)
        assert result == 0
        store.rpush.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize("label,now_hour,ev_a_kw,ev_b_kw", [
        ("non_overlapping", 12,
         dict(uid='a', dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC)),
         dict(uid='b', dtstart=datetime.datetime(2026, 3, 28, 16, 0, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 28, 17, 0, tzinfo=_UTC))),
        ("past_events", 16,
         dict(uid='a', dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC)),
         dict(uid='b', dtstart=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC))),
        ("all_day_excluded", 12,
         dict(uid='a', all_day=True, dtstart=datetime.datetime(2026, 3, 28, 0, 0, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 29, 0, 0, tzinfo=_UTC)),
         dict(uid='b', dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
              dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))),
    ], ids=["non_overlapping", "past_events", "all_day_excluded"])
    def test_no_alert_cases(self, label, now_hour, ev_a_kw, ev_b_kw):
        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, now_hour, 0, tzinfo=_UTC)
        result = cap._maybe_send_conflict_alert(
            [self._make_event(**ev_a_kw), self._make_event(**ev_b_kw)], now)
        assert result == 0

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_conflict_alert_uid_order_independent(self, mock_mcs):
        """Dedup key is canonicalized — UID order doesn't matter."""
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 12, 0, tzinfo=_UTC)
        ev_a = self._make_event(uid='z-uid',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))
        ev_b = self._make_event(uid='a-uid',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC))
        cap._maybe_send_conflict_alert([ev_a, ev_b], now)
        # Dedup key should use sorted UIDs: a-uid:z-uid
        flag_key = store.setex.call_args[0][0]
        assert "a-uid:z-uid" in flag_key

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_conflict_alert_includes_reschedule_hint(self, mock_mcs):
        """Conflict alert prompt should contain reschedule guidance."""
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)
        ev_a = self._make_event(uid='a', summary='Design Review',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))
        ev_b = self._make_event(uid='b', summary='Sprint Planning',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 30, tzinfo=_UTC))
        cap._maybe_send_conflict_alert([ev_a, ev_b], now)

        import json
        payload = json.loads(store.rpush.call_args[0][1])
        prompt = payload['prompt']
        assert "UID: a" in prompt
        assert "UID: b" in prompt
        assert "caldav_find_free_slots" in prompt
        assert "caldav_update_event" in prompt

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_conflict_alert_picks_shorter_event(self, mock_mcs):
        """Reschedule hint should suggest moving the shorter event."""
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        cap = _make_capability()
        now = datetime.datetime(2026, 3, 28, 8, 0, tzinfo=_UTC)
        ev_a = self._make_event(uid='short', summary='Quick Sync',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC))
        ev_b = self._make_event(uid='long', summary='Workshop',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 16, 0, tzinfo=_UTC))
        cap._maybe_send_conflict_alert([ev_a, ev_b], now)

        import json
        payload = json.loads(store.rpush.call_args[0][1])
        assert 'Quick Sync' in payload['prompt']
        assert 'Easiest to move: "Quick Sync"' in payload['prompt']


class TestBuildRescheduleHint:
    """Tests for :func:`_build_reschedule_hint`."""

    @staticmethod
    def _make_event(**overrides):
        base = {
            'uid': 'evt-1', 'summary': 'Meeting',
            'dtstart': datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
            'dtend': datetime.datetime(2026, 3, 28, 15, 0, tzinfo=_UTC),
            'all_day': False,
        }
        base.update(overrides)
        return base

    @pytest.mark.unit
    def test_hint_identifies_shorter_event(self):
        from capabilities.caldav_capability.capability import _build_reschedule_hint
        ev_a = self._make_event(uid='a', summary='Short',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 14, 30, tzinfo=_UTC))
        ev_b = self._make_event(uid='b', summary='Long',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 16, 0, tzinfo=_UTC))
        hint = _build_reschedule_hint(ev_a, ev_b)
        assert '"Short"' in hint
        assert 'UID: a' in hint
        assert '30min' in hint

    @pytest.mark.unit
    def test_hint_includes_tool_names(self):
        from capabilities.caldav_capability.capability import _build_reschedule_hint
        ev_a = self._make_event(uid='x')
        ev_b = self._make_event(uid='y',
                                dtstart=datetime.datetime(2026, 3, 28, 14, 0, tzinfo=_UTC),
                                dtend=datetime.datetime(2026, 3, 28, 16, 0, tzinfo=_UTC))
        hint = _build_reschedule_hint(ev_a, ev_b)
        assert 'caldav_find_free_slots' in hint
        assert 'caldav_update_event' in hint

    @pytest.mark.unit
    def test_hint_equal_duration_picks_first(self):
        from capabilities.caldav_capability.capability import _build_reschedule_hint
        ev_a = self._make_event(uid='a', summary='Alpha')
        ev_b = self._make_event(uid='b', summary='Beta')
        hint = _build_reschedule_hint(ev_a, ev_b)
        assert '"Alpha"' in hint  # ev_a picked when durations equal

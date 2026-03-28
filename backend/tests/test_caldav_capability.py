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
    8.  ``get_tools()`` returns exactly 5 dicts each with ``name`` and ``handler`` keys.
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
    # Test 8 — get_tools returns 5 tools
    # ------------------------------------------------------------------

    def test_get_tools_returns_five_tools(self):
        """``get_tools()`` must return a list of exactly 5 tool definition dicts.

        Each dict must have at minimum the keys ``name`` (str) and
        ``handler`` (callable), confirming that the tool definitions are
        ready for registration via ``register_tool()``.
        """
        cap = _make_capability()
        tools = cap.get_tools()

        assert isinstance(tools, list), "get_tools() must return a list"
        assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}"
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

        Iterates all 5 handlers returned by ``get_tools()`` and calls each
        with minimal arguments (``topic=""``, ``params={}``).  The
        capability is in its default disconnected state.  Asserts that every
        result is a dict containing the ``"error"`` key.
        """
        cap = _make_capability()
        assert cap.is_connected() is False

        tools = cap.get_tools()
        assert len(tools) == 5, "Prerequisite: get_tools() must return 5 tools"

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

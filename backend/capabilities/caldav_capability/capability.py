"""
CaldavCapability — concrete CalDAV calendar integration.

This module provides :class:`CaldavCapability`, a concrete implementation of
:class:`~capabilities.base.AbstractCapability` that connects to CalDAV-compatible
calendar providers (Google Calendar, Apple iCloud, Fastmail, Nextcloud, Synology,
Radicale) and exposes calendar data as knowledge facts and action tools.

Graceful degradation
--------------------
The ``caldav`` third-party library is imported inside a ``try/except`` block so
that the module can be imported even when the package is not installed.  Any
method that actually *uses* the library will raise :exc:`RuntimeError` in that
case.

Credential storage
------------------
Credentials are persisted under ``tool_name='caldav'`` in the ``tool_configs``
table.  Config keys follow the ``caldav:{field}`` convention:

- ``caldav:provider``  — provider identifier, e.g. ``"google"``
- ``caldav:username``  — account username / e-mail address
- ``caldav:password``  — app password (encrypted at rest)
"""

from __future__ import annotations

import datetime as _dt_module
import logging
import os
import pathlib
from datetime import timedelta
from typing import Any

import yaml

from capabilities.base import AbstractCapability
from capabilities.caldav_capability.providers import resolve_provider
from services.time_utils import utc_now, parse_utc  # noqa: F401 — available for subclasses

# ---------------------------------------------------------------------------
# Optional caldav import — graceful degradation when package is absent
# ---------------------------------------------------------------------------

try:
    import caldav as _caldav_lib  # type: ignore
    _CALDAV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _caldav_lib = None  # type: ignore
    _CALDAV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional icalendar import — graceful degradation when package is absent
# ---------------------------------------------------------------------------

try:
    import icalendar as _icalendar_lib  # type: ignore
    _ICALENDAR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _icalendar_lib = None  # type: ignore
    _ICALENDAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

# Credential config keys
_KEY_PROVIDER = "caldav:provider"
_KEY_USERNAME = "caldav:username"
_KEY_PASSWORD = "caldav:password"

# CalDAV connection timeout in seconds
_CONNECT_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_int(value: object, default: int) -> int:
    """Safely coerce *value* to an integer, returning *default* on failure.

    Args:
        value:   Value to coerce; typically a string loaded from config.
        default: Fallback integer returned when *value* is ``None``,
                 non-numeric, or raises during conversion.

    Returns:
        int: Parsed integer, or *default*.
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _make_date_utc(d: object) -> "_dt_module.datetime":
    """Normalise a ``datetime.date`` or ``datetime.datetime`` to UTC.

    iCalendar ``DTSTART``/``DTEND`` properties can carry either a
    :class:`datetime.date` (all-day events) or a :class:`datetime.datetime`
    (timed events).  This helper converts both to a timezone-aware UTC
    :class:`datetime.datetime` so that :func:`~services.time_utils.parse_utc`
    can be applied uniformly.

    A ``date``-only value is treated as midnight UTC (``00:00:00+00:00``).

    Args:
        d: A :class:`datetime.date`, :class:`datetime.datetime`, or any
           value accepted by :func:`~services.time_utils.parse_utc`.

    Returns:
        datetime.datetime: UTC-aware datetime.  ``date``-only values
        become midnight UTC.
    """
    if isinstance(d, _dt_module.datetime):
        return parse_utc(d)
    if isinstance(d, _dt_module.date):
        return parse_utc(_dt_module.datetime(d.year, d.month, d.day, 0, 0, 0))
    return parse_utc(d)


def _humanize_rrule(rrule_str: str, dtstart: "_dt_module.datetime", summary: str) -> str:
    """Build a concise human-readable recurrence pattern description.

    Parses a raw RRULE string (e.g. ``"FREQ=WEEKLY;BYDAY=MO,WE,FR"``) into
    natural language suitable for a knowledge fact value.

    Args:
        rrule_str: Raw RRULE value decoded from the iCalendar component,
                   e.g. ``"FREQ=WEEKLY;BYDAY=MO,WE,FR"``.
        dtstart:   UTC datetime of the first occurrence, used for the
                   human-readable time portion.
        summary:   Event title to embed in the description.

    Returns:
        str: Human-readable description, e.g.
             ``"Weekly: Team Standup (Mon/Wed/Fri 9:00 AM)"``.
             Falls back to ``"Recurring: {summary}"`` on parse error.
    """
    _FREQ_MAP = {
        'DAILY': 'Daily',
        'WEEKLY': 'Weekly',
        'MONTHLY': 'Monthly',
        'YEARLY': 'Yearly',
    }
    _DAY_MAP = {
        'MO': 'Mon', 'TU': 'Tue', 'WE': 'Wed', 'TH': 'Thu',
        'FR': 'Fri', 'SA': 'Sat', 'SU': 'Sun',
    }
    try:
        parts: dict = {}
        for token in rrule_str.split(';'):
            if '=' in token:
                k, v = token.split('=', 1)
                parts[k.strip().upper()] = v.strip()

        freq = _FREQ_MAP.get(parts.get('FREQ', ''), parts.get('FREQ', 'Recurring'))
        # strftime gives zero-padded hours; lstrip removes leading zero
        time_str = dtstart.strftime('%I:%M %p').lstrip('0') or '12:00 AM'

        if 'BYDAY' in parts:
            days = '/'.join(
                _DAY_MAP.get(d.strip()[-2:].upper(), d.strip())
                for d in parts['BYDAY'].split(',')
            )
            return f"{freq}: {summary} ({days} {time_str})"
        return f"{freq}: {summary} ({time_str})"
    except Exception:  # noqa: BLE001
        return f"Recurring: {summary}"


class CaldavCapability(AbstractCapability):
    """CalDAV calendar capability.

    Implements the :class:`~capabilities.base.AbstractCapability` interface for
    CalDAV-compatible calendar providers.  The 5 structural methods
    (:meth:`get_id`, :meth:`get_manifest`, :meth:`configure`, :meth:`connect`,
    :meth:`disconnect`) are fully implemented here.

    :meth:`ingest` and :meth:`get_tools` are concrete stubs that raise
    :exc:`NotImplementedError` — they will be implemented in subsequent tasks.

    Attributes:
        _connected (bool): Inherited from :class:`~capabilities.base.AbstractCapability`.
            ``True`` when a successful connection has been established.
    """

    def __init__(self) -> None:
        """Initialise the capability, setting connection state to ``False``."""
        super().__init__()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_id(self) -> str:
        """Return the unique capability identifier.

        Returns:
            str: Always ``"caldav"``.
        """
        return "caldav"

    def get_manifest(self) -> dict:
        """Load and return the parsed ``manifest.yaml`` for this capability.

        The manifest is read from disk on every call so that changes take effect
        without requiring a process restart.

        Returns:
            dict: Parsed YAML contents of ``manifest.yaml``, including at
            minimum ``id``, ``name``, ``version``, and ``entry_class``.

        Raises:
            FileNotFoundError: If ``manifest.yaml`` does not exist at the
                expected path alongside this module.
            yaml.YAMLError: If the manifest is not valid YAML.
        """
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Credential management & connection lifecycle
    # ------------------------------------------------------------------

    def configure(self, credentials: dict) -> None:
        """Validate, store, and test CalDAV credentials.

        Expects *credentials* to contain all three of ``provider``,
        ``username``, and ``password``.  The password is encrypted at rest via
        :meth:`~capabilities.base.AbstractCapability.store_credential`.  After
        storing, a live connection test is performed by calling
        :meth:`connect`; if the test fails a :exc:`ValueError` is raised and
        the stored credentials are removed.

        Args:
            credentials: Mapping with the following required keys:

                - ``provider`` (str): A supported provider name, e.g.
                  ``"google"``, ``"apple"``, ``"fastmail"``, ``"nextcloud"``,
                  ``"synology"``, or ``"radicale"``.
                - ``username`` (str): Account username or e-mail address.
                - ``password`` (str): App-specific or account password.

        Raises:
            ValueError: If any required key is missing, the provider is not
                recognised, or the connection test against the remote server
                fails.
        """
        # --- Validate required keys ---
        missing = [k for k in ("provider", "username", "password") if k not in credentials]
        if missing:
            raise ValueError(
                f"[caldav] configure() missing required credential fields: {missing}"
            )

        provider_name: str = credentials["provider"]
        username: str = credentials["username"]
        password: str = credentials["password"]

        # --- Validate provider is known before storing anything ---
        if resolve_provider(provider_name) is None:
            raise ValueError(
                f"[caldav] Unknown provider '{provider_name}'.  "
                f"Supported providers: google, apple, fastmail, nextcloud, synology, radicale."
            )

        # --- Persist credentials (password encrypted at rest) ---
        self.store_credential(_KEY_PROVIDER, provider_name)
        self.store_credential(_KEY_USERNAME, username)
        self.store_credential(_KEY_PASSWORD, password)

        # --- Test connectivity; roll back on failure ---
        if not self.connect():
            self.delete_credentials()
            raise ValueError(
                f"[caldav] Could not connect to provider '{provider_name}' "
                f"for user '{username}'.  Check credentials and try again."
            )

    def connect(self) -> bool:
        """Establish a connection to the CalDAV server.

        Loads stored credentials, resolves the provider's server URL, and
        performs a ``PROPFIND`` request (via :meth:`caldav.DAVClient.principal`)
        to verify that the credentials are valid.  The test uses a
        ``{_CONNECT_TIMEOUT}``-second timeout.

        Returns:
            bool: ``True`` if the connection was established successfully;
            ``False`` on any connection or authentication failure (the error is
            logged but not re-raised).
        """
        if not _CALDAV_AVAILABLE:
            logger.error(
                "[caldav] connect() called but 'caldav' package is not installed."
            )
            return False

        # --- Load stored credentials ---
        provider_name = self.load_credential(_KEY_PROVIDER)
        username = self.load_credential(_KEY_USERNAME)
        password = self.load_credential(_KEY_PASSWORD)

        if not provider_name or not username or not password:
            logger.warning(
                "[caldav] connect() aborted: one or more credentials are missing "
                "(provider=%r, username=%r, password=%s).",
                provider_name,
                username,
                "<set>" if password else "<missing>",
            )
            return False

        # --- Resolve provider config ---
        provider_config = resolve_provider(provider_name)
        if provider_config is None:
            logger.error(
                "[caldav] connect() failed: unknown provider '%s'.", provider_name
            )
            return False

        url: str = provider_config["url"]

        # --- Attempt connection ---
        try:
            client = _caldav_lib.DAVClient(
                url=url,
                username=username,
                password=password,
                timeout=_CONNECT_TIMEOUT,
            )
            client.principal()  # Performs a PROPFIND — raises on auth/network failure
            self._connected = True
            logger.info(
                "[caldav] Connected successfully (provider=%s, username=%s).",
                provider_name,
                username,
            )
            return True
        except Exception as exc:
            logger.error(
                "[caldav] connect() failed for provider=%s, username=%s: %s",
                provider_name,
                username,
                exc,
            )
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Tear down the active connection and delete stored credentials.

        Sets ``self._connected`` to ``False`` and removes all stored
        credentials from ``tool_configs``.  Does not raise.

        Returns:
            None
        """
        self._connected = False
        self.delete_credentials()
        logger.info("[caldav] Disconnected and credentials removed.")

    # ------------------------------------------------------------------
    # Stubs — implemented in subsequent tasks
    # ------------------------------------------------------------------

    def ingest(self) -> list:
        """Fetch calendar events from the CalDAV server and persist them as knowledge facts.

        Implements the full ingestion pipeline:

        1. Returns ``[]`` immediately if :meth:`is_connected` is ``False``.
        2. Loads configurable date range from ``tool_configs`` (keys
           ``caldav:range_past_days`` and ``caldav:range_future_days``),
           defaulting to 30 days in each direction.
        3. Opens a fresh :class:`caldav.DAVClient` connection and lists all
           calendars via the authenticated principal.
        4. For each calendar, fetches events within the configured date range
           via :meth:`caldav.Calendar.date_search`.  Per-calendar exceptions
           are caught and logged so that one failing calendar does not abort
           the rest.
        5. Parses each ``VEVENT`` component into a structured dict with 9
           normalised fields.
        6. Stores each event via :meth:`~services.knowledge_service.KnowledgeService.store`
           with ``kind='fact'``, ``entity='calendar'``, ``key='event:{uid}'``.
           Past events (``dtstart < now``) use ``decay_class='fast'``; future
           events use ``decay_class='standard'``.
        7. Stores one summary "recurrence" fact per unique recurring event UID
           with ``key='recurrence:{uid}'`` and ``decay_class='slow'``.
        8. Detects pairs of future overlapping events and stores a conflict fact
           for each pair with ``key='conflict:{uid1}:{uid2}'`` and
           ``decay_class='fast'``.
        9. Builds a ≤280-character schedule hint for the next 24 hours and
           injects it via
           :meth:`~services.world_state_service.WorldStateService.notify_external_signal`
           with ``signal_type='capability:caldav'``, ``source='caldav'``, and
           ``activation_energy=0.7``.

        Returns:
            list[dict]: Parsed event dicts, each containing keys ``uid``,
            ``summary``, ``dtstart``, ``dtend``, ``location``, ``attendees``,
            ``recurrence``, ``all_day``, and ``calendar_name``.  Returns ``[]``
            on connection failure, missing dependencies, or credential errors.
        """
        if not self.is_connected():
            return []

        if not _CALDAV_AVAILABLE:
            logger.error("[caldav] ingest() called but 'caldav' package is not installed.")
            return []

        if not _ICALENDAR_AVAILABLE:
            logger.error("[caldav] ingest() called but 'icalendar' package is not installed.")
            return []

        # --- Load configurable date range from tool_configs ---
        past_days = _safe_int(self._load_config_raw("caldav:range_past_days"), 30)
        future_days = _safe_int(self._load_config_raw("caldav:range_future_days"), 30)

        now = utc_now()
        range_start = now - timedelta(days=past_days)
        range_end = now + timedelta(days=future_days)

        # --- Load stored credentials ---
        provider_name = self.load_credential(_KEY_PROVIDER)
        username = self.load_credential(_KEY_USERNAME)
        password = self.load_credential(_KEY_PASSWORD)

        if not all([provider_name, username, password]):
            logger.warning("[caldav] ingest(): one or more credentials are missing.")
            return []

        provider_config = resolve_provider(provider_name)
        if provider_config is None:
            logger.error("[caldav] ingest(): unknown provider '%s'.", provider_name)
            return []

        url: str = provider_config["url"]

        # --- Open connection and enumerate calendars ---
        try:
            client = _caldav_lib.DAVClient(
                url=url,
                username=username,
                password=password,
                timeout=_CONNECT_TIMEOUT,
            )
            principal = client.principal()
            calendars = principal.calendars()
        except Exception as exc:
            logger.error("[caldav] ingest() failed to list calendars: %s", exc)
            return []

        # --- Fetch and parse events per calendar ---
        all_events: list[dict] = []

        for calendar in calendars:
            cal_name = "Unknown"
            try:
                cal_name = getattr(calendar, "name", None) or "Unknown"
                raw_events = calendar.date_search(
                    start=range_start, end=range_end, expand=True
                )
                for raw_event in raw_events:
                    try:
                        parsed = self._parse_caldav_event(raw_event, cal_name)
                        all_events.extend(parsed)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[caldav] Failed to parse event in '%s': %s", cal_name, exc
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[caldav] Failed to fetch from calendar '%s': %s", cal_name, exc
                )

        # --- Persist as knowledge facts ---
        self._store_events_as_facts(all_events, now)
        self._store_recurrence_facts(all_events)
        self._store_conflict_facts(all_events, now)

        # --- Inject world-state schedule hint ---
        self._inject_schedule_hint(all_events, now)

        logger.info(
            "[caldav] ingest() complete — %d events fetched.", len(all_events)
        )
        return all_events

    # ------------------------------------------------------------------
    # Private helpers for ingest()
    # ------------------------------------------------------------------

    def _load_config_raw(self, key: str, default: str = None) -> "str | None":
        """Load an unencrypted config value from ``tool_configs``.

        Unlike :meth:`~capabilities.base.AbstractCapability.load_credential`,
        this method reads the raw stored value without Fernet decryption.  It
        is used for non-secret configuration such as date-range settings.

        Args:
            key:     Config key, e.g. ``"caldav:range_past_days"``.
            default: Value to return when the key is absent or on error.

        Returns:
            str | None: Raw stored value, or *default* if absent or on error.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.tool_config_service import ToolConfigService

            svc = ToolConfigService(get_shared_db_service())
            config = svc.get_tool_config(self.get_id())
            return config.get(key, default)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[caldav] _load_config_raw(%r) failed: %s", key, exc)
            return default

    def _parse_caldav_event(self, raw_event: object, cal_name: str) -> list:
        """Parse a CalDAV event resource into one or more structured dicts.

        Walks the ``VCALENDAR`` component tree and converts every ``VEVENT``
        sub-component to a normalised dict with 9 fields.

        Args:
            raw_event: A ``caldav.CalendarObjectResource`` returned by
                       :meth:`caldav.Calendar.date_search`.
            cal_name:  Human-readable name of the owning calendar, used to
                       populate the ``calendar_name`` field.

        Returns:
            list[dict]: Zero or more event dicts.  Returns an empty list when
            the resource has no ``VEVENT`` components or if parsing fails
            entirely.
        """
        results: list[dict] = []

        # Obtain the VCALENDAR component — try both API styles
        try:
            ical_instance = raw_event.icalendar_instance
        except AttributeError:
            try:
                component = raw_event.icalendar_component
                ical_instance = _icalendar_lib.Calendar()
                ical_instance.add_component(component)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[caldav] Could not obtain ical instance: %s", exc)
                return results

        for component in ical_instance.walk():
            if component.name != 'VEVENT':
                continue

            dtstart_prop = component.get('DTSTART')
            if dtstart_prop is None:
                continue

            dt_raw = dtstart_prop.dt
            # all_day = date-only (not a datetime subclass)
            all_day: bool = (
                isinstance(dt_raw, _dt_module.date)
                and not isinstance(dt_raw, _dt_module.datetime)
            )
            dtstart = _make_date_utc(dt_raw)

            # Resolve dtend from DTEND or DURATION
            dtend_prop = component.get('DTEND')
            if dtend_prop is not None:
                dtend = _make_date_utc(dtend_prop.dt)
            else:
                dur_prop = component.get('DURATION')
                if dur_prop is not None:
                    try:
                        dtend = dtstart + dur_prop.dt
                    except Exception:  # noqa: BLE001
                        dtend = dtstart
                else:
                    dtend = dtstart

            # Core fields
            uid_prop = component.get('UID')
            uid: str = str(uid_prop) if uid_prop is not None else ""

            summary_prop = component.get('SUMMARY')
            summary: str = str(summary_prop) if summary_prop is not None else "No title"

            # Optional fields
            location_prop = component.get('LOCATION')
            location: "str | None" = str(location_prop) if location_prop is not None else None

            attendees_raw = component.get('ATTENDEE')
            if attendees_raw is None:
                attendees: list = []
            elif isinstance(attendees_raw, list):
                attendees = [str(a) for a in attendees_raw]
            else:
                attendees = [str(attendees_raw)]

            rrule_prop = component.get('RRULE')
            if rrule_prop is not None:
                try:
                    recurrence: "str | None" = rrule_prop.to_ical().decode('utf-8')
                except Exception:  # noqa: BLE001
                    recurrence = None
            else:
                recurrence = None

            results.append({
                'uid': uid,
                'summary': summary,
                'dtstart': dtstart,
                'dtend': dtend,
                'location': location,
                'attendees': attendees,
                'recurrence': recurrence,
                'all_day': all_day,
                'calendar_name': cal_name,
            })

        return results

    def _store_events_as_facts(self, events: list, now: "_dt_module.datetime") -> None:
        """Persist each event dict as a KnowledgeService fact.

        Events whose ``dtstart`` precedes *now* are stored with
        ``decay_class='fast'`` (past events lose relevance quickly); upcoming
        events use ``decay_class='standard'``.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime; used to classify past vs future.

        Returns:
            None
        """
        if not events:
            return
        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService

            ks = KnowledgeService(get_shared_db_service())
            for event in events:
                uid = event.get('uid', '')
                if not uid:
                    continue
                decay = 'fast' if event['dtstart'] < now else 'standard'
                ks.store(
                    kind='fact',
                    entity='calendar',
                    key=f'event:{uid}',
                    value=event.get('summary', 'No title'),
                    data=event,
                    decay_class=decay,
                    source='caldav',
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _store_events_as_facts() failed: %s", exc, exc_info=True
            )

    def _store_recurrence_facts(self, events: list) -> None:
        """Store a summary pattern fact for each unique recurring event.

        For events that carry a non-null ``recurrence`` field (raw RRULE value),
        one summary fact is written per unique UID so that downstream consumers
        can describe recurring patterns without iterating every occurrence.

        Fact schema: ``key='recurrence:{uid}'``, ``kind='fact'``,
        ``entity='calendar'``, ``decay_class='slow'``.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.

        Returns:
            None
        """
        if not events:
            return
        seen_uids: set = set()
        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService

            ks = KnowledgeService(get_shared_db_service())
            for event in events:
                uid = event.get('uid', '')
                recurrence = event.get('recurrence')
                if not uid or not recurrence or uid in seen_uids:
                    continue
                seen_uids.add(uid)
                human_pattern = _humanize_rrule(
                    recurrence,
                    event['dtstart'],
                    event.get('summary', 'Event'),
                )
                ks.store(
                    kind='fact',
                    entity='calendar',
                    key=f'recurrence:{uid}',
                    value=human_pattern,
                    data={
                        'uid': uid,
                        'rrule': recurrence,
                        'summary': event.get('summary'),
                    },
                    decay_class='slow',
                    source='caldav',
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _store_recurrence_facts() failed: %s", exc, exc_info=True
            )

    def _store_conflict_facts(self, events: list, now: "_dt_module.datetime") -> None:
        """Detect overlapping future events and store a conflict fact per pair.

        Two events *A* and *B* are considered overlapping when either:
        - ``A.dtstart <= B.dtstart < A.dtend``, or
        - ``B.dtstart <= A.dtstart < B.dtend``

        Only *upcoming* events (``dtstart >= now``) are evaluated to avoid
        noise from historical data.

        Conflict facts: ``key='conflict:{uid_a}:{uid_b}'``, ``kind='fact'``,
        ``entity='calendar'``, ``decay_class='fast'``.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime; events before this are skipped.

        Returns:
            None
        """
        upcoming = [e for e in events if e.get('dtstart') and e['dtstart'] >= now]
        if len(upcoming) < 2:
            return

        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService

            ks = KnowledgeService(get_shared_db_service())

            def _fmt_event(e: dict) -> str:
                """Format event title + time range for conflict descriptions.

                Args:
                    e: Parsed event dict.

                Returns:
                    str: E.g. ``"Meeting A (2:00 PM-3:00 PM)"``.
                """
                fmt = '%I:%M %p'
                start_s = e['dtstart'].strftime(fmt).lstrip('0') or '12:00 AM'
                end_s = e['dtend'].strftime(fmt).lstrip('0') or '12:00 AM'
                return f"{e.get('summary', 'Event')} ({start_s}-{end_s})"

            for i, ev_a in enumerate(upcoming):
                for ev_b in upcoming[i + 1:]:
                    a_start = ev_a['dtstart']
                    a_end = ev_a['dtend']
                    b_start = ev_b['dtstart']
                    b_end = ev_b['dtend']

                    overlap = (a_start <= b_start < a_end) or (b_start <= a_start < b_end)
                    if not overlap:
                        continue

                    uid_a = ev_a.get('uid', '')
                    uid_b = ev_b.get('uid', '')
                    if not uid_a or not uid_b:
                        continue

                    value = f"Overlap: {_fmt_event(ev_a)} \u2229 {_fmt_event(ev_b)}"
                    ks.store(
                        kind='fact',
                        entity='calendar',
                        key=f'conflict:{uid_a}:{uid_b}',
                        value=value,
                        data={'uid_a': uid_a, 'uid_b': uid_b},
                        decay_class='fast',
                        source='caldav',
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _store_conflict_facts() failed: %s", exc, exc_info=True
            )

    def _inject_schedule_hint(self, events: list, now: "_dt_module.datetime") -> None:
        """Build a schedule hint and inject it into world state.

        Collects events starting within the next 24 hours, formats them as a
        comma-separated time/title list (≤280 characters total), and pushes the
        result to :class:`~services.world_state_service.WorldStateService` via
        :meth:`~services.world_state_service.WorldStateService.notify_external_signal`.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime used as the hint window start.

        Returns:
            None
        """
        try:
            horizon = now + timedelta(hours=24)
            upcoming = sorted(
                [e for e in events if now <= e.get('dtstart', now) < horizon],
                key=lambda e: e['dtstart'],
            )
            if upcoming:
                parts = []
                for e in upcoming:
                    time_str = e['dtstart'].strftime('%H:%M')
                    parts.append(f"{time_str} {e.get('summary', 'Event')}")
                hint = "Next 24h: " + "; ".join(parts)
                if len(hint) > 280:
                    hint = hint[:277] + "..."
            else:
                hint = "No upcoming events in the next 24 hours."

            from services.world_state_service import WorldStateService

            WorldStateService().notify_external_signal(
                signal_type='capability:caldav',
                source='caldav',
                content=hint,
                activation_energy=0.7,
            )
            logger.info("[caldav] Injected schedule hint (%d chars).", len(hint))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _inject_schedule_hint() failed: %s", exc, exc_info=True
            )

    def get_tools(self) -> list:
        """Return CalDAV tool definitions for dynamic registration.

        .. note::
            Not yet implemented.  Will be implemented in a subsequent task.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "[caldav] get_tools() is not yet implemented."
        )

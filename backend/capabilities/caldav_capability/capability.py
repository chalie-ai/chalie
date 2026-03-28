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
import pathlib
from datetime import timedelta

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
_KEY_SERVER_URL = "caldav:server_url"

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
        # If timezone-aware but not UTC, convert to UTC explicitly
        if d.tzinfo is not None:
            return d.astimezone(_dt_module.timezone.utc)
        return parse_utc(d)
    if isinstance(d, _dt_module.date):
        return parse_utc(_dt_module.datetime(d.year, d.month, d.day, 0, 0, 0))
    return parse_utc(d)


def _events_overlap(a: dict, b: dict) -> bool:
    """Return True if two event time ranges overlap."""
    return max(a['dtstart'], b['dtstart']) < min(a['dtend'], b['dtend'])


def _find_overlap_pairs(events: list, now: "_dt_module.datetime") -> list:
    """Return overlap pairs as ``(ev_a, ev_b, canon_key)`` tuples.

    Filters to upcoming, non-all-day events with valid UIDs and detects
    time overlaps.  The ``canon_key`` is ``'{uid_a}:{uid_b}'`` with UIDs
    sorted for order-independent deduplication.
    """
    from itertools import combinations
    upcoming = [
        e for e in events
        if e.get('dtstart') and e.get('uid') and e['dtstart'] >= now and not e.get('all_day')
    ]
    return [
        (a, b, ":".join(sorted([a['uid'], b['uid']])))
        for a, b in combinations(upcoming, 2)
        if a['uid'] != b['uid'] and _events_overlap(a, b)
    ]


def _format_event_line(event: dict) -> str:
    """Format a single event dict as a compact one-line summary.

    Args:
        event: Parsed event dict from :meth:`CaldavCapability._parse_caldav_event`.

    Returns:
        str: Single-line summary.
    """
    start_str = event['dtstart'].strftime('%H:%M')
    end = event.get('dtend')
    time_part = "all-day" if event.get('all_day') else (
        f"{start_str}\u2013{end.strftime('%H:%M')}" if end else start_str
    )
    cal = event.get('calendar_name')
    loc = event.get('location')
    n = len(event.get('attendees') or [])
    return " ".join(filter(None, [
        time_part,
        event.get('summary', 'Event'),
        f"[{cal}]" if cal else "",
        f"@ {loc}" if loc else "",
        f"({n} attendees)" if n else "",
    ]))


def _format_schedule_hint(upcoming: list, max_events: int = 5) -> str:
    """Build a structured schedule hint from upcoming events.

    Args:
        upcoming: Sorted list of parsed event dicts (soonest first).
        max_events: Maximum number of events to include.

    Returns:
        str: Multi-line hint suitable for LLM context injection.
    """
    if not upcoming:
        return "No upcoming events in the next 24 hours."

    shown = upcoming[:max_events]
    lines = [_format_event_line(e) for e in shown]
    total = len(upcoming)
    remaining = total - len(shown)
    if remaining > 0:
        lines.append(f"(+{remaining} more)")
    return f"Next 24h ({total} events):\n" + "\n".join(lines)


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
        interval = parts.get('INTERVAL', '1')
        if interval not in ('', '1'):
            freq = f"Every {interval} {freq.lower()}s" if freq != 'Recurring' else freq
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
        self._manifest_cache: dict | None = None

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
        """Return the parsed ``manifest.yaml`` for this capability (cached).

        Returns:
            dict: Parsed YAML contents of ``manifest.yaml``, including at
            minimum ``id``, ``name``, ``version``, and ``entry_class``.

        Raises:
            FileNotFoundError: If ``manifest.yaml`` does not exist at the
                expected path alongside this module.
            yaml.YAMLError: If the manifest is not valid YAML.
        """
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

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
        server_url: str = credentials.get("server_url", "")

        # --- Validate provider is known before storing anything ---
        provider_config = resolve_provider(provider_name)
        if provider_config is None:
            raise ValueError(
                f"[caldav] Unknown provider '{provider_name}'.  "
                f"Supported providers: google, apple, fastmail, nextcloud, synology, radicale."
            )

        # Self-hosted providers require a server_url
        if provider_config.get("requires_server_url") and not server_url:
            raise ValueError(
                f"[caldav] Provider '{provider_name}' requires a 'server_url' "
                f"(e.g. 'https://your-server.com')."
            )

        # --- Persist credentials (password encrypted at rest) ---
        self.store_credential(_KEY_PROVIDER, provider_name)
        self.store_credential(_KEY_USERNAME, username)
        self.store_credential(_KEY_PASSWORD, password)
        if server_url:
            self.store_credential(_KEY_SERVER_URL, server_url)

        # --- Test connectivity; roll back on failure ---
        if not self.connect():
            self.delete_credentials()
            raise ValueError(
                f"[caldav] Could not connect to provider '{provider_name}' "
                f"for user '{username}'.  Check credentials and try again."
            )

    def _resolve_caldav_url(self, provider_config: dict, username: str = "") -> str:
        """Build an absolute CalDAV URL from provider config and stored server_url.

        For hosted providers (Google, Apple, Fastmail) the URL is already absolute.
        For self-hosted providers (Nextcloud, Synology, Radicale) the provider URL
        is a relative path that must be combined with the user's server_url.

        Args:
            provider_config: Provider dict from :data:`PROVIDERS`.
            username: Username for ``{username}`` substitution in URL templates.

        Returns:
            str: Absolute URL suitable for :class:`caldav.DAVClient`.
        """
        url: str = provider_config["url"]
        if "{username}" in url:
            url = url.replace("{username}", username)
        if url.startswith("http"):
            return url
        # Self-hosted: combine with stored server_url
        server_url = self.load_credential(_KEY_SERVER_URL) or ""
        server_url = server_url.rstrip("/")
        return f"{server_url}{url}"

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

        url: str = self._resolve_caldav_url(provider_config, username)

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

        url: str = self._resolve_caldav_url(provider_config, username)

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

        # --- Emit events to EventBusService ---
        self._emit_calendar_events(all_events, now)

        # --- Inject world-state schedule hint ---
        self._inject_schedule_hint(all_events, now)

        # --- First-connect greeting (once per lifetime) ---
        self._maybe_send_greeting(all_events, now)

        # --- Upcoming-event alerts (once per event UID) ---
        self._maybe_send_upcoming_alerts(all_events, now)

        # --- Daily schedule digest (once per UTC calendar day) ---
        self._maybe_send_daily_digest(all_events, now)

        # --- Conflict alerts (once per overlapping pair) ---
        self._maybe_send_conflict_alert(all_events, now)

        logger.info(
            "[caldav] ingest() complete — %d events fetched.", len(all_events)
        )
        return all_events

    def understand(self, items: list) -> list:
        """Extract structured knowledge from ingested calendar events.

        For CalDAV, the ingest phase already returns structured data (parsed
        VEVENT fields), so understanding is primarily storing facts and
        detecting patterns — handled inline by ingest's helpers. This method
        returns the items unchanged.

        Args:
            items: Parsed event dicts from :meth:`ingest`.

        Returns:
            list[dict]: Same items, unchanged.
        """
        return items

    def monitor(self) -> None:
        """Detect calendar changes and emit signals.

        Called by the capability scheduler each cycle. Delegates to
        :meth:`ingest` which handles the full pipeline: fetch, parse,
        store facts, detect conflicts, emit events, and inject world state.
        """
        self.ingest()

    def act(self, action: str, params: dict) -> dict:
        """Perform a calendar action by delegating to the corresponding tool handler.

        Args:
            action: One of ``list_events``, ``get_event``, ``create_event``,
                ``update_event``, ``delete_event``.
            params: Action-specific parameters.

        Returns:
            dict: Result from the tool handler.
        """
        action_map = {t['name'].replace('caldav_', ''): t['handler'] for t in self.get_tools()}
        handler = action_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

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
                attendees = [str(a).removeprefix('mailto:') for a in attendees_raw]
            else:
                attendees = [str(attendees_raw).removeprefix('mailto:')]

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
                # Serialize datetime fields to ISO strings for JSON storage
                serializable_data = dict(event)
                for field in ('dtstart', 'dtend'):
                    val = serializable_data.get(field)
                    if val is not None and hasattr(val, 'isoformat'):
                        serializable_data[field] = val.isoformat()
                ks.store(
                    kind='fact',
                    entity='calendar',
                    key=f'event:{uid}',
                    value=event.get('summary', 'No title'),
                    data=serializable_data,
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
        upcoming = [e for e in events if e.get('dtstart') and e['dtstart'] >= now and not e.get('all_day')]
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
                    # Skip expanded recurrence occurrences of the same event
                    if uid_a == uid_b:
                        continue

                    # Canonicalize UID pair to avoid order-dependent duplicates
                    canon_a, canon_b = sorted([uid_a, uid_b])
                    value = f"Overlap: {_fmt_event(ev_a)} \u2229 {_fmt_event(ev_b)}"
                    ks.store(
                        kind='fact',
                        entity='calendar',
                        key=f'conflict:{canon_a}:{canon_b}',
                        value=value,
                        data={'uid_a': uid_a, 'uid_b': uid_b},
                        decay_class='fast',
                        source='caldav',
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _store_conflict_facts() failed: %s", exc, exc_info=True
            )

    def _emit_calendar_events(self, events: list, now: "_dt_module.datetime") -> None:
        """Emit calendar events to EventBusService for goal ecology and proactive systems.

        Emits a single ``calendar_ingested`` event with summary counts, plus
        individual ``calendar_conflict_detected`` events for any conflicts found.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime.
        """
        try:
            from services.event_bus_service import EventBusService

            bus = EventBusService()
            upcoming = [e for e in events if e.get('dtstart') and e['dtstart'] >= now]
            bus.emit('calendar_ingested', {
                'source': 'caldav',
                'total_events': len(events),
                'upcoming_events': len(upcoming),
                'timestamp': now.isoformat(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("[caldav] _emit_calendar_events() failed: %s", exc)

    _GREETING_FLAG = "caldav:greeting_sent"

    def _maybe_send_greeting(self, events: list, now: "_dt_module.datetime") -> bool:
        """Push a one-time greeting prompt when calendar data first arrives.

        On the very first successful ingest after the capability is configured,
        pushes a greeting prompt to the ``prompt-queue``.  The digest worker
        picks it up, synthesises a natural message via the LLM, and delivers it
        as a proactive message to the user.

        Subsequent calls are no-ops (the :attr:`_GREETING_FLAG` MemoryStore key
        persists for the process lifetime).

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime.

        Returns:
            bool: ``True`` if a greeting was enqueued, ``False`` otherwise.
        """
        try:
            from services.memory_client import MemoryClientService
            import json as _json

            store = MemoryClientService.create_connection()
            if store.get(self._GREETING_FLAG):
                return False

            hint = _format_schedule_hint(
                sorted(
                    [e for e in events
                     if e.get('dtstart') and now <= e['dtstart'] < now + timedelta(hours=24)],
                    key=lambda e: e['dtstart'],
                ),
                max_events=self._HINT_MAX_EVENTS,
            )

            store.rpush('prompt-queue', _json.dumps({
                'prompt': (
                    "[CAPABILITY GREETING: CALENDAR CONNECTED]\n"
                    "The user just connected their calendar for the first time. "
                    "Here is their upcoming schedule:\n"
                    f"{hint}\n\n"
                    "Greet them warmly, mention 1-2 highlights from their schedule. "
                    "Be brief and personal — this is the moment they see Chalie "
                    "knows their real life."
                ),
                'metadata': {
                    'type': 'proactive_drift',
                    'source': 'capability_greeting',
                    'topic': 'proactive',
                },
            }))
            store.set(self._GREETING_FLAG, "1")
            logger.info("[caldav] First-connect greeting enqueued.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("[caldav] _maybe_send_greeting() failed: %s", exc)
            return False

    # Alert window: events starting within this range trigger a proactive nudge.
    _ALERT_MINUTES_MIN = 15
    _ALERT_MINUTES_MAX = 30
    _ALERT_KEY_PREFIX = "caldav:alerted:"
    _ALERT_TTL_SECONDS = 48 * 3600  # 48 hours

    def _maybe_send_upcoming_alerts(self, events: list, now: "_dt_module.datetime") -> int:
        """Push a proactive alert for events starting within 15-30 minutes.

        Each event UID is alerted at most once (deduped via MemoryStore key
        with a 48-hour TTL).

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime.

        Returns:
            int: Number of alerts enqueued.
        """
        try:
            from services.memory_client import MemoryClientService
            import json as _json

            window_start = now + timedelta(minutes=self._ALERT_MINUTES_MIN)
            window_end = now + timedelta(minutes=self._ALERT_MINUTES_MAX)
            imminent = [
                e for e in events
                if e.get('dtstart') and window_start <= e['dtstart'] <= window_end
            ]
            if not imminent:
                return 0

            store = MemoryClientService.create_connection()
            sent = 0
            for event in imminent:
                uid = event.get('uid', '')
                flag_key = f"{self._ALERT_KEY_PREFIX}{uid}"
                if store.get(flag_key):
                    continue

                line = _format_event_line(event)
                mins = int((event['dtstart'] - now).total_seconds() // 60)
                store.rpush('prompt-queue', _json.dumps({
                    'prompt': (
                        f"[CALENDAR ALERT]\n"
                        f"{line}\n"
                        f"This event starts in {mins} minutes.\n\n"
                        f"Give the user a brief heads-up. If there's a location, "
                        f"mention it. Be concise — one or two sentences max."
                    ),
                    'metadata': {
                        'type': 'proactive_drift',
                        'source': 'calendar_alert',
                        'topic': 'proactive',
                    },
                }))
                store.setex(flag_key, self._ALERT_TTL_SECONDS, "1")
                sent += 1
                logger.info("[caldav] Upcoming-event alert for '%s' in %d min.",
                            event.get('summary', '?'), mins)
            return sent
        except Exception as exc:  # noqa: BLE001
            logger.debug("[caldav] _maybe_send_upcoming_alerts() failed: %s", exc)
            return 0

    # Daily digest: one per UTC calendar day, deduped via MemoryStore key.
    _DIGEST_KEY_PREFIX = "caldav:digest:"
    _DIGEST_TTL_SECONDS = 24 * 3600  # 24 hours
    _DIGEST_MAX_EVENTS = 10

    def _maybe_send_daily_digest(self, events: list, now: "_dt_module.datetime") -> bool:
        """Push a once-per-day full schedule digest via prompt-queue.

        Fires on the first ingest cycle of each UTC calendar day.  Subsequent
        calls on the same date are no-ops (deduped via a MemoryStore key with
        24-hour TTL).

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime.

        Returns:
            bool: ``True`` if a digest was enqueued, ``False`` otherwise.
        """
        try:
            from services.memory_client import MemoryClientService
            import json as _json

            date_key = now.strftime("%Y-%m-%d")
            flag_key = f"{self._DIGEST_KEY_PREFIX}{date_key}"

            store = MemoryClientService.create_connection()
            if store.get(flag_key):
                return False

            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(hours=24)
            todays = sorted(
                [e for e in events if e.get('dtstart') and today_start <= e['dtstart'] < today_end],
                key=lambda e: e['dtstart'],
            )

            hint = _format_schedule_hint(todays, max_events=self._DIGEST_MAX_EVENTS)

            store.rpush('prompt-queue', _json.dumps({
                'prompt': (
                    "[DAILY CALENDAR DIGEST]\n"
                    f"{hint}\n\n"
                    "Give the user a concise morning-style briefing of their day. "
                    "Mention any conflicts, back-to-back meetings, or large free blocks. "
                    "Keep it warm and brief — 3-4 sentences max."
                ),
                'metadata': {
                    'type': 'proactive_drift',
                    'source': 'calendar_digest',
                    'topic': 'proactive',
                },
            }))
            store.setex(flag_key, self._DIGEST_TTL_SECONDS, "1")
            logger.info("[caldav] Daily digest enqueued for %s.", date_key)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("[caldav] _maybe_send_daily_digest() failed: %s", exc)
            return False

    # Conflict alert: deduped per overlapping pair, 48-hour TTL.
    _CONFLICT_ALERT_KEY_PREFIX = "caldav:conflict_alerted:"
    _CONFLICT_ALERT_TTL_SECONDS = 48 * 3600  # 48 hours

    def _maybe_send_conflict_alert(self, events: list, now: "_dt_module.datetime") -> int:
        """Push a proactive alert for each pair of overlapping future events.

        Reuses the same overlap logic as :meth:`_store_conflict_facts` but
        instead of writing knowledge facts, pushes a prompt-queue message so the
        user is notified.  Each conflict pair is alerted at most once (deduped
        via a MemoryStore key with a 48-hour TTL).

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime; events before this are skipped.

        Returns:
            int: Number of conflict alerts enqueued.
        """
        pairs = _find_overlap_pairs(events, now)
        if not pairs:
            return 0

        try:
            from services.memory_client import MemoryClientService
            import json as _json

            store = MemoryClientService.create_connection()
            sent = 0

            for ev_a, ev_b, canon_key in pairs:
                flag_key = f"{self._CONFLICT_ALERT_KEY_PREFIX}{canon_key}"
                if store.get(flag_key):
                    continue

                store.rpush('prompt-queue', _json.dumps({
                    'prompt': (
                        "[CALENDAR CONFLICT]\n"
                        f"These two events overlap:\n"
                        f"  1. {_format_event_line(ev_a)}\n"
                        f"  2. {_format_event_line(ev_b)}\n\n"
                        "Alert the user about this scheduling conflict. "
                        "Be concise — mention both events and ask if they'd "
                        "like to reschedule one."
                    ),
                    'metadata': {
                        'type': 'proactive_drift',
                        'source': 'calendar_conflict',
                        'topic': 'proactive',
                    },
                }))
                store.setex(flag_key, self._CONFLICT_ALERT_TTL_SECONDS, "1")
                sent += 1
                logger.info(
                    "[caldav] Conflict alert: '%s' overlaps '%s'.",
                    ev_a.get('summary', '?'), ev_b.get('summary', '?'),
                )
            return sent
        except Exception as exc:  # noqa: BLE001
            logger.debug("[caldav] _maybe_send_conflict_alert() failed: %s", exc)
            return 0

    # Maximum number of events included in the schedule hint.
    _HINT_MAX_EVENTS = 5

    def _inject_schedule_hint(self, events: list, now: "_dt_module.datetime") -> None:
        """Build a structured schedule hint and inject it into world state.

        Selects up to :attr:`_HINT_MAX_EVENTS` events starting within the next
        24 hours and formats each as a one-line structured summary including
        calendar name, location, attendee count, and duration.  The hint is
        pushed to :class:`~services.world_state_service.WorldStateService` via
        :meth:`~services.world_state_service.WorldStateService.notify_external_signal`.

        Args:
            events: Parsed event dicts from :meth:`_parse_caldav_event`.
            now:    Current UTC datetime used as the hint window start.
        """
        try:
            horizon = now + timedelta(hours=24)
            upcoming = sorted(
                [e for e in events if now <= e.get('dtstart', now) < horizon],
                key=lambda e: e['dtstart'],
            )
            hint = _format_schedule_hint(upcoming, self._HINT_MAX_EVENTS)

            from services.world_state_service import WorldStateService

            WorldStateService().notify_external_signal(
                signal_type='capability:caldav',
                source='caldav',
                content=hint,
                activation_energy=0.7,
            )
            logger.info("[caldav] Injected schedule hint (%d events, %d chars).",
                        min(len(upcoming), self._HINT_MAX_EVENTS), len(hint))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[caldav] _inject_schedule_hint() failed: %s", exc, exc_info=True
            )

    def get_tools(self) -> list:
        """Return CalDAV tool definitions for dynamic registration.

        Returns a list of 6 tool definition dicts.  Each has ``name``,
        ``description``, ``parameters``, ``returns``, ``constraints``,
        and ``handler`` (closure bound to this instance).  Handlers
        return ``{'error': 'Not connected'}`` when disconnected.
        """
        capability = self  # capture for closures

        # ------------------------------------------------------------------
        # Helper: open a fresh DAVClient from stored credentials
        # ------------------------------------------------------------------
        def _open_client():
            """Open a :class:`caldav.DAVClient` using stored credentials.

            Returns:
                caldav.DAVClient: Authenticated client ready for use.

            Raises:
                RuntimeError: If the ``caldav`` package is unavailable.
                ValueError: If any stored credential is missing or the
                    provider is not recognised.
            """
            if not _CALDAV_AVAILABLE:
                raise RuntimeError("'caldav' package is not installed.")
            provider_name = capability.load_credential(_KEY_PROVIDER)
            username = capability.load_credential(_KEY_USERNAME)
            password = capability.load_credential(_KEY_PASSWORD)
            if not all([provider_name, username, password]):
                raise ValueError("One or more CalDAV credentials are missing.")
            provider_config = resolve_provider(provider_name)
            if provider_config is None:
                raise ValueError(f"Unknown CalDAV provider: '{provider_name}'.")
            url = capability._resolve_caldav_url(provider_config, username)
            return _caldav_lib.DAVClient(
                url=url,
                username=username,
                password=password,
                timeout=_CONNECT_TIMEOUT,
            )

        # ------------------------------------------------------------------
        # Helper: serialize an event dict (convert datetimes to ISO strings)
        # ------------------------------------------------------------------
        def _serialize_event(event: dict) -> dict:
            """Return a copy of *event* with datetime fields serialised to ISO 8601.

            Args:
                event: Parsed event dict produced by :meth:`_parse_caldav_event`.

            Returns:
                dict: Shallow copy of *event* where ``dtstart`` and ``dtend``
                values are ISO 8601 strings rather than
                :class:`datetime.datetime` objects.
            """
            out = dict(event)
            for field in ("dtstart", "dtend"):
                val = out.get(field)
                if val is not None and hasattr(val, "isoformat"):
                    out[field] = val.isoformat()
            return out

        # ------------------------------------------------------------------
        # caldav_list_events
        # ------------------------------------------------------------------
        def _list_events_execute(
            topic: str,
            params: dict,
            config: dict = None,
            telemetry: dict = None,
        ) -> dict:
            """Ingest and return a filtered list of calendar events.

            Calls :meth:`~CaldavCapability.ingest` to fetch a fresh event list
            from the CalDAV server, then applies optional filters for date
            range, calendar name, and result count.

            Args:
                topic:     Unused; present for handler-signature compatibility.
                params:    Optional filter keys:

                    - ``date_from`` (str) — ISO 8601 lower bound for ``dtstart``
                    - ``date_to``   (str) — ISO 8601 upper bound for ``dtstart``
                    - ``calendar_name`` (str) — filter to a specific calendar
                    - ``limit`` (int) — maximum number of results to return

                config:    Unused.
                telemetry: Unused.

            Returns:
                dict: ``{'events': [...], 'count': int}`` on success, or
                ``{'error': str}`` on failure.
            """
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                # Read from knowledge store (populated by scheduler-driven ingest)
                # rather than triggering a full CalDAV sync on every tool call
                try:
                    from services.database_service import get_shared_db_service
                    from services.knowledge_service import KnowledgeService
                    import json as _json

                    ks = KnowledgeService(get_shared_db_service())
                    facts = ks.search(entity='calendar', kind='fact', limit=200)
                    events = []
                    for fact in facts:
                        key = fact.get('key', '')
                        if not key.startswith('event:'):
                            continue
                        data = fact.get('data')
                        if isinstance(data, str):
                            try:
                                data = _json.loads(data)
                            except (ValueError, TypeError):
                                continue
                        if isinstance(data, dict):
                            # Re-parse datetimes for filtering
                            for field in ('dtstart', 'dtend'):
                                val = data.get(field)
                                if isinstance(val, str):
                                    try:
                                        data[field] = parse_utc(val)
                                    except Exception:  # noqa: BLE001
                                        pass
                            events.append(data)
                except Exception:  # noqa: BLE001
                    # Fallback to live ingest if knowledge store read fails
                    events = capability.ingest()

                # --- Apply optional filters ---
                date_from_raw = params.get("date_from")
                date_to_raw = params.get("date_to")
                calendar_name_filter = params.get("calendar_name")
                limit_raw = params.get("limit")

                filtered = list(events)

                if date_from_raw:
                    try:
                        dt_from = parse_utc(date_from_raw)
                        filtered = [
                            e for e in filtered
                            if e.get("dtstart") and e["dtstart"] >= dt_from
                        ]
                    except Exception:  # noqa: BLE001
                        pass  # Ignore malformed date; return unfiltered

                if date_to_raw:
                    try:
                        dt_to = parse_utc(date_to_raw)
                        filtered = [
                            e for e in filtered
                            if e.get("dtstart") and e["dtstart"] <= dt_to
                        ]
                    except Exception:  # noqa: BLE001
                        pass

                if calendar_name_filter:
                    cal_lower = calendar_name_filter.lower()
                    filtered = [
                        e for e in filtered
                        if e.get("calendar_name", "").lower() == cal_lower
                    ]

                if limit_raw is not None:
                    try:
                        filtered = filtered[: int(limit_raw)]
                    except (ValueError, TypeError):
                        pass

                serialized = [_serialize_event(e) for e in filtered]
                return {"events": serialized, "count": len(serialized)}
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_list_events handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_get_event
        # ------------------------------------------------------------------
        def _get_event_execute(
            topic: str,
            params: dict,
            config: dict = None,
            telemetry: dict = None,
        ) -> dict:
            """Fetch a single calendar event by UID.

            First checks the local knowledge store for a cached fact.  Falls
            back to a direct ``calendar.search(uid=uid)`` request against the
            CalDAV server when the knowledge store lookup returns nothing.

            Args:
                topic:     Unused.
                params:    Must contain ``uid`` (str) — the iCalendar UID of
                           the event to retrieve.
                config:    Unused.
                telemetry: Unused.

            Returns:
                dict: ``{'event': {...}, 'source': str}`` on success, or
                ``{'error': str}`` when the UID is not found or on failure.
            """
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                uid = (params.get("uid") or "").strip()
                if not uid:
                    return {"error": "Parameter 'uid' is required."}

                # --- Knowledge store fast path ---
                try:
                    from services.database_service import get_shared_db_service
                    from services.knowledge_service import KnowledgeService

                    import json as _json

                    ks = KnowledgeService(get_shared_db_service())
                    fact = ks.get(entity="calendar", key=f"event:{uid}")
                    if fact and fact.get("data"):
                        data = fact["data"]
                        # Knowledge store may return raw JSON string
                        if isinstance(data, str):
                            data = _json.loads(data)
                        return {
                            "event": _serialize_event(data),
                            "source": "knowledge_store",
                        }
                except Exception:  # noqa: BLE001
                    pass  # Fall through to direct CalDAV lookup

                # --- Direct CalDAV lookup fallback ---
                if not _CALDAV_AVAILABLE:
                    return {"error": f"Event not found (UID: {uid})"}

                client = _open_client()
                principal = client.principal()
                for calendar in principal.calendars():
                    cal_name = getattr(calendar, "name", None) or "Unknown"
                    try:
                        results = calendar.search(uid=uid)
                        if results:
                            parsed = capability._parse_caldav_event(results[0], cal_name)
                            if parsed:
                                return {
                                    "event": _serialize_event(parsed[0]),
                                    "source": "caldav",
                                }
                    except Exception:  # noqa: BLE001
                        continue

                return {"error": f"Event not found (UID: {uid})"}
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_get_event handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_create_event
        # ------------------------------------------------------------------
        def _create_event_execute(
            topic: str,
            params: dict,
            config: dict = None,
            telemetry: dict = None,
        ) -> dict:
            """Create a new VEVENT on the CalDAV server.

            Builds an iCalendar ``VCALENDAR`` object containing a single
            ``VEVENT`` component and saves it to the target calendar via
            :meth:`caldav.Calendar.save_event`.

            Args:
                topic:     Unused.
                params:    Required keys: ``summary`` (str), ``dtstart`` (ISO
                           8601 UTC str), ``dtend`` (ISO 8601 UTC str).
                           Optional keys: ``location`` (str),
                           ``description`` (str), ``calendar_name`` (str —
                           target calendar; first available used if omitted).
                config:    Unused.
                telemetry: Unused.

            Returns:
                dict: ``{'uid': str, 'summary': str, 'dtstart': str,
                'dtend': str, 'calendar_name': str}`` on success, or
                ``{'error': str}`` on failure.
            """
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                if not _CALDAV_AVAILABLE:
                    return {"error": "'caldav' package is not installed."}
                if not _ICALENDAR_AVAILABLE:
                    return {"error": "'icalendar' package is not installed."}

                summary = (params.get("summary") or "").strip()
                dtstart_raw = (params.get("dtstart") or "").strip()
                dtend_raw = (params.get("dtend") or "").strip()

                if not summary:
                    return {"error": "Parameter 'summary' is required."}
                if not dtstart_raw:
                    return {"error": "Parameter 'dtstart' is required (ISO 8601 UTC)."}
                if not dtend_raw:
                    return {"error": "Parameter 'dtend' is required (ISO 8601 UTC)."}

                dtstart = parse_utc(dtstart_raw)
                dtend = parse_utc(dtend_raw)
                location = params.get("location") or ""
                description = params.get("description") or ""
                calendar_name_pref = params.get("calendar_name") or ""

                client = _open_client()
                principal = client.principal()
                calendars = principal.calendars()

                if not calendars:
                    return {"error": "No calendars found on the CalDAV server."}

                # Resolve target calendar
                target_cal = None
                if calendar_name_pref:
                    for cal in calendars:
                        if getattr(cal, "name", "") == calendar_name_pref:
                            target_cal = cal
                            break
                if target_cal is None:
                    target_cal = calendars[0]

                # Build iCalendar payload
                import uuid as _uuid_mod

                event_uid = str(_uuid_mod.uuid4())

                ical = _icalendar_lib.Calendar()
                ical.add("prodid", "-//CaldavCapability//EN")
                ical.add("version", "2.0")

                vevent = _icalendar_lib.Event()
                vevent.add("uid", event_uid)
                vevent.add("summary", summary)
                vevent.add("dtstart", dtstart)
                vevent.add("dtend", dtend)
                vevent.add("dtstamp", utc_now())
                if location:
                    vevent.add("location", location)
                if description:
                    vevent.add("description", description)

                ical.add_component(vevent)
                target_cal.save_event(ical.to_ical().decode("utf-8"))

                cal_label = getattr(target_cal, "name", None) or "Unknown"
                logger.info(
                    "[caldav] Created event uid=%s summary=%r calendar=%r",
                    event_uid,
                    summary,
                    cal_label,
                )
                return {
                    "uid": event_uid,
                    "summary": summary,
                    "dtstart": dtstart.isoformat(),
                    "dtend": dtend.isoformat(),
                    "calendar_name": cal_label,
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_create_event handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_update_event
        # ------------------------------------------------------------------
        def _update_event_execute(
            topic: str,
            params: dict,
            config: dict = None,
            telemetry: dict = None,
        ) -> dict:
            """Update an existing CalDAV event by UID.

            Searches all connected calendars for an event matching *uid*.
            Only fields explicitly present in *params* are modified; absent
            keys leave the existing VEVENT property untouched.

            Mutable fields: ``summary``, ``dtstart``, ``dtend``, ``location``,
            ``description``.

            Args:
                topic:     Unused.
                params:    Must contain ``uid`` (str).  Optionally any of:
                           ``summary``, ``dtstart``, ``dtend``, ``location``,
                           ``description``.
                config:    Unused.
                telemetry: Unused.

            Returns:
                dict: ``{'uid': str, 'updated': True}`` on success, or
                ``{'error': str}`` if the event was not found or saving failed.
            """
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                if not _CALDAV_AVAILABLE:
                    return {"error": "'caldav' package is not installed."}
                if not _ICALENDAR_AVAILABLE:
                    return {"error": "'icalendar' package is not installed."}

                uid = (params.get("uid") or "").strip()
                if not uid:
                    return {"error": "Parameter 'uid' is required."}

                client = _open_client()
                principal = client.principal()

                found_event = None
                for calendar in principal.calendars():
                    try:
                        results = calendar.search(uid=uid)
                        if results:
                            found_event = results[0]
                            break
                    except Exception:  # noqa: BLE001
                        continue

                if found_event is None:
                    return {"error": f"Event not found (UID: {uid})"}

                # Parse the existing iCalendar data
                try:
                    ical_data = (
                        found_event.data
                        if isinstance(found_event.data, str)
                        else found_event.data.decode("utf-8")
                    )
                    ical = _icalendar_lib.Calendar.from_ical(ical_data)
                except AttributeError:
                    # Newer caldav versions expose icalendar_instance directly
                    ical = found_event.icalendar_instance

                # Mutate the target VEVENT component
                for component in ical.walk():
                    if component.name != "VEVENT":
                        continue

                    if "summary" in params and params["summary"] is not None:
                        component.pop("SUMMARY", None)
                        component.add("summary", str(params["summary"]))

                    if "dtstart" in params and params["dtstart"] is not None:
                        component.pop("DTSTART", None)
                        component.add("dtstart", parse_utc(params["dtstart"]))

                    if "dtend" in params and params["dtend"] is not None:
                        component.pop("DTEND", None)
                        component.add("dtend", parse_utc(params["dtend"]))

                    if "location" in params:
                        component.pop("LOCATION", None)
                        if params["location"]:
                            component.add("location", str(params["location"]))

                    if "description" in params:
                        component.pop("DESCRIPTION", None)
                        if params["description"]:
                            component.add("description", str(params["description"]))

                    break  # Only update the first VEVENT

                # Persist the mutated calendar object
                found_event.data = ical.to_ical().decode("utf-8")
                found_event.save()

                logger.info("[caldav] Updated event uid=%s", uid)
                return {"uid": uid, "updated": True}
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_update_event handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_delete_event
        # ------------------------------------------------------------------
        def _delete_event_execute(
            topic: str,
            params: dict,
            config: dict = None,
            telemetry: dict = None,
        ) -> dict:
            """Delete a calendar event from the CalDAV server by UID.

            Searches all connected calendars for an event with the given UID
            and calls :meth:`caldav.CalendarObjectResource.delete` when found.

            Args:
                topic:     Unused.
                params:    Must contain ``uid`` (str) — the event to delete.
                config:    Unused.
                telemetry: Unused.

            Returns:
                dict: ``{'uid': str, 'deleted': True}`` on success, or
                ``{'error': str}`` if the event was not found or deletion
                failed.
            """
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                if not _CALDAV_AVAILABLE:
                    return {"error": "'caldav' package is not installed."}

                uid = (params.get("uid") or "").strip()
                if not uid:
                    return {"error": "Parameter 'uid' is required."}

                client = _open_client()
                principal = client.principal()

                for calendar in principal.calendars():
                    try:
                        results = calendar.search(uid=uid)
                        if results:
                            results[0].delete()
                            logger.info("[caldav] Deleted event uid=%s", uid)
                            return {"uid": uid, "deleted": True}
                    except Exception:  # noqa: BLE001
                        continue

                return {"error": f"Event not found (UID: {uid})"}
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_delete_event handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_find_free_slots
        # ------------------------------------------------------------------
        def _find_free_slots_execute(
            topic: str, params: dict,
            config: dict = None, telemetry: dict = None,
        ) -> dict:
            """Find available time windows between busy blocks."""
            if not capability.is_connected():
                return {"error": "Not connected"}
            try:
                # Read events from knowledge store (same as list_events)
                try:
                    from services.database_service import get_shared_db_service
                    from services.knowledge_service import KnowledgeService
                    import json as _json
                    ks = KnowledgeService(get_shared_db_service())
                    events = []
                    for fact in ks.search(entity='calendar', kind='fact', limit=200):
                        if not fact.get('key', '').startswith('event:'):
                            continue
                        data = fact.get('data')
                        if isinstance(data, str):
                            try:
                                data = _json.loads(data)
                            except (ValueError, TypeError):
                                continue
                        if not isinstance(data, dict):
                            continue
                        for f in ('dtstart', 'dtend'):
                            v = data.get(f)
                            if isinstance(v, str):
                                try:
                                    data[f] = parse_utc(v)
                                except Exception:  # noqa: BLE001
                                    pass
                        events.append(data)
                except Exception:  # noqa: BLE001
                    events = capability.ingest()

                now = utc_now()
                dt_from = parse_utc(params["date_from"]) if params.get("date_from") else now.replace(hour=0, minute=0, second=0, microsecond=0)
                dt_to = parse_utc(params["date_to"]) if params.get("date_to") else dt_from + timedelta(days=7)
                min_td = timedelta(minutes=int(params.get("min_duration_minutes", 30)))
                wh_s = int(params.get("working_hours_start", 8))
                wh_e = int(params.get("working_hours_end", 18))

                # Collect busy intervals (skip all-day events)
                busy = sorted(
                    (e['dtstart'], e['dtend']) for e in events
                    if not e.get('all_day') and e.get('dtstart') and e.get('dtend')
                    and e['dtend'] > dt_from and e['dtstart'] < dt_to
                )

                slots = []
                cur_day = dt_from
                while cur_day < dt_to:
                    ds = cur_day.replace(hour=wh_s, minute=0, second=0, microsecond=0)
                    de = cur_day.replace(hour=wh_e, minute=0, second=0, microsecond=0)
                    if de <= now:
                        cur_day += timedelta(days=1)
                        continue
                    if ds < now:
                        ds = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

                    # Clamp busy blocks to this day's working hours
                    day_busy = sorted(
                        (max(s, ds), min(e, de)) for s, e in busy
                        if e > ds and s < de
                    )

                    cursor = ds
                    for ev_s, ev_e in day_busy:
                        gap = ev_s - cursor
                        if ev_s > cursor and gap >= min_td:
                            slots.append({"date": cur_day.strftime("%Y-%m-%d"),
                                          "start": cursor.isoformat(),
                                          "end": ev_s.isoformat(),
                                          "duration_minutes": int(gap.total_seconds() // 60)})
                        cursor = max(cursor, ev_e)

                    gap = de - cursor
                    if cursor < de and gap >= min_td:
                        slots.append({"date": cur_day.strftime("%Y-%m-%d"),
                                      "start": cursor.isoformat(),
                                      "end": de.isoformat(),
                                      "duration_minutes": int(gap.total_seconds() // 60)})
                    cur_day += timedelta(days=1)

                return {"slots": slots, "count": len(slots)}
            except Exception as exc:  # noqa: BLE001
                logger.error("[caldav] caldav_find_free_slots failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # Assemble and return tool definition list
        # ------------------------------------------------------------------
        return [
            {
                "name": "caldav_list_events",
                "description": (
                    "List calendar events from connected CalDAV calendars. "
                    "Ingests a fresh event list and returns it, optionally filtered "
                    "by date range, calendar name, or result count."
                ),
                "parameters": {
                    "date_from": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "ISO 8601 UTC lower bound for dtstart "
                            "(e.g. '2026-03-01T00:00:00Z'). Omit for no lower bound."
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "ISO 8601 UTC upper bound for dtstart "
                            "(e.g. '2026-04-01T00:00:00Z'). Omit for no upper bound."
                        ),
                    },
                    "calendar_name": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Return only events from this calendar (exact name match)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "required": False,
                        "description": "Maximum number of events to return.",
                    },
                },
                "returns": {
                    "events": {
                        "type": "array",
                        "description": (
                            "List of event dicts with uid, summary, dtstart, dtend, "
                            "location, attendees, recurrence, all_day, calendar_name."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of events returned.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if the operation failed.",
                    },
                },
                "constraints": {"timeout_seconds": 60},
                "handler": _list_events_execute,
            },
            {
                "name": "caldav_get_event",
                "description": (
                    "Fetch a single calendar event by its iCalendar UID. "
                    "Checks the local knowledge store first, then queries the "
                    "CalDAV server directly if no cached fact is found."
                ),
                "parameters": {
                    "uid": {
                        "type": "string",
                        "required": True,
                        "description": "The unique identifier (UID) of the calendar event.",
                    },
                },
                "returns": {
                    "event": {
                        "type": "object",
                        "description": (
                            "Event dict with uid, summary, dtstart (ISO 8601), "
                            "dtend (ISO 8601), location, attendees, recurrence, "
                            "all_day, calendar_name."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": "'knowledge_store' or 'caldav'.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if the event was not found.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _get_event_execute,
            },
            {
                "name": "caldav_create_event",
                "description": (
                    "Create a new calendar event on the CalDAV server. "
                    "Returns the generated UID and event details on success."
                ),
                "parameters": {
                    "summary": {
                        "type": "string",
                        "required": True,
                        "description": "Event title / summary.",
                    },
                    "dtstart": {
                        "type": "string",
                        "required": True,
                        "description": (
                            "Event start as ISO 8601 UTC datetime "
                            "(e.g. '2026-04-01T09:00:00Z')."
                        ),
                    },
                    "dtend": {
                        "type": "string",
                        "required": True,
                        "description": (
                            "Event end as ISO 8601 UTC datetime "
                            "(e.g. '2026-04-01T10:00:00Z')."
                        ),
                    },
                    "location": {
                        "type": "string",
                        "required": False,
                        "description": "Optional event location.",
                    },
                    "description": {
                        "type": "string",
                        "required": False,
                        "description": "Optional event description / notes.",
                    },
                    "calendar_name": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Target calendar name. Uses the first available "
                            "calendar when omitted."
                        ),
                    },
                },
                "returns": {
                    "uid": {
                        "type": "string",
                        "description": "Assigned UID of the newly created event.",
                    },
                    "summary": {"type": "string"},
                    "dtstart": {
                        "type": "string",
                        "description": "ISO 8601 UTC start time.",
                    },
                    "dtend": {
                        "type": "string",
                        "description": "ISO 8601 UTC end time.",
                    },
                    "calendar_name": {
                        "type": "string",
                        "description": "Calendar the event was added to.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if creation failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _create_event_execute,
            },
            {
                "name": "caldav_update_event",
                "description": (
                    "Update an existing calendar event on the CalDAV server by UID. "
                    "Only the fields supplied in params are modified; all other "
                    "VEVENT properties are preserved."
                ),
                "parameters": {
                    "uid": {
                        "type": "string",
                        "required": True,
                        "description": "The UID of the event to update.",
                    },
                    "summary": {
                        "type": "string",
                        "required": False,
                        "description": "New event title.",
                    },
                    "dtstart": {
                        "type": "string",
                        "required": False,
                        "description": "New start time as ISO 8601 UTC datetime.",
                    },
                    "dtend": {
                        "type": "string",
                        "required": False,
                        "description": "New end time as ISO 8601 UTC datetime.",
                    },
                    "location": {
                        "type": "string",
                        "required": False,
                        "description": "New location. Pass empty string to clear.",
                    },
                    "description": {
                        "type": "string",
                        "required": False,
                        "description": "New description. Pass empty string to clear.",
                    },
                },
                "returns": {
                    "uid": {"type": "string"},
                    "updated": {
                        "type": "boolean",
                        "description": "True when the event was successfully updated.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if the update failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _update_event_execute,
            },
            {
                "name": "caldav_delete_event",
                "description": (
                    "Delete a calendar event from the CalDAV server by UID. "
                    "Searches all connected calendars before deleting."
                ),
                "parameters": {
                    "uid": {
                        "type": "string",
                        "required": True,
                        "description": "The UID of the event to delete.",
                    },
                },
                "returns": {
                    "uid": {"type": "string"},
                    "deleted": {
                        "type": "boolean",
                        "description": "True when the event was found and deleted.",
                    },
                    "error": {
                        "type": "string",
                        "description": (
                            "Error message if deletion failed or the event "
                            "was not found."
                        ),
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _delete_event_execute,
            },
            {
                "name": "caldav_find_free_slots",
                "description": (
                    "Find available time slots in the calendar for scheduling. "
                    "Returns free windows within working hours. Use for "
                    "'when am I free?' queries."
                ),
                "parameters": {
                    "date_from": {"type": "string", "required": False,
                                  "description": "ISO 8601 UTC range start (default: today)."},
                    "date_to": {"type": "string", "required": False,
                                "description": "ISO 8601 UTC range end (default: +7 days)."},
                    "min_duration_minutes": {"type": "integer", "required": False,
                                             "description": "Minimum slot length (default: 30)."},
                    "working_hours_start": {"type": "integer", "required": False,
                                            "description": "Hour 0-23 (default: 8)."},
                    "working_hours_end": {"type": "integer", "required": False,
                                          "description": "Hour 0-23 (default: 18)."},
                },
                "returns": {
                    "slots": {"type": "array",
                              "description": "Free slot dicts: date, start, end, duration_minutes."},
                    "count": {"type": "integer", "description": "Number of free slots."},
                    "error": {"type": "string", "description": "Error message on failure."},
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _find_free_slots_execute,
            },
        ]

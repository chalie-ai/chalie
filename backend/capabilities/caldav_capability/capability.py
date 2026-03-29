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
from services.time_utils import utc_now, parse_utc

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


_BACK_TO_BACK_GAP = timedelta(minutes=5)


def _find_back_to_back_pairs(events: list, now: "_dt_module.datetime") -> list:
    """Return consecutive event pairs with gap < 5 minutes.

    Returns ``(ev_a, ev_b, gap_minutes, canon_key)`` tuples.
    """
    threshold = _BACK_TO_BACK_GAP.total_seconds() / 60
    upcoming = sorted(
        [e for e in events
         if (e.get('dtstart') and e.get('dtend') and e.get('uid')
             and e['dtstart'] >= now and not e.get('all_day'))],
        key=lambda e: e['dtstart'],
    )
    pairs = []
    for i in range(len(upcoming) - 1):
        a, b = upcoming[i], upcoming[i + 1]
        gap = (b['dtstart'] - a['dtend']).total_seconds() / 60
        if a['uid'] != b['uid'] and 0 <= gap < threshold:
            canon = ":".join(sorted([a['uid'], b['uid']]))
            pairs.append((a, b, round(gap), canon))
    return pairs


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


def _get_user_tz():
    """Return the user's ZoneInfo timezone, or None if unavailable.

    NOTE: Returns ``None`` (not UTC) so callers like ``_next_morning_8am``
    can distinguish "no timezone configured" from "UTC configured".
    """
    try:
        from services.database_service import get_shared_db_service
        from services.settings_service import SettingsService
        settings = SettingsService(get_shared_db_service())
        tz_name = settings.get("user_timezone")
        if tz_name:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz_name)
    except Exception:
        pass
    return None


def _next_morning_8am() -> "_dt_module.datetime":
    """Return the next 08:00 in the user's local timezone, as UTC.

    Falls back to 08:00 UTC when no timezone is available.
    """
    now = utc_now()
    tz = _get_user_tz()
    if tz:
        local_now = now.astimezone(tz)
        local_8am = local_now.replace(hour=8, minute=0, second=0, microsecond=0)
        if local_8am <= local_now:
            local_8am += timedelta(days=1)
        from datetime import timezone as _tz
        return local_8am.astimezone(_tz.utc)
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=8, minute=0, second=0, microsecond=0)


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
            self._ensure_sync_registration()
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
        """Tear down the active connection, cancel scheduled_items, and delete credentials."""
        self._connected = False
        self.delete_credentials()
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='caldav' AND status='pending'"
                )
                conn.commit()
        except Exception as exc:
            logger.warning("[caldav] disconnect cleanup: %s", exc)
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
                    except Exception as exc:
                        logger.warning(
                            "[caldav] Failed to parse event in '%s': %s", cal_name, exc
                        )
            except Exception as exc:
                logger.error(
                    "[caldav] Failed to fetch from calendar '%s': %s", cal_name, exc
                )

        # --- Index attendee identities for contact resolution ---
        from capabilities.contact_resolver import index_person
        for event in all_events:
            for attendee in event.get("attendees", []):
                index_person(attendee, source="caldav")

        # --- Upsert events to scheduler + create derivative items ---
        self._upsert_events_to_scheduler(all_events, now)

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

    def _do_monitor(self) -> None:
        """Detect calendar changes and emit signals.

        Called by the scheduler via system handler dispatch. Auto-reconnects
        if disconnected.
        """
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            return
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
        except Exception as exc:
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
            except Exception as exc:
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
                    except Exception:
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
                except Exception:
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

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _ensure_sync_registration(self):
        """Register CalDAV sync handler + create recurring system scheduled_item."""
        try:
            from services.scheduler_service import register_system_handler
            register_system_handler('caldav:sync', self.monitor)

            from services.database_service import get_shared_db_service
            import uuid
            db = get_shared_db_service()
            now = utc_now()

            with db.connection() as conn:
                cursor = conn.cursor()
                # Recurring sync item (every 15 minutes)
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid = ?",
                    ('system:caldav:sync',),
                )
                if not cursor.fetchone():
                    cursor.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, recurrence, status,
                            topic, source, external_uid, hidden, created_at)
                         VALUES (?, 'system', 'CalDAV calendar sync', ?, 'interval:15',
                                 'pending', 'caldav:sync', 'caldav', 'system:caldav:sync',
                                 1, ?)""",
                        (uuid.uuid4().hex[:8], now.isoformat(), now.isoformat()),
                    )
                conn.commit()
            logger.info("[caldav] Sync handler registered + scheduled_items ensured")
        except Exception as exc:
            logger.warning("[caldav] _ensure_sync_registration: %s", exc)

    def _upsert_events_to_scheduler(self, events: list, now: "_dt_module.datetime") -> None:
        """Mark-sweep delta sync: upsert events into scheduled_items + create derivatives."""
        import json as _json
        import uuid

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                cursor = conn.cursor()

                # Step 1: Mark existing caldav events for stale check
                cursor.execute(
                    "UPDATE scheduled_items SET status='stale_check' "
                    "WHERE source='caldav' AND item_type='event' AND status='pending'"
                )

                # Step 2: Upsert each event
                for ev in events:
                    uid = ev.get('uid')
                    if not uid:
                        continue

                    external_uid = f"caldav:{uid}"
                    summary = ev.get('summary', 'Event')
                    cal_name = ev.get('calendar_name', '')
                    location = ev.get('location', '')
                    dtstart = ev.get('dtstart')
                    dtend = ev.get('dtend')

                    # Build message line
                    parts = [summary]
                    if cal_name:
                        parts.append(f"[{cal_name}]")
                    if location:
                        parts.append(f"@ {location}")
                    message = " ".join(parts)

                    # Build metadata JSON
                    metadata = _json.dumps({
                        "uid": uid,
                        "dtstart": dtstart.isoformat() if dtstart else None,
                        "dtend": dtend.isoformat() if dtend else None,
                        "location": location,
                        "attendees": ev.get('attendees', []),
                        "recurrence": ev.get('recurrence'),
                        "all_day": ev.get('all_day', False),
                        "calendar_name": cal_name,
                    })

                    due_at = dtstart.isoformat() if dtstart else now.isoformat()

                    cursor.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, status, topic,
                            source, external_uid, metadata, hidden, created_at)
                         VALUES (?, 'event', ?, ?, 'pending', 'calendar',
                                 'caldav', ?, ?, 1, ?)
                         ON CONFLICT(external_uid) DO UPDATE SET
                            message=excluded.message,
                            due_at=excluded.due_at,
                            metadata=excluded.metadata,
                            hidden=1,
                            status='pending'""",
                        (uuid.uuid4().hex[:8], message, due_at,
                         external_uid, metadata, now.isoformat()),
                    )

                # Step 3: Cancel stale events (not seen in this sync)
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='caldav' AND item_type='event' AND status='stale_check'"
                )

                # --- Derivative items ---

                # Upcoming alerts: 15min before events in next 24h
                upcoming_cutoff = now + timedelta(hours=24)
                for ev in events:
                    dtstart = ev.get('dtstart')
                    uid = ev.get('uid')
                    if not dtstart or not uid or dtstart < now or dtstart > upcoming_cutoff:
                        continue
                    if ev.get('all_day'):
                        continue
                    alert_uid = f"caldav:{uid}:alert"
                    alert_due = (dtstart - timedelta(minutes=15)).isoformat()
                    alert_msg = f"In 15 min: {ev.get('summary', 'Event')}"
                    if ev.get('location'):
                        alert_msg += f" @ {ev['location']}"
                    cursor.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, topic,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'caldav', ?, 0, ?)""",
                        (uuid.uuid4().hex[:8], alert_msg, alert_due,
                         alert_uid, now.isoformat()),
                    )

                # Conflict detection
                overlap_pairs = _find_overlap_pairs(events, now)
                for ev_a, ev_b, canon_key in overlap_pairs:
                    conflict_uid = f"caldav:conflict:{canon_key}"
                    conflict_msg = (
                        f"Schedule conflict: \"{ev_a.get('summary', 'Event')}\" and "
                        f"\"{ev_b.get('summary', 'Event')}\" overlap"
                    )
                    cursor.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, topic,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'caldav', ?, 0, ?)""",
                        (uuid.uuid4().hex[:8], conflict_msg, now.isoformat(),
                         conflict_uid, now.isoformat()),
                    )
                    try:
                        from capabilities.signal_bridge import emit_capability_signal
                        emit_capability_signal(
                            "caldav", "novel_observation",
                            conflict_msg,
                            source="caldav:conflict",
                        )
                    except Exception as exc:
                        logger.debug(
                            "[caldav] signal emit failed: %s", exc,
                        )

                # Back-to-back detection (< 5 min gap)
                b2b = _find_back_to_back_pairs(events, now)
                for ev_a, ev_b, gap_min, canon_key in b2b:
                    b2b_uid = f"caldav:b2b:{canon_key}"
                    b2b_msg = (
                        f"Tight transition ({gap_min}min gap): "
                        f"\"{ev_a.get('summary', 'Event')}\" \u2192 "
                        f"\"{ev_b.get('summary', 'Event')}\""
                    )
                    cursor.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, topic,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'caldav', ?, 0, ?)""",
                        (uuid.uuid4().hex[:8], b2b_msg,
                         ev_a.get('dtend', now).isoformat(),
                         b2b_uid, now.isoformat()),
                    )

                # Daily digest (recurring prompt, hidden)
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid = ?",
                    ('caldav:daily-digest',),
                )
                if not cursor.fetchone():
                    digest_due = _next_morning_8am()
                    cursor.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, recurrence, status, topic,
                            source, external_uid, hidden, created_at, is_prompt)
                         VALUES (?, 'prompt',
                                 'Summarize today''s calendar: highlight key meetings, conflicts, and free blocks. Keep it brief — 3-4 sentences.',
                                 ?, 'daily', 'pending', 'calendar',
                                 'caldav', 'caldav:daily-digest', 1, ?, 1)""",
                        (uuid.uuid4().hex[:8], digest_due.isoformat(), now.isoformat()),
                    )

                # First-connect greeting (one-time notification)
                greeting_uid = 'caldav:greeting'
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid = ?",
                    (greeting_uid,),
                )
                if not cursor.fetchone():
                    n = len(events)
                    greeting_msg = (
                        f"Calendar connected! Found {n} event{'s' if n != 1 else ''} "
                        f"across your calendars."
                    )
                    cursor.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, status, topic,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'caldav', ?, 0, ?)""",
                        (uuid.uuid4().hex[:8], greeting_msg, now.isoformat(),
                         greeting_uid, now.isoformat()),
                    )

                conn.commit()
                logger.info("[caldav] Upserted %d events + derivative items", len(events))
        except Exception as exc:
            logger.error("[caldav] _upsert_events_to_scheduler failed: %s", exc)

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
        # caldav_list_events
        # ------------------------------------------------------------------
        def _list_events_execute(topic, params, config=None, telemetry=None):
            """List calendar events from scheduled_items."""
            if not capability.is_connected():
                return {"error": "CalDAV not connected. Configure via /settings."}
            try:
                import json as _json
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()

                date_from = params.get("date_from")
                date_to = params.get("date_to")
                calendar_name = params.get("calendar_name")
                limit = min(int(params.get("limit", 50)), 200)

                query = (
                    "SELECT message, due_at, metadata FROM scheduled_items "
                    "WHERE source='caldav' AND item_type='event' AND status='pending'"
                )
                query_params = []
                if date_from:
                    query += " AND due_at >= ?"
                    query_params.append(date_from)
                if date_to:
                    query += " AND due_at <= ?"
                    query_params.append(date_to)
                query += " ORDER BY due_at ASC LIMIT ?"
                query_params.append(limit)

                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, query_params)
                    rows = cursor.fetchall()

                results = []
                for msg, due_at, meta_raw in rows:
                    meta = _json.loads(meta_raw) if meta_raw else {}
                    if calendar_name and meta.get("calendar_name", "").lower() != calendar_name.lower():
                        continue
                    results.append({
                        "summary": meta.get("uid", msg),
                        "title": msg,
                        "dtstart": due_at,
                        "dtend": meta.get("dtend"),
                        "location": meta.get("location"),
                        "attendees": meta.get("attendees", []),
                        "all_day": meta.get("all_day", False),
                        "calendar_name": meta.get("calendar_name"),
                        "uid": meta.get("uid"),
                    })
                return {"events": results, "count": len(results)}
            except Exception as exc:
                return {"error": f"Failed to list events: {exc}"}

        # ------------------------------------------------------------------
        # caldav_get_event
        # ------------------------------------------------------------------
        def _get_event_execute(topic, params, config=None, telemetry=None):
            """Get a single event by UID from scheduled_items, with CalDAV fallback."""
            if not capability.is_connected():
                return {"error": "CalDAV not connected. Configure via /settings."}
            uid = params.get("uid") or params.get("event_uid")
            if not uid:
                return {"error": "uid is required"}
            try:
                import json as _json
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()

                external_uid = f"caldav:{uid}"
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT message, due_at, metadata FROM scheduled_items "
                        "WHERE external_uid = ? AND item_type='event'",
                        (external_uid,),
                    )
                    row = cursor.fetchone()

                if row:
                    msg, due_at, meta_raw = row
                    meta = _json.loads(meta_raw) if meta_raw else {}
                    return {"event": {
                        "uid": uid,
                        "title": msg,
                        "dtstart": due_at,
                        "dtend": meta.get("dtend"),
                        "location": meta.get("location"),
                        "attendees": meta.get("attendees", []),
                        "all_day": meta.get("all_day", False),
                        "calendar_name": meta.get("calendar_name"),
                        "recurrence": meta.get("recurrence"),
                    }}
                return {"error": f"Event '{uid}' not found"}
            except Exception as exc:
                return {"error": f"Failed to get event: {exc}"}

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
            except Exception as exc:
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
                    except Exception:
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
            except Exception as exc:
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
                    except Exception:
                        continue

                return {"error": f"Event not found (UID: {uid})"}
            except Exception as exc:
                logger.error("[caldav] caldav_delete_event handler failed: %s", exc)
                return {"error": str(exc)}

        # ------------------------------------------------------------------
        # caldav_find_free_slots
        # ------------------------------------------------------------------
        def _find_free_slots_execute(topic, params, config=None, telemetry=None):
            """Find free time slots by querying scheduled_items.

            Clamps results to working hours per day.  The working-hours
            window is interpreted in the user's local timezone (via
            ClientContextService), falling back to UTC when unavailable.
            """
            if not capability.is_connected():
                return {"error": "CalDAV not connected. Configure via /settings."}
            try:
                import json as _json
                from datetime import timezone as _tz
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()

                date_from = params.get("date_from", utc_now().isoformat())
                date_to = params.get("date_to", (utc_now() + timedelta(days=7)).isoformat())
                min_minutes = int(params.get("min_duration_minutes", 30))
                wh_start = int(params.get("working_hours_start", 8))
                wh_end = int(params.get("working_hours_end", 18))

                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT due_at, metadata FROM scheduled_items "
                        "WHERE source='caldav' AND item_type='event' AND status='pending' "
                        "AND due_at >= ? AND due_at <= ? "
                        "ORDER BY due_at ASC",
                        (date_from, date_to),
                    )
                    rows = cursor.fetchall()

                # Build busy periods
                busy = []
                for due_at_str, meta_raw in rows:
                    meta = _json.loads(meta_raw) if meta_raw else {}
                    start = parse_utc(due_at_str)
                    end_str = meta.get("dtend")
                    if end_str:
                        end = parse_utc(end_str)
                    else:
                        end = start + timedelta(hours=1)
                    if not meta.get("all_day", False):
                        busy.append((start, end))

                busy.sort(key=lambda x: x[0])

                # Build per-day working-hours windows in UTC
                tz = _get_user_tz() or _tz.utc
                window_start = parse_utc(date_from)
                window_end = parse_utc(date_to)
                work_windows = []
                day = window_start.astimezone(tz).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                last_day = window_end.astimezone(tz).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                while day <= last_day:
                    ws = day.replace(hour=wh_start).astimezone(_tz.utc)
                    we = day.replace(hour=wh_end).astimezone(_tz.utc)
                    # Clamp to requested range
                    ws = max(ws, window_start)
                    we = min(we, window_end)
                    if ws < we:
                        work_windows.append((ws, we))
                    day += timedelta(days=1)

                # Find gaps within working windows only
                slots = []
                for ww_start, ww_end in work_windows:
                    cursor_time = ww_start
                    for bstart, bend in busy:
                        if bend <= ww_start:
                            continue
                        if bstart >= ww_end:
                            break
                        clamped_start = max(bstart, ww_start)
                        if clamped_start > cursor_time:
                            gap = (clamped_start - cursor_time).total_seconds() / 60
                            if gap >= min_minutes:
                                slots.append({
                                    "start": cursor_time.isoformat(),
                                    "end": clamped_start.isoformat(),
                                    "duration_minutes": int(gap),
                                })
                        cursor_time = max(cursor_time, min(bend, ww_end))
                    if cursor_time < ww_end:
                        gap = (ww_end - cursor_time).total_seconds() / 60
                        if gap >= min_minutes:
                            slots.append({
                                "start": cursor_time.isoformat(),
                                "end": ww_end.isoformat(),
                                "duration_minutes": int(gap),
                            })

                return {"free_slots": slots, "count": len(slots)}
            except Exception as exc:
                return {"error": f"Failed to find free slots: {exc}"}

        # ------------------------------------------------------------------
        # caldav_get_attendees
        # ------------------------------------------------------------------
        def _get_attendees_execute(topic, params, config=None, telemetry=None):
            """Return resolved attendees for a calendar event by UID."""
            if not capability.is_connected():
                return {"error": "CalDAV not connected. Configure via /settings."}
            uid = params.get("uid") or params.get("event_uid")
            if not uid:
                return {"error": "uid is required"}
            try:
                import json as _json
                from services.database_service import get_shared_db_service
                from capabilities.contact_resolver import resolve

                db = get_shared_db_service()
                external_uid = f"caldav:{uid}"
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT message, metadata FROM scheduled_items "
                        "WHERE external_uid = ? AND item_type='event'",
                        (external_uid,),
                    )
                    row = cursor.fetchone()

                if not row:
                    return {"error": f"Event '{uid}' not found"}

                title, meta_raw = row
                meta = _json.loads(meta_raw) if meta_raw else {}
                raw_attendees = meta.get("attendees", [])

                resolved = []
                for email in raw_attendees:
                    matches = resolve(email, limit=1)
                    if matches:
                        resolved.append({
                            "email": email,
                            "name": matches[0].get("name", ""),
                        })
                    else:
                        resolved.append({"email": email, "name": ""})

                return {
                    "event_title": title,
                    "attendees": resolved,
                    "count": len(resolved),
                }
            except Exception as exc:
                return {"error": f"Failed to get attendees: {exc}"}

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
            {
                "name": "caldav_get_attendees",
                "description": (
                    "Get resolved attendees for a calendar event by UID. "
                    "Returns each attendee's email and display name from the "
                    "people index. Use for 'who's in my next meeting?' queries."
                ),
                "parameters": {
                    "uid": {
                        "type": "string",
                        "required": True,
                        "description": "The UID of the calendar event.",
                    },
                },
                "returns": {
                    "event_title": {
                        "type": "string",
                        "description": "Title of the event.",
                    },
                    "attendees": {
                        "type": "array",
                        "description": (
                            "List of attendee dicts with 'email' and 'name' fields. "
                            "Name is empty string if not resolved."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of attendees.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error on failure.",
                    },
                },
                "constraints": {"timeout_seconds": 10},
                "handler": _get_attendees_execute,
            },
        ]

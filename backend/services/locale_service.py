"""Locale Service — single chokepoint for ALL localisation in Chalie.

Rules (enforced by code review):
- ALL dates/times stored in the DB MUST pass through this service and be UTC.
- ALL dates/times shown to users or used in triggers MUST pass through
  this service and be converted to the user's local timezone.
- ALL locale-sensitive values (currency, language, location) MUST be
  read from this service — never directly from telemetry, settings, or env.

The backing store is the ``telemetry`` table (populated by the frontend
heartbeat via ClientContextService.save()). This service is the
exclusive read interface for locale fields.
"""

import logging
from datetime import datetime, timezone
from typing import cast
from zoneinfo import ZoneInfo

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_LOCALE_KEYS = frozenset({
    "timezone", "locale", "language", "currency",
    "location.lat", "location.lon", "location_name",
})


def _read_locale_fields() -> dict[str, object]:
    """Extract locale-relevant fields from the telemetry cache.

    Returns a flat dict with keys like 'timezone', 'locale', 'location.lat'.
    Returns empty dict when no heartbeat has been received yet.
    """
    try:
        from services.heartbeat_service import heartbeat_service
        ctx = heartbeat_service.read()
        if not ctx:
            return {}
        result: dict[str, object] = {}
        for key in _LOCALE_KEYS:
            parts = key.split(".")
            value: object = ctx
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is not None:
                result[key] = value
        return result
    except Exception:
        return {}


def get_timezone() -> ZoneInfo:
    """Return the user's IANA timezone as a ZoneInfo object.

    Falls back to UTC when no heartbeat has been received yet.
    """
    fields = _read_locale_fields()
    tz_name = fields.get("timezone")
    if tz_name:
        try:
            return ZoneInfo(cast(str, tz_name))
        except Exception:
            logger.debug("[LOCALE] Invalid timezone '%s', falling back to UTC", tz_name)
    return ZoneInfo("UTC")


def get_timezone_name() -> str:
    """Return the user's IANA timezone name string (e.g. 'Europe/Malta')."""
    fields = _read_locale_fields()
    return cast(str, fields.get("timezone")) or "UTC"


def get_locale() -> str:
    """Return the user's BCP-47 locale tag (e.g. 'en-MT').

    Falls back to 'en-US' when unavailable.
    """
    fields = _read_locale_fields()
    return cast(str, fields.get("locale")) or "en-US"


def get_language() -> str:
    """Return the user's preferred language (e.g. 'en-GB').

    Falls back to 'en' when unavailable.
    """
    fields = _read_locale_fields()
    return cast(str, fields.get("language")) or "en"


def get_currency() -> str:
    """Return the user's ISO 4217 currency code (e.g. 'EUR').

    Falls back to 'USD' when unavailable.
    """
    fields = _read_locale_fields()
    return cast(str, fields.get("currency")) or "USD"


def get_location() -> dict[str, object]:
    """Return the user's last known location.

    Returns:
        Dict with keys 'lat', 'lon', 'name' (any may be None).
    """
    fields = _read_locale_fields()
    return {
        "lat": fields.get("location.lat"),
        "lon": fields.get("location.lon"),
        "name": fields.get("location_name"),
    }


# ── Format constants ──────────────────────────────────────────────────────

CHAT_TIMESTAMP_FMT = "%d %b %H:%M"

# ── Formatting helpers ────────────────────────────────────────────────────


def format_date(dt: datetime | str | None, fmt: str = "%Y-%m-%d %H:%M", for_ui: bool = False) -> str | None:
    """Format a datetime for storage or display.

    This is THE chokepoint for all date/time formatting in the system.

    Args:
        dt: A datetime (naive or aware), ISO string, or None.
            Naive datetimes are assumed UTC.
        fmt: strftime format string.
        for_ui: If True, converts to user's local timezone before formatting.
                If False, formats as UTC (for DB storage, logs, internal use).

    Returns:
        Formatted string, or None if dt is None/unparseable.
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        if not dt.strip():
            return None
        dt = parse_utc(dt)
    elif isinstance(dt, datetime):
        dt = parse_utc(dt)
    else:
        return None

    if dt.year <= 1:
        return None

    if for_ui:
        dt = dt.astimezone(get_timezone())
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime(fmt)


def to_utc(dt: datetime | str) -> datetime:
    """Ensure a datetime is timezone-aware UTC.

    Use this before ANY database write involving a timestamp.

    Args:
        dt: A datetime (naive or aware), or ISO string.

    Returns:
        Timezone-aware UTC datetime.
    """
    return parse_utc(dt)


def to_local(dt: datetime | str) -> datetime:
    """Convert a datetime to the user's local timezone.

    Use this before displaying any timestamp to the user or evaluating
    triggers (e.g. 'is it past 10am for the user?').

    Args:
        dt: A datetime (naive or aware), or ISO string.

    Returns:
        Timezone-aware datetime in the user's local timezone.
    """
    parsed = parse_utc(dt)
    return parsed.astimezone(get_timezone())


def local_now() -> datetime:
    """Return the current time in the user's local timezone.

    Use for trigger evaluation, 'today/tomorrow' resolution, etc.
    """
    return utc_now().astimezone(get_timezone())


def calculate_interval(
    date_time: datetime | str,
    interval_type: str,
    interval: int,
) -> tuple[datetime, datetime]:
    """Calculate the next recurrence from a UTC base datetime.

    Pure arithmetic — adds the interval to the base datetime and returns
    both UTC and local representations. No quiet-hours logic; that belongs
    at execution time.

    Args:
        date_time: Base datetime (naive or aware) or ISO string. Parsed as UTC.
        interval_type: One of 'day', 'hour', 'minute'.
        interval: Number of units to add.

    Returns:
        Tuple of (utc_datetime, local_datetime).
    """
    from datetime import timedelta

    base = parse_utc(date_time)

    deltas = {
        "day": timedelta(days=interval),
        "hour": timedelta(hours=interval),
        "minute": timedelta(minutes=interval),
    }
    delta = deltas.get(interval_type)
    if delta is None:
        raise ValueError(f"Unknown interval_type: {interval_type!r}. Use 'day', 'hour', or 'minute'.")

    next_utc = base + delta
    next_local = next_utc.astimezone(get_timezone())
    return next_utc, next_local

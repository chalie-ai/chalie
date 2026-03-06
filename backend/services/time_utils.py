"""
Canonical datetime utilities for Chalie.

Rule: ALL datetime values in this codebase must be timezone-aware UTC.
- Use utc_now() instead of datetime.now() or datetime.utcnow()
- Use parse_utc() whenever reading a datetime from SQLite, JSON, or any external source
- Never create naive datetimes (datetimes without tzinfo)
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def parse_utc(value) -> datetime:
    """
    Parse any datetime-like value into a timezone-aware UTC datetime.

    Handles:
    - Already-aware datetime: returned as-is (converted to UTC if needed)
    - Naive datetime: assumed UTC, tzinfo injected
    - ISO string with offset (e.g. "2024-01-01T12:00:00+00:00"): parsed correctly
    - ISO string without offset (e.g. "2024-01-01 12:00:00"): assumed UTC
    - None / unparseable: returns datetime.min in UTC (safe sentinel)
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        try:
            # Normalize SQLite format ("2024-01-01 12:00:00") to ISO 8601
            normalized = value.strip().replace(' ', 'T').replace('Z', '+00:00')
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass

    return datetime.min.replace(tzinfo=timezone.utc)

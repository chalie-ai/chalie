"""
Canonical datetime utilities for Chalie.

Rule: ALL datetime values in this codebase must be timezone-aware UTC.
- Use utc_now() instead of datetime.now() or datetime.utcnow()
- Use parse_utc() whenever reading a datetime from SQLite, JSON, or any external source
- Never create naive datetimes (datetimes without tzinfo)
- Use get_user_tz() to get the user's IANA timezone (for display/conversion)
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: datetime | str) -> datetime:
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


def get_user_tz() -> ZoneInfo:
    from services.locale_service import get_timezone
    return get_timezone()

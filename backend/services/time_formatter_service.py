"""
TimeFormatterService — shared chokepoint for compact elapsed-time formatting.

Replaces open-coded ``_relative_time()`` helpers scattered across the codebase.
All duration values in WorldState and news output pass through here, and every
absolute timestamp surfaced to the LLM (Previous Messages, compaction entries,
tool results) goes through :meth:`local` so the model only ever sees the
user's local wall-clock time, never raw UTC.
"""

import logging

from services.time_utils import utc_now, parse_utc
from services.locale_service import format_date

logger = logging.getLogger(__name__)


class TimeFormatterService:

    @staticmethod
    def duration(seconds: int | float) -> str:
        secs = max(0, int(seconds))
        if secs <= 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m"
        if secs < 86400:
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}h {m}m"
        d = secs // 86400
        h = (secs % 86400) // 3600
        return f"{d}d {h}h"

    @staticmethod
    def ago(past) -> str:
        """Return '{duration} ago' for *past*.

        Accepts either a past datetime (aware or naive — parsed via
        ``parse_utc``) or an int/float number of elapsed seconds.
        Negative deltas (future datetimes, negative seconds) clamp to
        ``'0s ago'``.

        Args:
            past: A datetime object, ISO-8601 string, or elapsed seconds
                  (int/float).  Datetimes must be in the past; future
                  datetimes clamp to '0s ago'.

        Returns:
            Formatted string like '3d 2h ago' or '0s ago'.
        """
        if isinstance(past, (int, float)):
            secs = max(0.0, float(past))
        else:
            try:
                dt = parse_utc(past)
                delta = utc_now() - dt
                secs = max(0.0, delta.total_seconds())
            except Exception as e:
                logger.debug("[TimeFormatter] unparseable input %r → '0s ago': %s", past, e)
                secs = 0.0
        return f"{TimeFormatterService.duration(secs)} ago"

    @staticmethod
    def local(value, fmt: str = "%Y-%m-%d %H:%M") -> str | None:
        """Format *value* in the user's local timezone.

        Delegates to locale_service.format_date(for_ui=True) — the single
        chokepoint for all user-facing timestamp formatting.

        Args:
            value: A datetime (naive or aware), an ISO-8601 string, or a SQLite
                ``YYYY-MM-DD HH:MM:SS`` string. Naive inputs are treated as UTC
                (matches ``parse_utc``). ``None`` and unparseable strings return
                ``None`` so callers can decide their own placeholder.
            fmt: ``strftime`` format string, default ``'%Y-%m-%d %H:%M'``.

        Returns:
            Formatted local-time string, or ``None`` if *value* is missing or
            unparseable.
        """
        return format_date(value, fmt, for_ui=True)

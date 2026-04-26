"""
TimeFormatterService — shared chokepoint for compact elapsed-time formatting.

Replaces open-coded ``_relative_time()`` helpers scattered across the codebase.
All duration values in WorldState and news output pass through here, and every
absolute timestamp surfaced to the LLM (Previous Messages, compaction entries,
tool results) goes through :meth:`local` so the model only ever sees the
user's local wall-clock time, never raw UTC.
"""

import logging
from datetime import datetime

from services.time_utils import utc_now, parse_utc, get_user_tz

logger = logging.getLogger(__name__)


class TimeFormatterService:
    """Compact, unit-normalised duration formatter.

    All methods are static — no state, no constructor arguments, no DI.
    """

    @staticmethod
    def duration(seconds: int | float) -> str:
        """Return a compact human-readable duration string for *seconds*.

        Tiers (breakpoints are inclusive on the lower bound):
          0   – 60s    → '{X}s'      e.g. '0s', '45s', '60s'
          61  – 3599s  → '{X}m'      e.g. '7m', '59m'
          3600– 86399s → '{X}h {Y}m' e.g. '1h 0m', '23h 59m'
          86400+s      → '{X}d {Y}h' e.g. '1d 0h', '3d 12h'

        Always non-negative — caller adds directional context ('ago' / 'in')
        in its own template. Negative input clamps to ``'0s'`` (defensive;
        matches the ``ago()`` contract). Fractional seconds are floored at
        each unit boundary.

        Args:
            seconds: Duration in seconds, may be fractional. Negative values
                clamp to zero.

        Returns:
            Compact formatted string, e.g. '2h 30m'.
        """
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

        Single chokepoint for any absolute timestamp the LLM will see — Previous
        Messages, compaction entries, tool results, scheduler confirmations, etc.
        Storing UTC and rendering local is the rule; surfacing raw UTC to the
        model is the bug this method exists to prevent.

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
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            dt = value if isinstance(value, datetime) else parse_utc(value)
            dt = parse_utc(dt)  # normalise tz to UTC
            if dt.year <= 1:
                return None
            return dt.astimezone(get_user_tz()).strftime(fmt)
        except Exception as e:
            logger.debug("[TimeFormatter] local() unparseable %r: %s", value, e)
            return None

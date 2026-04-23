"""
TimeFormatterService — shared chokepoint for compact elapsed-time formatting.

Replaces open-coded ``_relative_time()`` helpers scattered across the codebase.
All duration values in WorldState, news, and introspect output pass through here.
"""

import logging

from services.time_utils import utc_now, parse_utc

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

"""Cron schedule helpers — a real 5-field crontab engine.

A schedule is five crontab field expressions:

    minute   minute of hour  0-59   ``*`` = every
    hour     hour of day     0-23   ``*`` = every
    dom      day of month    1-31   ``*`` = every
    month    month of year   1-12   ``*`` = every
    dow      day of week     0-7    ``*`` = every

Fields are expressed in the USER'S LOCAL timezone (they come straight from
the World State wall clock — the LLM and UI never perform timezone maths).
There is no materialized ``due_at`` to walk toward: the poller wakes on every
wall-clock minute and asks ``matches`` a stateless yes/no question — does the
current instant, converted to local time, satisfy all five fields?

Day-of-week numbering follows standard crontab: 0 = Sunday, 1 = Monday, …,
6 = Saturday. 7 is also accepted and treated as Sunday (folded to 0 on parse).

Day-of-month / day-of-week interaction follows the Vixie crond default: when
*both* fields are restricted (neither is ``*``) the schedule fires on either
(a logical OR). When only one is restricted it must match; when neither is
restricted the day is always a match.

Fields are numeric-only — no name tokens like ``jan`` or ``mon``.

Two wall-clock edge cases are accepted by design (the dumb-cron model owns
no calendar arithmetic, only a per-minute equality test):

- DST transitions. A fixed local HH:MM that the spring-forward gap skips
  never matches that day (the minute has no UTC instant); a fall-back
  repeat makes the local minute occur twice, so a fixed HH:MM fires twice.
  Callers who cannot tolerate either use ``hour="*"``, ``minute="*"``, etc.
- Short months. ``dom`` is a literal day number, so 29/30/31 simply never
  match in months without that day (e.g. dom=31 skips February and the
  30-day months) — there is no "last day of month" roll.
"""

from __future__ import annotations

import functools
from datetime import datetime

from services.locale_service import LocaleService


@functools.lru_cache(maxsize=256)
def parse_field(expr: str, lo: int, hi: int) -> frozenset[int]:
    """Parse ONE crontab field expression into the set of ints it matches.

    Grammar (numeric only, no name tokens): comma-separated tokens, each one of:

        ``*``          every value in [lo, hi]
        ``*/S``        lo, lo+S, lo+2S, … within [lo, hi]   (S >= 1)
        ``N``          exactly N
        ``N/S``        N, N+S, N+2S, … up to hi             (S >= 1; Vixie N-max/S)
        ``N-M``        inclusive range N..M
        ``N-M/S``      N, N+S, … within N..M               (S >= 1)

    Raises ``ValueError`` with a clear message on empty expr, malformed token,
    value outside [lo, hi], N > M in a range, or step S <= 0.

    The result is memoised via ``lru_cache`` on ``(expr, lo, hi)``.
    """
    if not expr or not expr.strip():
        raise ValueError("Empty cron field expression.")

    values: set[int] = set()
    for token in expr.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"Empty token in cron expression {expr!r}.")
        values |= _parse_token(token, lo, hi)
    return frozenset(values)


def _parse_token(token: str, lo: int, hi: int) -> set[int]:
    step = None
    if "/" in token:
        token, s = token.split("/", 1)
        try:
            step = int(s)
        except ValueError:
            raise ValueError(f"Invalid step in token {token!r}.")
        if step <= 0:
            raise ValueError(f"Step must be >= 1, got {step}.")

    if token == "*":
        if step is not None:
            return set(range(lo, hi + 1, step))
        return set(range(lo, hi + 1))

    if "-" in token:
        parts = token.split("-", 1)
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid range in token {token!r}.")
        if start > end:
            raise ValueError(
                f"Range start {start} is greater than end {end} in token {token!r}."
            )
        if start < lo or end > hi:
            raise ValueError(
                f"Range {start}-{end} is outside allowed bounds [{lo}, {hi}] "
                f"in token {token!r}."
            )
        if step is not None:
            return set(range(start, end + 1, step))
        return set(range(start, end + 1))

    # Plain integer (with optional step)
    try:
        n = int(token)
    except ValueError:
        raise ValueError(f"Invalid cron token {token!r}.")
    if n < lo or n > hi:
        raise ValueError(
            f"Value {n} is outside allowed bounds [{lo}, {hi}] in token {token!r}."
        )
    if step is not None:
        # ``N/S`` means N, N+S, N+2S, … up to hi
        return set(range(n, hi + 1, step))
    return {n}


def validate_cron(
    minute: str, hour: str, dom: str, month: str, dow: str
) -> None:
    """Validate five crontab field expressions. Raises ``ValueError`` on the
    first invalid field, prefixed with the field name.

    Any combination of fields is legal — standard crontab semantics.
    """
    _validate_one("minute", minute, 0, 59)
    _validate_one("hour", hour, 0, 23)
    _validate_one("dom", dom, 1, 31)
    _validate_one("month", month, 1, 12)
    _validate_one("dow", dow, 0, 7)


def _validate_one(name: str, expr: str, lo: int, hi: int) -> None:
    try:
        parse_field(expr, lo, hi)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None


def matches(
    now_utc: datetime,
    minute: str,
    hour: str,
    dom: str,
    month: str,
    dow: str,
) -> bool:
    """True iff ``now_utc``, converted to the user's local wall clock, matches
    all five crontab fields.

    Day-of-week in the parsed set has any 7s folded to 0 so both 0 and 7
    match Sunday.
    """
    local = now_utc.astimezone(LocaleService.get_timezone())

    minset = parse_field(minute, 0, 59)
    hourset = parse_field(hour, 0, 23)
    domset = parse_field(dom, 1, 31)
    monthset = parse_field(month, 1, 12)
    dowset = parse_field(dow, 0, 7)
    # Fold 7 -> 0 so both 0 and 7 match Sunday.
    if 7 in dowset:
        dowset = frozenset((dowset - {7}) | {0})

    dow_now = (local.weekday() + 1) % 7  # Python Mon=0 -> cron Sun=0

    dom_star = dom.strip() == "*"
    dow_star = dow.strip() == "*"
    if dom_star and dow_star:
        day_ok = True
    elif dom_star != dow_star:
        day_ok = (dow_now in dowset) if dom_star else (local.day in domset)
    else:
        day_ok = (local.day in domset) or (dow_now in dowset)

    return (
        (local.minute in minset)
        and (local.hour in hourset)
        and (local.month in monthset)
        and day_ok
    )

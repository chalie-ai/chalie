"""Cron schedule helpers — the single source of truth for the scheduler's
day-of-month / hour / minute recurrence model.

A schedule fires on a coarse-to-fine crontab of three fields:

    cron_dom    day-of-month  1-31   (NULL = every)
    cron_hour   hour          0-23   (NULL = every)
    cron_minute minute        0-59   (NULL = every)

The fields are expressed in the USER'S LOCAL timezone (they come straight from
the World State wall clock — the LLM and UI never perform timezone maths).
There is no materialized ``due_at`` to walk toward: the poller wakes on every
wall-clock minute and asks ``matches`` a stateless yes/no question — does the
current instant, converted to local time, satisfy the three fields? A NULL
field always answers yes.

Invariant (``validate_cron``): the "every" (NULL) fields must form a prefix from
the coarse end — you cannot set "every" on a field finer than a fixed one. Only
these four shapes are legal (E = every, F = fixed):

    E E E   every minute
    E E F   every hour at :MM
    E F F   every day at HH:MM
    F F F   monthly on day DD at HH:MM
"""

from datetime import datetime

from services.locale_service import get_timezone


def validate_cron(dom: int | None, hour: int | None, minute: int | None) -> None:
    """Validate a (dom, hour, minute) cron triple. Raises ValueError on violation.

    Enforces field ranges and the coarse-to-fine "every-prefix" rule: a fixed
    field forces every finer field to be fixed too.
    """
    if dom is not None and not (1 <= dom <= 31):
        raise ValueError(f"Day of month must be 1-31, got {dom!r}.")
    if hour is not None and not (0 <= hour <= 23):
        raise ValueError(f"Hour must be 0-23, got {hour!r}.")
    if minute is not None and not (0 <= minute <= 59):
        raise ValueError(f"Minute must be 0-59, got {minute!r}.")

    if hour is not None and minute is None:
        raise ValueError(
            "Minute cannot be 'every' when Hour is fixed — 'every' is only allowed "
            "on a field coarser than every fixed field."
        )
    if dom is not None and (hour is None or minute is None):
        raise ValueError(
            "Hour and Minute cannot be 'every' when Day of month is fixed — 'every' "
            "is only allowed on a field coarser than every fixed field."
        )


def matches(now_utc: datetime, dom: int | None, hour: int | None, minute: int | None) -> bool:
    """True iff now (converted to the user's local wall clock) matches the cron
    fields; a NULL field matches every value (§ every-prefix invariant).

    Two wall-clock edge cases are accepted by design (the dumb-cron model owns
    no calendar arithmetic, only a per-minute equality test):

    - DST transitions. A fixed local HH:MM that the spring-forward gap skips
      never matches that day (the minute has no UTC instant); a fall-back
      repeat makes the local minute occur twice, so a fixed HH:MM fires twice.
      Callers who cannot tolerate either use ``cron_hour=None`` (every-hour).
    - Short months. ``cron_dom`` is a literal local day number, so 29/30/31
      simply never match in months without that day (e.g. dom=31 skips
      February and the 30-day months) — there is no "last day of month" roll.
    """
    local = now_utc.astimezone(get_timezone())
    return (
        (minute is None or local.minute == minute)
        and (hour is None or local.hour == hour)
        and (dom is None or local.day == dom)
    )

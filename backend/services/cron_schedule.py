"""Cron schedule helpers — the single source of truth for the scheduler's
day-of-month / hour / minute recurrence model.

A schedule fires on a coarse-to-fine crontab of three fields:

    cron_dom    day-of-month  1-31   (NULL = every)
    cron_hour   hour          0-23   (NULL = every)
    cron_minute minute        0-59   (NULL = every)

The fields are expressed in the USER'S LOCAL timezone (they come straight from
the World State wall clock — the LLM and UI never perform timezone maths). The
stored ``start_at`` / ``due_at`` are UTC. ``next_fire`` bridges the two: it walks
forward in local wall-clock time to the first instant that matches the fixed
fields, then hands back the UTC instant for storage and the poller.

Invariant (``validate_cron``): the "every" (NULL) fields must form a prefix from
the coarse end — you cannot set "every" on a field finer than a fixed one. Only
these four shapes are legal (E = every, F = fixed):

    E E E   every minute
    E E F   every hour at :MM
    E F F   every day at HH:MM
    F F F   monthly on day DD at HH:MM
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from services.locale_service import get_timezone
from services.time_utils import parse_utc, utc_now

# Safety bound for the wall-clock walk. Real searches converge in well under a
# hundred jumps (minute/hour jumps are direct; day steps are one per day and a
# valid day-of-month recurs at least every ~2 months). This only guards against
# a pathological tz making the loop spin.
_MAX_STEPS = 400_000


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


def next_fire(
    not_before: datetime | str,
    dom: int | None,
    hour: int | None,
    minute: int | None,
) -> tuple[datetime, datetime]:
    """Return the first cron match at or after ``not_before``.

    Args:
        not_before: UTC datetime (naive assumed UTC) or ISO string. The returned
            fire is the earliest cron match whose instant is >= this.
        dom/hour/minute: the cron fields (NULL/None = every), in local wall clock.

    Returns:
        (utc_datetime, local_datetime) of the next fire.

    Raises:
        ValueError: if the cron triple is invalid, or no match is found within
            the safety bound.
    """
    validate_cron(dom, hour, minute)

    tz = get_timezone()
    floor_utc = parse_utc(not_before)

    # Walk in NAIVE local wall-clock time so day/DST arithmetic is correct, then
    # re-attach the zone to get the true UTC instant.
    naive = floor_utc.astimezone(tz).replace(second=0, microsecond=0, tzinfo=None)
    if _to_utc(naive, tz) < floor_utc:
        # Rounding down to the minute landed before the floor — step to next minute.
        naive += timedelta(minutes=1)

    for _ in range(_MAX_STEPS):
        if (
            (minute is None or naive.minute == minute)
            and (hour is None or naive.hour == hour)
            and (dom is None or naive.day == dom)
        ):
            aware = naive.replace(tzinfo=tz)
            return aware.astimezone(timezone.utc), aware

        # Jump to the next plausible candidate. The every-prefix invariant means a
        # fixed coarser field always sits above fixed finer fields, so fixing the
        # finest mismatch never disturbs an already-matched finer field.
        if minute is not None and naive.minute != minute:
            naive += timedelta(minutes=(minute - naive.minute) % 60)
        elif hour is not None and naive.hour != hour:
            naive += timedelta(hours=(hour - naive.hour) % 24)
        else:  # dom fixed and mismatched — step a day, keeping HH:MM
            naive += timedelta(days=1)

    raise ValueError(
        f"No cron match within bound for dom={dom} hour={hour} minute={minute}."
    )


def next_due_from(
    anchor: datetime | str,
    dom: int | None,
    hour: int | None,
    minute: int | None,
) -> tuple[datetime, datetime]:
    """Compute the next materialized ``due_at`` from an anchor, never in the past.

    The anchor is floored to "now" so a future ``start_at`` is honoured while a
    stalled poller or a past ``start_at`` never triggers a catch-up storm of
    missed occurrences — cron only fires forward.

    Use with ``anchor = start_at`` for the first fire, or
    ``anchor = <previous due_at> + 1 minute`` to advance after a fire.

    Returns:
        (utc_datetime, local_datetime) of the next fire.
    """
    floored = max(parse_utc(anchor), utc_now())
    return next_fire(floored, dom, hour, minute)


def _to_utc(naive_local: datetime, tz: ZoneInfo) -> datetime:
    """Attach ``tz`` to a naive local datetime and convert to UTC."""
    return naive_local.replace(tzinfo=tz).astimezone(timezone.utc)

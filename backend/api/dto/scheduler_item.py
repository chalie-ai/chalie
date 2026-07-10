"""DTOs for the scheduler resource — read/create/update HTTP contract.

One file per concept (the namespace convention; mirrors ``dto/list.py``).
Field-level rules (message length) live on ``pydantic.Field`` / validators;
the crontab-shape rule on the five field expressions
(``minute``/``hour``/``day``/``month``/``weekday``) lives on a model validator
shared by create and update, delegating to
:func:`services.cron_schedule.validate_cron` — the single source of truth for
the cron shape. Each field is a standard crontab expression (``*``, ``N``,
``N-M``, ``*/S``, ``N-M/S`` and comma-unions; ``*`` = every). Datetimes
serialize as ISO-8601 UTC via :class:`backend.api.dto.base.DTO`; the read DTO
additionally localizes ``start_at`` to the user's timezone for display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from services.cron_schedule import validate_cron
from services.locale_service import format_date
from .base import DTO

# ISO 8601 with offset — the localized wire format for start_at on the read
# DTO, matching the format the frontend already parses for other localized
# timestamps.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


class SchedulerItem(DTO):
    """Read shape for a scheduled item — the full list row.

    ``scheduled_items`` is a prompt-only, crontab table: one row per
    schedule, forever. ``id`` is the SQLite ``INTEGER PRIMARY KEY
    AUTOINCREMENT`` — also the schedule's ``turn_id`` on the ``'schedule'``
    channel. ``start_at`` is localized to the user's timezone for display
    (the DB always stores it as UTC). ``minute``/``hour``/``day``/``month``/
    ``weekday`` mirror the DB's ``cron_minute``/``cron_hour``/``cron_dom``/
    ``cron_month``/``cron_dow`` — each a crontab field expression (``*`` =
    every). There is no ``due_at``/``status``/``group_id``/``source``/
    ``external_uid`` — those columns, and the lifecycle they modeled, no
    longer exist.
    """

    id: int
    message: str
    start_at: str | None
    minute: str
    hour: str
    day: str
    month: str
    weekday: str
    enabled: int
    channel: str | None
    created_by_session: str | None
    created_at: datetime

    @field_validator("start_at", mode="before")
    @classmethod
    def _localize(cls, value: object) -> object:
        return format_date(value, _LOCAL_ISO_FMT, for_ui=True) if isinstance(value, str) else value


class SchedulerTurn(DTO):
    """One active prompt-schedule thread, collapsed to its ``turn_id`` (§13.5).

    The scheduler dock lists these — each is a growing, replyable thread on the
    ``schedule`` channel. ``turn_id`` is now simply the schedule's own ``id``
    (no separate stored turn key). ``preview`` is the prompt; ``gist`` its
    generated one-line label (``None`` until the first fire generates it);
    ``minute``/``hour``/``day``/``month``/``weekday`` are the schedule's crontab
    field expressions (``*`` = every)."""

    turn_id: int
    gist: str | None = None
    preview: str
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    weekday: str = "*"


class _ScheduledItemWrite(DTO):
    """Shared create/update body + the crontab-shape validation rule.

    ``channel`` is create-only; ``SchedulerItemCreate`` adds it while
    ``SchedulerItemUpdate`` inherits this base unchanged (PUT never writes
    ``channel``). ``start_at`` is optional local wall-clock ISO text (default
    = local now, applied by the handler via ``parse_local``); a past
    ``start_at`` is legal — it simply means the schedule is already past its
    activation floor. The five cron fields each default to ``"*"`` (every) and
    are validated as crontab expressions by ``validate_cron``. ``enabled`` is a
    plain field on this write body (not a separate verb): a disabled schedule is
    skipped by the poller, and re-enabling via update simply resumes matching
    against the current wall-clock minute — there is no ``due_at`` to recompute.
    """

    message: str = Field(..., min_length=1, max_length=1000)
    start_at: str | None = None
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    weekday: str = "*"
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_cron_fields(self) -> Self:
        try:
            validate_cron(self.minute, self.hour, self.day, self.month, self.weekday)
        except ValueError as exc:
            raise PydanticCustomError("value_error", str(exc), None) from exc
        return self


class SchedulerItemCreate(_ScheduledItemWrite):
    """Inbound body to create a scheduled item."""

    channel: str = "general"


class SchedulerItemUpdate(_ScheduledItemWrite):
    """Inbound body to update a scheduled item; ``channel`` is never writable."""

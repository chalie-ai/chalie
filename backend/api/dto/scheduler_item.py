"""DTOs for the scheduler resource — read/create/update HTTP contract.

One file per concept (the namespace convention; mirrors ``dto/list.py``).
Field-level rules (message length) live on ``pydantic.Field`` /
validators; the cross-field cron rule (the every-prefix invariant on
``day``/``hour``/``minute``) lives on a model validator shared by create and
update, delegating to :func:`services.cron_schedule.validate_cron` — the
single source of truth for the cron shape. Datetimes serialize as ISO-8601
UTC via :class:`backend.api.dto.base.DTO`; the read DTO additionally
localizes ``start_at``/``due_at`` to the user's timezone for display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from services.cron_schedule import validate_cron
from services.locale_service import format_date
from .base import DTO

# ISO 8601 with offset — the localized wire format for start_at/due_at on the
# read DTO, matching the format the frontend already parses for other
# localized timestamps.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


class SchedulerItem(DTO):
    """Read shape for a scheduled item — the full list row.

    ``start_at``/``due_at`` are localized to the user's timezone for display
    (the DB always stores them as UTC). ``day``/``hour``/``minute`` mirror the
    DB's ``cron_dom``/``cron_hour``/``cron_minute`` (``None`` = every).
    ``item_type`` is internal-only now (every user-facing row is ``'prompt'``)
    and is not part of this read shape.

    The 15-column list read adds ``source`` and ``external_uid``; the
    13-column get/create/update/group reads omit them (both default to
    ``None`` so one DTO covers every read).
    """

    id: str
    message: str
    start_at: str
    due_at: str
    day: int | None
    hour: int | None
    minute: int | None
    status: str
    enabled: int
    channel: str | None
    created_by_session: str | None
    created_at: datetime
    group_id: str | None
    source: str | None = None
    external_uid: str | None = None

    @field_validator("start_at", "due_at", mode="before")
    @classmethod
    def _localize(cls, value: object) -> object:
        return format_date(value, _LOCAL_ISO_FMT, for_ui=True) if isinstance(value, str) else value


class SchedulerTurn(DTO):
    """One active prompt-schedule thread, collapsed to its ``turn_id`` (§13.5).

    The scheduler dock lists these — each is a growing, replyable thread on the
    ``schedule`` channel. A series' occurrences share one ``turn_id``, so the row
    is grouped to one entry. ``preview`` is the prompt; ``gist`` its generated
    one-line label (``None`` until the first fire generates it); ``day``/
    ``hour``/``minute`` are the series' cron fields (``None`` = every)."""

    turn_id: int
    gist: str | None = None
    preview: str
    day: int | None = None
    hour: int | None = None
    minute: int | None = None


class _ScheduledItemWrite(DTO):
    """Shared create/update body + the cron every-prefix validation rule.

    ``channel`` is create-only; ``SchedulerItemCreate`` adds it while
    ``SchedulerItemUpdate`` inherits this base unchanged (PUT never writes
    ``channel``). ``item_type`` is always ``'prompt'`` for user-created rows
    and is set by the handler, not carried here. ``start_at`` is optional
    local wall-clock ISO text (default = local now, applied by the handler via
    ``parse_local``); a past ``start_at`` is legal — it simply floors to now.
    ``enabled`` is a plain field on this write body (not a separate verb): a
    disabled series is skipped by the poller, and re-enabling via update
    resumes at the next occurrence (the handler recomputes ``due_at`` forward).
    """

    message: str = Field(..., min_length=1, max_length=1000)
    start_at: str | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_cron_fields(self) -> Self:
        try:
            validate_cron(self.day, self.hour, self.minute)
        except ValueError as exc:
            raise PydanticCustomError("value_error", str(exc), None) from exc
        return self


class SchedulerItemCreate(_ScheduledItemWrite):
    """Inbound body to create a scheduled item."""

    channel: str = "general"


class SchedulerItemUpdate(_ScheduledItemWrite):
    """Inbound body to update a pending item; ``channel`` is never writable."""

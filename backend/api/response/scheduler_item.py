"""Response DTO for the scheduler endpoint group.

Mirrors the field shape of the DTO this endpoint group replaced, so the wire
format stays identical. Field renames DB → wire: ``cron_minute`` →
``minute``, ``cron_hour`` → ``hour``, ``cron_dom`` → ``day``, ``cron_month``
→ ``month``, ``cron_dow`` → ``weekday``. ``start_at`` is localized to the
user's timezone for display (the DB always stores it as UTC).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from pydantic import field_validator

from models.scheduled_item import ScheduledItem
from services.locale_service import LocaleService
from .response import Response

# ISO 8601 with offset — the localized wire format for start_at, matching the
# format the frontend already parses for other localized timestamps.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


class SchedulerItem(Response):
    """Read shape for one scheduled item — the full list row."""

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
        return LocaleService.format_date(value, _LOCAL_ISO_FMT, for_ui=True) if isinstance(value, str) else value

    @classmethod
    def from_model(cls, item: ScheduledItem) -> "SchedulerItem":
        """Project a :class:`~models.scheduled_item.ScheduledItem` into this read DTO."""
        return cls(
            id=cast(int, item.id),
            message=item.message,
            start_at=item.start_at,
            minute=item.cron_minute,
            hour=item.cron_hour,
            day=item.cron_dom,
            month=item.cron_month,
            weekday=item.cron_dow,
            enabled=item.enabled,
            channel=item.channel,
            created_by_session=item.created_by_session,
            created_at=cast(datetime, item.created_at),
        )

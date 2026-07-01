"""DTOs for the scheduler resource — read/create/update HTTP contract.

One file per concept (the namespace convention; mirrors ``dto/list.py``).
Field-level rules (message length, item_type membership) live on
``pydantic.Field`` / validators; cross-field scheduling rules (future
``due_at``, recurrence shape) live on a model validator
shared by create and update. Datetimes serialize as ISO-8601 UTC via
:class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from services.time_utils import parse_utc, utc_now
from .base import DTO

_VALID_TYPES = frozenset({"notification", "prompt"})
_VALID_RECURRENCES = frozenset({"daily", "weekly", "monthly", "weekdays", "hourly"})
_INTERVAL_PREFIX = "interval:"
_INTERVAL_MIN = 1
_INTERVAL_MAX = 1440


class SchedulerItem(DTO):
    """Read shape for a scheduled item — the 14-column list row.

    The 12-column reads (get/create/update/group) omit ``source`` and
    ``external_uid``; both default to ``None`` so one DTO covers every read.
    """

    id: str
    item_type: str
    message: str
    due_at: datetime
    recurrence: str | None
    status: str
    channel: str | None
    created_by_session: str | None
    created_at: datetime
    last_fired_at: datetime | None
    group_id: str | None
    is_prompt: bool
    source: str | None = None
    external_uid: str | None = None


class SchedulerTurn(DTO):
    """One active prompt-schedule thread, collapsed to its ``turn_id`` (§13.5).

    The scheduler dock lists these — each is a growing, replyable thread on the
    ``schedule`` channel. A series' occurrences share one ``turn_id``, so the row
    is grouped to one entry. ``preview`` is the prompt; ``gist`` its generated
    one-line label (``None`` until the first fire generates it); ``next_due_at``
    is the soonest still-pending occurrence (``None`` for a fired one-shot)."""

    turn_id: int
    gist: str | None = None
    preview: str
    recurrence: str | None = None
    last_fired_at: datetime | None = None
    next_due_at: datetime | None = None


class _ScheduledItemWrite(DTO):
    """Shared create/update body + cross-field scheduling rules.

    ``channel`` is create-only; ``SchedulerItemCreate`` adds it while
    ``SchedulerItemUpdate`` inherits this base unchanged (PUT never writes
    ``channel``). ``is_prompt`` is derived in the handler, not carried here.
    """

    message: str = Field(..., min_length=1, max_length=1000)
    due_at: datetime
    item_type: str = "notification"
    recurrence: str | None = None

    @field_validator("due_at", mode="before")
    @classmethod
    def _parse_due_at(cls, value: object) -> object:
        return parse_utc(value) if isinstance(value, str) else value

    @field_validator("item_type")
    @classmethod
    def _item_type_must_be_user_settable(cls, value: str) -> str:
        if value in ("event", "system"):
            raise PydanticCustomError(
                "value_error", "item_type 'event' and 'system' are reserved for internal use", None
            )
        if value not in _VALID_TYPES:
            raise PydanticCustomError(
                "value_error", f"item_type must be one of: {', '.join(sorted(_VALID_TYPES))}", None
            )
        return value

    @field_validator("recurrence", mode="before")
    @classmethod
    def _normalize_recurrence(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        recurrence = value.strip()
        if recurrence in _VALID_RECURRENCES:
            return recurrence
        if recurrence.startswith(_INTERVAL_PREFIX):
            try:
                minutes = int(recurrence[len(_INTERVAL_PREFIX):])
            except ValueError:
                raise PydanticCustomError(
                    "value_error", "interval recurrence must be 'interval:N' where N is 1-1440", None
                ) from None
            if not (_INTERVAL_MIN <= minutes <= _INTERVAL_MAX):
                raise PydanticCustomError(
                    "value_error", "interval must be between 1 and 1440 minutes", None
                )
            return f"interval:{minutes}"
        raise PydanticCustomError(
            "value_error",
            f"recurrence must be one of: {', '.join(sorted(_VALID_RECURRENCES))}, or 'interval:N'",
            None,
        )

    @model_validator(mode="after")
    def _enforce_scheduling_rules(self) -> Self:
        if self.due_at <= utc_now():
            raise PydanticCustomError("value_error", "due_at must be in the future", None)
        return self


class SchedulerItemCreate(_ScheduledItemWrite):
    """Inbound body to create a scheduled item."""

    channel: str = "general"


class SchedulerItemUpdate(_ScheduledItemWrite):
    """Inbound body to update a pending item; ``channel`` is never writable."""

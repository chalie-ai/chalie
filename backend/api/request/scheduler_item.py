"""Inbound DTO for the scheduler endpoint group.

One request DTO covers both create and update — the base contract's single
``post()`` handler branches on :meth:`~api.endpoint.Endpoint.is_create`. Unlike
``ProviderRequest``, every field here is REQUIRED-with-default rather than
optional: the legacy handlers used REPLACE semantics on update (a PUT body
that omitted a cron field reset it to ``"*"``, an omitted ``enabled`` reset it
to ``True``), and this DTO preserves that exactly — it is never treated as a
partial/PATCH body. ``channel`` is accepted on every request but the endpoint
must never write it on update (create-only, matching the legacy contract).

The five cron fields each default to ``"*"`` (every) and are validated as
crontab expressions by ``validate_cron`` — the single source of truth for the
cron shape (``*``, ``N``, ``N-M``, ``*/S``, ``N-M/S`` and comma-unions).
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from services.cron_schedule import validate_cron
from .request import Request


class SchedulerItemRequest(Request):
    """Inbound body for scheduler create/update — REPLACE semantics on update."""

    message: str = Field(..., min_length=1, max_length=1000)
    start_at: str | None = None
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    weekday: str = "*"
    enabled: bool = True
    channel: str = "general"

    @model_validator(mode="after")
    def _validate_cron_fields(self) -> Self:
        try:
            validate_cron(self.minute, self.hour, self.day, self.month, self.weekday)
        except ValueError as exc:
            raise PydanticCustomError("value_error", str(exc), None) from exc
        return self

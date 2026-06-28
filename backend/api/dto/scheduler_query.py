"""Query-param DTOs for the scheduler resource's non-CRUD endpoints.

Each is parsed from the query string via ``expects(..., source="args")``.
``extra="forbid"`` (inherited from :class:`DTO`) rejects unknown params.
The group endpoint preserves the legacy lenient ``limit`` (a non-int coerces
to its default, unlike the list endpoint which 422s on a non-int limit).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import DTO

_LIST_LIMIT_CAP = 200
_LIST_LIMIT_DEFAULT = 50
_HISTORY_DEFAULT_DAYS = 30
_GROUP_LIMIT_CAP = 50
_GROUP_LIMIT_DEFAULT = 10


class SchedulerListQuery(DTO):
    """Filters for GET /scheduler — status, hidden-flag, pagination."""

    status: Literal["all", "pending", "fired", "failed", "cancelled"] = "all"
    include_hidden: bool = False
    limit: int = Field(default=_LIST_LIMIT_DEFAULT, le=_LIST_LIMIT_CAP)
    offset: int = Field(default=0, ge=0)


class SchedulerHistoryQuery(DTO):
    """Params for DELETE /scheduler/history."""

    older_than_days: int = Field(default=_HISTORY_DEFAULT_DAYS, ge=1)


class SchedulerGroupQuery(DTO):
    """Params for GET /scheduler/group/<group_id>.

    A non-int ``limit`` coerces to the default (legacy lenient behavior),
    distinct from the list endpoint which rejects it with 422.
    """

    limit: int = _GROUP_LIMIT_DEFAULT

    @field_validator("limit", mode="before")
    @classmethod
    def _lenient_limit(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                return _GROUP_LIMIT_DEFAULT
        return min(value, _GROUP_LIMIT_CAP) if isinstance(value, int) else value

"""Query-param DTO for the scheduler list endpoint.

Parsed from the query string via ``expects(..., source="args")``.
``extra="forbid"`` (inherited from :class:`DTO`) rejects unknown params.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO

_LIST_LIMIT_CAP = 200
_LIST_LIMIT_DEFAULT = 50


class SchedulerListQuery(DTO):
    """Pagination for GET /scheduler.

    There is no status or hidden filter: the prompt-only table has neither a
    ``status`` nor a ``hidden`` column. The endpoint always returns every live
    schedule (one row each), labelled active/disabled by ``enabled``.
    """

    limit: int = Field(default=_LIST_LIMIT_DEFAULT, le=_LIST_LIMIT_CAP)
    offset: int = Field(default=0, ge=0)

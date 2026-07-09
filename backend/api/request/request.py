"""Abstract base for every inbound request DTO.

Subclass :class:`Request` to declare the request contract for an endpoint.
``extra="forbid"`` rejects unknown keys outright; every ``datetime`` field
serializes as ISO-8601 UTC (``+00:00``), the exact format the frontend consumes,
so handlers never reformat dates themselves.
"""

from __future__ import annotations

from api.dto.base import DTO


class Request(DTO):
    """Abstract base for every inbound request DTO.

    Every inbound payload DTO must inherit from this class. It is intentionally
    empty — subclasses add the fields that define their endpoint's contract.
    The inherited ``extra="forbid"`` from :class:`DTO` guarantees unknown keys
    are rejected at the boundary.
    """

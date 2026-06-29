"""Base DTO for every HTTP-boundary data object.

Subclass :class:`DTO` to declare the request/response contract for an endpoint.
``extra="forbid"`` rejects unknown keys outright; every ``datetime`` field
serializes as ISO-8601 UTC (``+00:00``), the exact format the frontend consumes,
so handlers never reformat dates themselves.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_serializer

from services.time_utils import parse_utc


class DTO(BaseModel):
    """Base for every HTTP-boundary data object: validates inbound, serializes outbound."""

    model_config = ConfigDict(extra="forbid")

    @field_serializer("*")
    @classmethod
    def _serialize_datetime(cls, value: object) -> object:
        """Emit any datetime field as ISO-8601 UTC, matching the frontend's existing wire format."""
        return parse_utc(value).isoformat() if isinstance(value, datetime) else value

"""Response DTO for the settings/personality action.

Mirrors the field shape of the DTO this action replaces (wire key ``tuple``,
Python field ``tuple_`` to avoid shadowing the builtin), so the wire format
stays identical.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .response import Response


class PersonalityResponse(Response):
    """Read/write shape: the 5-axis personality tuple and its derived voice."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    tuple_: list[int] = Field(alias="tuple", min_length=5, max_length=5)
    voice: str

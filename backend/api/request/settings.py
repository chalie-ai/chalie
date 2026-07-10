"""Inbound DTO for the settings/personality action.

The wire key is ``tuple`` (what the frontend sends); the Python field name is
``tuple_`` to avoid shadowing the builtin, mirroring the legacy DTO this
action replaces. ``StrictInt`` rejects ``bool`` — the old hand-rolled
``isinstance(v, int) and not isinstance(v, bool)`` check is now enforced
structurally by the element type.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, StrictInt

from .request import Request


class PersonalityRequest(Request):
    """Request shape: exactly five ints, each in the [-2, 2] slider range."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    tuple_: list[Annotated[StrictInt, Field(ge=-2, le=2)]] = Field(
        alias="tuple", min_length=5, max_length=5
    )

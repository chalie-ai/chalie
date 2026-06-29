"""DTOs for the personality resource — the 5-axis tuple read/write contract.

The wire key is ``tuple`` (what the frontend sends and receives); the Python
field name is ``tuple_`` to avoid shadowing the builtin. ``serialize_by_alias``
makes the foundation's ``model_dump(mode="json")`` emit the alias as the JSON key
without needing ``by_alias=True`` at the call site. ``StrictInt`` rejects
``bool`` — the old hand-rolled ``isinstance(v, int) and not isinstance(v, bool)``
check is now enforced structurally by the element type.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, StrictInt

from .base import DTO


class Personality(DTO):
    """Read/response shape: the 5-axis personality tuple and its derived voice."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    tuple_: list[int] = Field(alias="tuple", min_length=5, max_length=5)
    voice: str


class PersonalityUpdate(DTO):
    """Request shape: exactly five ints, each in the [-2, 2] slider range."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    tuple_: list[Annotated[StrictInt, Field(ge=-2, le=2)]] = Field(
        alias="tuple", min_length=5, max_length=5
    )

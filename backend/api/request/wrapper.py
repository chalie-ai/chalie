"""Inbound DTOs for the wrappers endpoint group."""

from __future__ import annotations

from pydantic import Field

from .request import Request


class WrapperCreate(Request):
    """Inbound body to mint a new wrapper token."""

    name: str = Field(..., min_length=1, max_length=200)
    metadata: dict[str, object] = {}

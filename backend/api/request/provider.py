"""Inbound DTO for the providers endpoint group.

One request DTO covers both create and update: every field is optional —
``None``/absent means "leave unchanged" on update, while the endpoint's
``post()`` enforces ``name`` and ``platform`` are required on create
(``model`` is enforced by the provider service itself, whose error message
is already allowlisted). ``timeout`` defaults to ``120`` so an omitted value
still fills in on create, while an update that omits it leaves the stored
value untouched (``exclude_unset`` tracks whether the key was present in the
request JSON, not whether its value equals the default).
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class ProviderRequest(Request):
    """Inbound body for provider create/update."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    platform: str | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    host: str | None = None
    api_key: str | None = None
    dimensions: int | None = None
    timeout: int | None = Field(default=120)
    context_window: int | None = None

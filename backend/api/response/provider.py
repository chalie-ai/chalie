"""Response DTO for the providers endpoint group.

Mirrors the field shape of the DTO this endpoint group replaced, so the wire
format stays identical. ``api_key`` is never included — the read shape never
carries the secret back to the client.
"""

from __future__ import annotations

from typing import cast

from .response import Response


class Provider(Response):
    """Read shape for one provider — api_key is never included."""

    id: int
    name: str
    platform: str
    model: str
    host: str | None
    dimensions: int | None
    timeout: int | None
    supports_vision: bool
    context_window: int | None

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "Provider":
        """Build a Provider DTO from a ProviderDbService row dict, stripping api_key."""
        return cls(
            id=cast(int, row["id"]),
            name=cast(str, row["name"]),
            platform=cast(str, row["platform"]),
            model=cast(str, row["model"]),
            host=cast("str | None", row.get("host")),
            dimensions=cast("int | None", row.get("dimensions")),
            timeout=cast("int | None", row.get("timeout")),
            supports_vision=bool(row.get("supports_vision")),
            context_window=cast("int | None", row.get("context_window")),
        )

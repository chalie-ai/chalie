"""DTO for ``GET /system/status`` — comprehensive health and diagnostics.

``status`` is ``ok`` or ``degraded`` (stays 200 even when degraded). ``memory`` and
``storage`` are stable small sub-dicts (kept as raw passthrough to preserve the
exact nested keys the tests pin); ``memory_store_error`` / ``database_error`` surface
only on degradation and are omitted otherwise.
"""

from __future__ import annotations

from .base import DTO


class SystemStatus(DTO):
    """System health: aggregate status + memory/storage sub-dicts + optional degradation errors."""

    status: str
    memory: dict[str, object]
    storage: dict[str, object]
    memory_store_error: str | None = None
    database_error: str | None = None
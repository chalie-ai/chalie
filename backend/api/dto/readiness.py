"""DTO for the readiness probe — a typed ``ready`` flag over dynamic components.

The component keys are dynamic (``database``, ``memory_store``, ``embeddings``, …)
and each carries a service-owned body (e.g. ``{status, connected}``,
``{status, message}``). They round-trip as top-level keys alongside ``ready`` via
``extra="allow"`` — the probe only types ``ready``; the components dict passes
through verbatim so whole-subobject equality (e.g. ``database == {status, connected}``)
is preserved.
"""

from __future__ import annotations

from pydantic import ConfigDict

from .base import DTO


class Readiness(DTO):
    """Readiness probe result — ``ready`` plus the raw per-component map as top-level keys."""

    model_config = ConfigDict(extra="allow")

    ready: bool
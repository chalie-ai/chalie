"""DTO for ``GET /system/context-usage`` — the composer context-window indicator.

The empty/unknown state serializes to exactly
``{"last_request_tokens": null, "context_window": null}`` (a protected test does
whole-body equality), so ``ContextUsage`` carries only these two fields.
"""

from __future__ import annotations

from .base import DTO


class ContextUsage(DTO):
    """Last user-turn request size + the selected provider's context window."""

    last_request_tokens: int | None
    context_window: int | None
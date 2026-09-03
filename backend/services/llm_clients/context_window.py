"""The one context-window default the codebase has.

``DEFAULT_WINDOW`` is a last resort, not a guess about the model: it applies only
when a provider answered successfully but volunteered no size for the configured
model. Every platform client probes its provider's own published limit first —
that measurement always wins.

A provider that could not be reached yields ``None`` instead, leaving
``providers.context_window`` null so the next send re-probes rather than pinning
a fabricated figure onto the row.
"""

from __future__ import annotations

DEFAULT_WINDOW = 128_000

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared infrastructure for delegate tools (subagent-as-tools).

A delegate tool is a standalone Ability that builds its OWN
``ProcessorConfig`` subclass inside ``run()`` and calls
``MessageProcessor.process(...).result()`` — there is no MessageProcessor
subclass, no SUBAGENT_TYPES registry, and no make_subagent_config() factory.

The framework provides one base class — :class:`DelegateAbility` — that every
delegate tool inherits. It adds no behaviour of its own; it exists so delegate
tools share a declared identity (and so the registry can treat them uniformly).
The ``delegate_result`` / ``render_trail`` helpers below stay module-level
functions each delegate calls to shape its return value.

Per-tool ``run()`` bodies are intentionally NOT templated — each delegate
builds a different ``*Config``, takes different query params, and applies
different post-processing (e.g. ``web_browse`` attaches screenshots, ``vision``
returns ``ok`` directly without going through ``delegate_result``).
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, cast

from abilities._ability import Ability, B
from abilities._result import ToolResult

if TYPE_CHECKING:
    from typing import Protocol

    class _ActTrailRenderer(Protocol):
        def _render_act_trail(self) -> str | None: ...


logger = logging.getLogger(__name__)


class DelegateAbility(Ability[B], ABC):
    """Base class for delegate tools — a declared shared identity, nothing more.
    Abstract (lists ``ABC`` directly, like the other base mix-ins) so it is never
    registered as a tool in its own right; concrete delegates fill in the
    getters. Per-tool ``run()`` bodies are left to the subclass — each builds a
    different ``*Config`` and query params — so the base is generic over the
    subclass's bag (``DelegateAbility[ItsBag]``)."""


def delegate_result(result: str, *, hint: str) -> ToolResult:
    """An empty body is NOT success — mapped to ``code=delegate-no-answer`` so a
    weak outer model self-corrects instead of trusting the silence."""
    if not result.strip():
        return ToolResult.err(
            "The delegate finished without producing an answer "
            "(it exhausted its iteration budget or was cancelled).",
            code="delegate-no-answer",
            hint=hint,
        )
    return ToolResult.ok(result)


def render_trail(mp: object) -> str:
    """Render the current act-trail for a delegate's user prompt, or '' on miss."""
    try:
        trail = cast("_ActTrailRenderer", mp)._render_act_trail()
        return trail or ""
    except Exception:
        logger.warning("[DELEGATE] act-trail render failed", exc_info=True)
        return ""

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared helpers for delegate tools (subagent-as-tools).

Spec §5b / §10f.  Each delegate tool is a standalone Ability that builds its
OWN ProcessorConfig inside run() and calls MessageProcessor.process().  There
is NO subclass, NO SUBAGENT_TYPES registry, and NO make_subagent_config()
factory — only these small shared primitives.
"""

from __future__ import annotations

import logging

from abilities._result import ToolResult

logger = logging.getLogger(__name__)


def delegate_goal(params: dict) -> str:
    """Extract the delegate's goal/query from the tool params.

    Delegates accept either ``goal`` (web_browse) or ``query`` (web_search) —
    normalise to a single string.
    """
    return params.get("goal") or params.get("query") or ""


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
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        return trail or ""
    except Exception:
        logger.warning("[DELEGATE] act-trail render failed", exc_info=True)
        return ""

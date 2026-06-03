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

# Every delegate tool name — used to build each delegate's recursion-guard
# blocklist (spec §5b property #4 / K5): a delegate never invokes another
# delegate.
DELEGATE_TOOL_NAMES: frozenset[str] = frozenset(
    {"web_search", "web_browse"}
)

# Wall-clock horizon for a delegate ACT loop (K9).
DELEGATE_DEADLINE_SECONDS: int = 600


def build_blocked(surface: list[str]) -> frozenset[str]:
    """Block every delegate tool NOT in this delegate's own surface.

    Recursion guard (spec §5b property #4 / K5).  A delegate cannot invoke
    another delegate unless it explicitly lists that delegate in its own
    always_available surface; every other delegate is blocked.
    """
    return frozenset(name for name in DELEGATE_TOOL_NAMES if name not in surface)


def delegate_goal(params: dict) -> str:
    """Extract the delegate's goal/query from the tool params.

    Delegates accept either ``goal`` (web_browse) or ``query`` (web_search) —
    normalise to a single string.
    """
    return params.get("goal") or params.get("query") or ""


def render_trail(mp: object) -> str:
    """Render the current act-trail for a delegate's user prompt, or '' on miss."""
    try:
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        return trail or ""
    except Exception:
        return ""

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

from services.processor_config import ProcessorConfig

# Wall-clock horizon for a delegate ACT loop (K9).
DELEGATE_DEADLINE_SECONDS: int = 600


def policy_channel_for(channel: str) -> "ProcessorConfig.POLICY_CHANNEL":
    """Map a caller's transcript channel → the policy channel a delegate inherits.

    A delegate's internal tool calls are gated under the SAME policy channel as
    the caller that invoked the delegate tool, rather than a hardcoded value.
    The map is total: the user channel → CHAT, an external-agent channel →
    EXTERNAL_AGENT, every background channel → SUBCONSCIOUS.  ``channel`` is the
    caller's ``config.channel`` (set by ``Ability.execute``) and is always
    present in a real dispatch; an empty/unknown string is only reachable by a
    direct non-dispatch call and falls into the SUBCONSCIOUS branch.
    """
    pc = ProcessorConfig.POLICY_CHANNEL
    if channel == "user":
        return pc.CHAT
    if channel.startswith("external-agent:"):
        return pc.EXTERNAL_AGENT
    return pc.SUBCONSCIOUS


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

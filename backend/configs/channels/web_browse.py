# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseConfig — the delegate channel for the ``web_browse`` tool.

The typed ``ProcessorConfig`` for the interactive web-browsing delegate (spec
§5b / §10f). Paired with ``WebBrowseAbility`` in ``abilities/web_browse.py``,
whose ``run()`` instantiates this config and calls ``MessageProcessor.process()``.

Drives the raw ``browser`` tool (render / screenshot / interact / monitor) plus
``read`` in a clean-context loop. ``policy_channel`` is inherited from the caller
that invoked the tool; the user-facing permission check happens at the outer
``web_browse`` tool.
"""

from __future__ import annotations

from abilities._delegate import render_trail
from services.processor_config import ProcessorConfig

_WEB_BROWSE_SYSTEM_PROMPT = (
    "You are a focused web-browsing agent. You receive a single goal and pursue "
    "it by driving a real browser: render JavaScript-heavy pages, take "
    "screenshots, fill forms, click buttons, navigate multi-step flows, and "
    "read what you find.\n\n"
    "Work step by step: open the page, observe its actual state, act, then "
    "re-observe before acting again. Ground every claim in what the page "
    "actually shows — do not invent content, URLs, or results. If the goal "
    "cannot be completed in the browser, say so plainly and explain why.\n\n"
    "Return a clear answer that directly addresses the goal, citing the pages "
    "you actually visited. You have no conversation history and no user "
    "personality — work only from the goal you were given."
)

_WEB_BROWSE_TOOLS: tuple[str, ...] = ("browser", "read")


class WebBrowseConfig(ProcessorConfig):
    """ProcessorConfig for the web_browse delegate.

    Mirrors the TKT-803 ProcessorConfig subclasses: a typed ``__init__`` that
    calls ``super().__init__(...)`` against the frozen base.  ``policy_channel``
    is supplied by the caller (inherited from whoever invoked the tool) rather
    than hardcoded.
    """

    def __init__(self, policy_channel: "ProcessorConfig.POLICY_CHANNEL") -> None:
        tools = list(_WEB_BROWSE_TOOLS)
        super().__init__(
            channel="delegate:web_browse",
            role="web_browse",
            policy_channel=policy_channel,
            always_available=[*tools, "memory"],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=50,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        """Goal-driven user prompt: the goal plus the act-trail so far."""
        parts = [f"Browsing goal:\n{mp._raw_input}"]  # type: ignore[attr-defined]
        trail = render_trail(mp)
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def get_system_prompt(self, mp) -> str:
        return _WEB_BROWSE_SYSTEM_PROMPT

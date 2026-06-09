# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchConfig — the delegate channel for the ``web_search`` tool.

The typed ``ProcessorConfig`` for the web-research delegate (spec §5b / §10f,
TKT-732). Paired with ``WebSearchAbility`` in ``abilities/web_search.py``, whose
``run()`` instantiates this config and calls ``MessageProcessor.process()``.

Clean *cross-turn* context (``suppress_history=True``) but a real per-turn
transcript row on its own ``delegate:web_search`` channel, so the delegate can
render its own act-trail across ACT iterations. Without that row the turn uid is
never assigned and ``_render_act_trail`` returns "" — the loop then re-searches
blind to its own results until it exhausts ``max_iterations`` (TKT-881).
``skip_input_row`` (HiddenInput) is deliberately *not* set: it is the async-return
mechanism (``deliver_async_result`` / ``with_hidden_input``), not a delegate
property. A goal-driven system prompt and a finite tool surface
(search/read/web_download + memory, discoverable=[]) keep the delegate from ever
spawning another delegate. ``policy_channel`` is inherited from the caller that
invoked the tool.
"""

from __future__ import annotations

from abilities._delegate import render_trail
from services.processor_config import ProcessorConfig

_WEB_SEARCH_SYSTEM_PROMPT = (
    "You are a focused web-research agent. You receive a single research query "
    "and answer it by searching the web and reading the most relevant sources.\n\n"
    "Loop: search → read the best results → search again to fill gaps → "
    "synthesise. Cite the sources you actually read. Do not fabricate URLs, "
    "quotes, or facts. If the web yields nothing useful, say so honestly.\n\n"
    "Return a concise, well-grounded synthesis that directly answers the query. "
    "You have no conversation history and no user personality — work only from "
    "the query you were given."
)

_WEB_SEARCH_TOOLS: tuple[str, ...] = ("search", "read", "web_download")


class WebSearchConfig(ProcessorConfig):
    """ProcessorConfig for the web_search delegate.

    Mirrors the TKT-803 ProcessorConfig subclasses: a typed ``__init__`` that
    calls ``super().__init__(...)`` against the frozen base.  ``policy_channel``
    is supplied by the caller (inherited from whoever invoked the tool) rather
    than hardcoded.
    """

    def __init__(self, policy_channel: "ProcessorConfig.POLICY_CHANNEL") -> None:
        tools = list(_WEB_SEARCH_TOOLS)
        super().__init__(
            channel="delegate:web_search",
            role="web_search",
            policy_channel=policy_channel,
            always_available=[*tools, "memory"],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=50,
            skip_transcript=False,  # write a delegate-channel transcript row so
            skip_input_row=False,   # _setup assigns the uid the act-trail needs
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        """Goal-driven user prompt: the raw query plus the act-trail so far."""
        parts = [f"Research query:\n{mp._raw_input}"]  # type: ignore[attr-defined]
        trail = render_trail(mp)
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def get_system_prompt(self, mp) -> str:
        return _WEB_SEARCH_SYSTEM_PROMPT

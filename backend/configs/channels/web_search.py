# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchConfig — delegate channel for the ``web_search`` tool.

Writes a real per-turn transcript row on its own ``delegate:web_search``
channel so the delegate can render its own act-trail across ACT iterations.
Without that row the turn uid is never assigned and ``_render_act_trail``
returns "" — the loop would re-search blind to its own results with no way to
converge. ``skip_input_row`` (HiddenInput) is
deliberately *not* set: it is the async-return mechanism
(``deliver_async_result`` / ``with_hidden_input``), not a delegate property.
Paired with ``WebSearchAbility`` (abilities/web_search.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from configs.enums.channels import Channel
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from configs.enums.policy_channel import PolicyChannel

_WEB_SEARCH_TOOLS: tuple[str, ...] = ("search", "news", "read", "web_download")


class WebSearchConfig(ProcessorConfig):
    """``policy_channel`` is supplied by the caller (inherited from whoever
    invoked the tool) rather than hardcoded."""

    uses_delegate_provider: ClassVar[bool] = True

    # Pin thinking to LOW (the floor — "no thinking flag" at the provider) so a
    # user's persisted high/medium override never leaks a thinking turn into the
    # focused search loop. resolve_thinking_mode() gives this config pin priority
    # over the override and the gate-computed level.
    thinking_mode: ClassVar[str] = "low"

    def __init__(self, policy_channel: "PolicyChannel") -> None:
        tools = list(_WEB_SEARCH_TOOLS)
        super().__init__(
            channel=Channel.DELEGATE_WEB_SEARCH.value,
            role="web_search",
            policy_channel=policy_channel,
            always_available=[*tools, "memory"],
            skip_transcript=False,  # write a delegate-channel transcript row so
            skip_input_row=False,   # _setup assigns the uid the act-trail needs
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """You are a focused web-research agent. You receive one research query and answer it from the web.

Be efficient. Run ONE search, then work from the result snippets — they usually already answer the query. Read a full page only when the snippets are genuinely insufficient for a specific missing detail, and read at most the one or two most relevant pages. Do not re-search to fill gaps; synthesise from what the first search and any targeted reads gave you.

Cite the sources you actually used. Never fabricate URLs, quotes, or facts. If the web yields nothing useful, say so plainly.

Return a concise synthesis that directly answers the query. You have no conversation history and no user personality — work only from the query."""

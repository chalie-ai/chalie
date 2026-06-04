# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchAbility — delegate a web-research task to a focused search agent.

The foundational delegate-tool template (spec §5b / §10f, TKT-732).  A delegate
tool is a standalone Ability that pairs with a typed ``ProcessorConfig``
subclass (``WebSearchConfig``).  ``run()`` instantiates the subclass and calls
``MessageProcessor.process()`` — there is no MessageProcessor subclass, no
SUBAGENT_TYPES registry, and no make_subagent_config() factory.

Properties (spec §5b "Properties of every delegate tool"):
  - Clean context — skip_transcript / skip_input_row / suppress_history all True,
    no personality, no history, no world state.
  - Goal-driven system prompt — short, task-specific.
  - Finite tool surface — always_available lists exactly what the delegate needs;
    discoverable=[] prevents discovery of anything else.
  - No recursion — with no find_tools and discoverable=[], a delegate can only
    call what is in always_available; delegate tools are not in that surface, so
    a delegate can never spawn another delegate.
  - ASYNC_CAPABLE=True — the framework (Ability.execute) wraps run() in a
    daemon thread for async-capable origins.  run() is ALWAYS synchronous.

Permission boundary — ``policy_channel`` is inherited from the caller that
invoked the ``web_search`` tool (``self.MessageProcessor.config.policy_channel``):
the delegate's internal tool calls are gated under the SAME policy channel as
the caller, not a hardcoded value.  The user-facing permission check still
happens at the outer ``web_search`` tool.
"""

from typing import ClassVar

from abilities._base import Ability
from abilities._delegate import (
    delegate_goal,
    render_trail,
)
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
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
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


class WebSearchAbility(Ability):
    NAME = "web_search"
    SEARCH_TOOLTIP = "delegate a focused web search"
    ASYNC_CAPABLE = True
    SUMMARY = (
        "Delegate a web-research task to a focused agent that searches the web, "
        "reads the best sources, and returns a grounded synthesis with citations."
    )
    EXAMPLES = [
        "search the web for the latest news on AI regulation",
        "find out what the current consensus is on intermittent fasting",
        "research the best practices for Postgres connection pooling",
        "look up recent reviews of the new Framework laptop",
        "what are the latest developments in fusion energy",
        "find current pricing for AWS Lambda vs Cloudflare Workers",
        "search for documentation on the Stripe webhooks API",
        "what happened in the latest SpaceX Starship test flight",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to research on the web.",
            },
        },
        "required": ["query"],
    }

    def run(self, params: dict) -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        result = MessageProcessor.process(
            delegate_goal(params),
            WebSearchConfig(self.MessageProcessor.config.policy_channel),
        )
        return {"status": "success", "result": result}

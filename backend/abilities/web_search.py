# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchAbility — delegate a web-research task to a focused search agent.

The foundational delegate-tool template (spec §5b / §10f, TKT-732).  A delegate
tool is a standalone Ability that builds its OWN ProcessorConfig inside run()
and calls the normal MessageProcessor.process().  There is no subclass, no
SUBAGENT_TYPES registry, and no make_subagent_config() factory.

Properties (spec §5b "Properties of every delegate tool"):
  - Clean context — skip_transcript / skip_input_row / suppress_history all True,
    no personality, no history, no world state.
  - Goal-driven system prompt — short, task-specific.
  - Finite tool surface — always_available lists exactly what the delegate needs;
    discoverable=[] prevents discovery of anything else.
  - No recursion — delegate tools are not in the surface and are blocked.
  - ASYNC_CAPABLE=True — the framework (Ability.dispatch) wraps run() in a
    daemon thread for async-capable origins.  run() is ALWAYS synchronous.
"""

import time
from typing import ClassVar

from abilities._base import Ability
from abilities._delegate import (
    DELEGATE_DEADLINE_SECONDS,
    build_blocked,
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


def _web_search_user_prompt(mp: object) -> str:
    """Goal-driven user prompt: the raw query plus the act-trail so far."""
    parts = [f"Research query:\n{mp._raw_input}"]  # type: ignore[attr-defined]
    trail = render_trail(mp)
    if trail:
        parts.append(trail)
    return "\n\n".join(parts)


class WebSearchAbility(Ability):
    NAME = "web_search"
    SEARCH_TOOLTIP = "delegate a focused web search"
    POLICY_CATEGORY = "Delegate Agents"
    POLICY_LABELS: ClassVar[dict[str, str]] = {"": "Web search agent"}
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
    TIMEOUT = DELEGATE_DEADLINE_SECONDS

    _TOOLS: ClassVar[list[str]] = ["search", "read"]

    def run(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = ProcessorConfig(
            channel=f"delegate:{self.NAME}",
            role=self.NAME,
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            build_user_prompt=_web_search_user_prompt,
            build_user_definition=lambda _mp: "",
            build_system_prompt=lambda _mp: _WEB_SEARCH_SYSTEM_PROMPT,
            always_available=self._TOOLS,
            discoverable=[],
            blocked=build_blocked(self._TOOLS),
            max_iterations=50,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        result = MessageProcessor.process(
            delegate_goal(params),
            config,
            deadline=time.time() + DELEGATE_DEADLINE_SECONDS,
        )
        return {"status": "success", "result": result}

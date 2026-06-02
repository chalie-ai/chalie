# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ResearchAbility — delegate a general research task to a focused agent.

Replaces the former ``general_purpose`` subagent type (spec §5b mapping).  A
standalone delegate tool: builds its own ProcessorConfig and calls the normal
MessageProcessor.process().  No subclass, no SUBAGENT_TYPES, no factory.
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

_RESEARCH_SYSTEM_PROMPT = (
    "You are a focused research agent. You receive a single goal and pursue it "
    "to completion using memory recall, web search, reading sources, and "
    "find_tools to discover any additional capability you need.\n\n"
    "Work methodically: recall what is already known, search and read to fill "
    "gaps, then synthesise. Ground every claim in what you actually found — do "
    "not fabricate. If the goal cannot be met, say so plainly and explain why.\n\n"
    "Return a clear, well-organised answer that fully addresses the goal. You "
    "have no conversation history and no user personality — work only from the "
    "goal you were given."
)


def _research_user_prompt(mp: object) -> str:
    """Goal-driven user prompt: the goal plus the act-trail so far."""
    parts = [f"Research goal:\n{mp._raw_input}"]  # type: ignore[attr-defined]
    trail = render_trail(mp)
    if trail:
        parts.append(trail)
    return "\n\n".join(parts)


class ResearchAbility(Ability):
    NAME = "research"
    SEARCH_TOOLTIP = "delegate a research task"
    POLICY_CATEGORY = "Delegate Agents"
    POLICY_LABELS: ClassVar[dict[str, str]] = {"": "Research agent"}
    ASYNC_CAPABLE = True
    SUMMARY = (
        "Delegate a research task to a focused agent that recalls memory, "
        "searches, reads, and synthesises a grounded answer to a goal."
    )
    EXAMPLES = [
        "research current AI regulations in the EU",
        "find out everything we know about the user's travel plans",
        "investigate the best approach for migrating from MySQL to Postgres",
        "research the trade-offs between gRPC and REST for our use case",
        "dig into what caused the outage last Tuesday",
        "research how competitors price their enterprise tier",
        "gather background on the history of the Rust borrow checker",
        "look into whether we can self-host the embedding model",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What to research.",
            },
        },
        "required": ["goal"],
    }
    TIMEOUT = DELEGATE_DEADLINE_SECONDS

    _TOOLS: ClassVar[list[str]] = ["memory", "search", "read", "find_tools"]

    def run(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = ProcessorConfig(
            channel=f"delegate:{self.NAME}",
            role=self.NAME,
            usage_class="subconscious",
            build_user_prompt=_research_user_prompt,
            build_user_definition=lambda _mp: "",
            build_system_prompt=lambda _mp: _RESEARCH_SYSTEM_PROMPT,
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

    def execute(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        """Delegate tools implement run() directly; execute() forwards to it."""
        return self.run(channel, params, telemetry)

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SummariserAbility — delegate a summarisation task to a focused agent.

Replaces the former ``summariser`` subagent type (spec §5b mapping).  Per K4,
this delegate gets ``web_search`` (a delegate it may delegate further to), NOT
raw ``search`` — delegates can delegate one further step.

Standalone delegate tool: builds its own ProcessorConfig and calls the normal
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

_SUMMARISER_SYSTEM_PROMPT = (
    "You are a focused summarisation agent. You receive a single goal "
    "describing what to summarise and produce a faithful, well-structured "
    "summary.\n\n"
    "Read the relevant documents and sources. When you need fresh external "
    "context, delegate it with web_search rather than searching raw yourself. "
    "Stay faithful to the source material — do not invent facts or add opinions "
    "that are not supported by what you read.\n\n"
    "Return a clear summary that captures the key points at the requested level "
    "of detail. You have no conversation history and no user personality — work "
    "only from the goal you were given."
)


def _summariser_user_prompt(mp: object) -> str:
    """Goal-driven user prompt: the goal plus the act-trail so far."""
    parts = [f"Summarisation goal:\n{mp._raw_input}"]  # type: ignore[attr-defined]
    trail = render_trail(mp)
    if trail:
        parts.append(trail)
    return "\n\n".join(parts)


class SummariserAbility(Ability):
    NAME = "summariser"
    SEARCH_TOOLTIP = "delegate a summarisation task"
    POLICY_CATEGORY = "Delegate Agents"
    POLICY_LABELS: ClassVar[dict[str, str]] = {"": "Summariser agent"}
    ASYNC_CAPABLE = True
    SUMMARY = (
        "Delegate a summarisation task to a focused agent that reads documents "
        "and sources and produces a faithful, well-structured summary."
    )
    EXAMPLES = [
        "summarise the attached report into three bullet points",
        "give me a one-paragraph summary of this document",
        "condense these meeting notes into action items",
        "summarise the key findings from the research paper",
        "produce an executive summary of the quarterly review",
        "summarise this long email thread for me",
        "boil down the documentation into a quick-start guide",
        "summarise the main arguments in this article",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What to summarise (and at what level of detail).",
            },
        },
        "required": ["goal"],
    }
    TIMEOUT = DELEGATE_DEADLINE_SECONDS

    _TOOLS: ClassVar[list[str]] = ["read", "document", "web_search"]

    def run(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = ProcessorConfig(
            channel=f"delegate:{self.NAME}",
            role=self.NAME,
            usage_class="subconscious",
            build_user_prompt=_summariser_user_prompt,
            build_user_definition=lambda _mp: "",
            build_system_prompt=lambda _mp: _SUMMARISER_SYSTEM_PROMPT,
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

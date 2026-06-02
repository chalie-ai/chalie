# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseAbility — delegate an interactive web-browsing task to a focused agent.

The fourth delegate tool (spec §5b / §10f).  Replaces the browsing role the
former ``web_surfer`` subagent performed, scoped down to a clean-context agent
that drives the raw ``browser`` tool (render / screenshot / interact / monitor)
and reads what it finds.

Named ``web_browse`` (not ``browser``) to avoid a flat-registry collision with
the raw ``browser`` ability — its tool *surface* still uses the raw ``browser``
tool (spec amendment 2026-06-02, Dylan, AC-3).

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


def _web_browse_user_prompt(mp: object) -> str:
    """Goal-driven user prompt: the goal plus the act-trail so far."""
    parts = [f"Browsing goal:\n{mp._raw_input}"]  # type: ignore[attr-defined]
    trail = render_trail(mp)
    if trail:
        parts.append(trail)
    return "\n\n".join(parts)


class WebBrowseAbility(Ability):
    NAME = "web_browse"
    SEARCH_TOOLTIP = "delegate an interactive web-browsing task"
    POLICY_CATEGORY = "Delegate Agents"
    POLICY_LABELS: ClassVar[dict[str, str]] = {"": "Web browse agent"}
    ASYNC_CAPABLE = True
    SUMMARY = (
        "Delegate an interactive web-browsing task to a focused agent that "
        "drives a real browser — rendering pages, taking screenshots, filling "
        "forms, and navigating flows — and reports what it finds."
    )
    EXAMPLES = [
        "browse this site and book the earliest available appointment",
        "log into my account on this page and read my current balance",
        "navigate this multi-step checkout and tell me the final total",
        "open the dashboard and extract the table after it finishes loading",
        "click through this cookie banner and read the article behind it",
        "fill in this web form with my details and submit it",
        "step through this site's signup flow and report where it fails",
        "open this JavaScript app and tell me what the page shows",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What to accomplish in the browser.",
            },
        },
        "required": ["goal"],
    }
    TIMEOUT = DELEGATE_DEADLINE_SECONDS

    _TOOLS: ClassVar[list[str]] = ["browser", "read"]

    def run(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = ProcessorConfig(
            channel=f"delegate:{self.NAME}",
            role=self.NAME,
            usage_class="subconscious",
            build_user_prompt=_web_browse_user_prompt,
            build_user_definition=lambda _mp: "",
            build_system_prompt=lambda _mp: _WEB_BROWSE_SYSTEM_PROMPT,
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

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchAbility — delegate a web-research task to a focused search agent.

The foundational delegate-tool template.  A delegate
tool is a standalone Ability that pairs with a typed ``ProcessorConfig``
subclass (``WebSearchConfig``, in ``configs/channels/web_search.py`` with the
other channel configs).  ``run()`` instantiates the subclass and calls
``MessageProcessor.process()`` — there is no MessageProcessor subclass, no
SUBAGENT_TYPES registry, and no make_subagent_config() factory.

Properties of every delegate tool:
  - Clean context — skip_transcript / skip_input_row / suppress_history all True,
    no personality, no history, no world state.
  - Goal-driven system prompt — short, task-specific.
  - Finite tool surface — always_available lists exactly what the delegate needs;
    it does NOT pin find_tools, so the delegate cannot discover anything else.
  - No recursion — with no find_tools pinned, a delegate can only call what is in
    always_available; delegate tools are not in that surface, so a delegate can
    never spawn another delegate.
  - Per-call async — the model may pass ``async: true`` (exposed only on
    SUPPORTS_ASYNC channels) to run the search in the background and receive the
    result as a later turn; the framework (Ability.execute) wraps run() in a
    daemon thread when it does.  run() is ALWAYS synchronous in itself.

Permission boundary — ``policy_channel`` is inherited from the caller that
invoked the ``web_search`` tool (``self.mp.config.policy_channel``):
the delegate's internal tool calls are gated under the SAME policy channel as
the caller, not a hardcoded value.  The user-facing permission check still
happens at the outer ``web_search`` tool.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._delegate import DelegateAbility, delegate_result
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from configs.channels.web_search import WebSearchConfig
from contracts.params.param_bag import ParamBag
from contracts.params.delegate_params_bag import DelegateParamsBag


class WebSearchAbility(DelegateAbility[DelegateParamsBag]):
    def get_name(self) -> str:
        return "web_search"

    def get_summary(self) -> str:
        return (
            "Research something online using various search engines. A focused agent "
            "runs the searches, reads the best sources, and returns a grounded "
            "synthesis with citations. Use for looking things up — not for "
            "interacting with a specific page."
        )

    def get_examples(self) -> list[str]:
        return [
            "search the web for the latest news on AI regulation",
            "find out what the current consensus is on intermittent fasting",
            "research the best practices for Postgres connection pooling",
            "look up recent reviews of the new Framework laptop",
            "what are the latest developments in fusion energy",
            "find current pricing for AWS Lambda vs Cloudflare Workers",
            "search for documentation on the Stripe webhooks API",
            "what happened in the latest SpaceX Starship test flight",
        ]

    def get_search_tooltip(self) -> str:
        return "delegate a focused web search"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.instructions: {
                "type": "string",
                "description": "What to research on the web.",
            },
        },
        "required": [Keys.instructions],
    }

    # An action-less delegate: the dispatcher's ACTION_REQUIRED pre-gate (the
    # ``""`` key covers action-less tools) rejects missing/empty ``instructions``
    # with ``code=missing-params`` BEFORE the policy gate and BEFORE run() — so an
    # empty instruction never spawns an expensive delegate on nothing.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.instructions,)}

    # The typed input contract: the dispatch seam builds the shared delegate
    # bag via DelegateParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = DelegateParamsBag

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: DelegateParamsBag) -> ToolResult:
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError("web_search.run() dispatched without a bound MessageProcessor")

        result = MessageProcessor.process(
            WebSearchConfig(mp.config.policy_channel),
            raw_input=params.instructions,
        ).result()
        return delegate_result(
            result, hint="Narrow the query or split it into smaller searches, then retry."
        )

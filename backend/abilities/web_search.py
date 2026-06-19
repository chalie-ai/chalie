# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebSearchAbility — delegate a web-research task to a focused search agent.

The foundational delegate-tool template (spec §5b / §10f).  A delegate
tool is a standalone Ability that pairs with a typed ``ProcessorConfig``
subclass (``WebSearchConfig``, in ``configs/channels/web_search.py`` with the
other channel configs).  ``run()`` instantiates the subclass and calls
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

from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from typing import Protocol
    from services.processor_config import ProcessorConfig

    class _MpWithConfig(Protocol):
        config: "ProcessorConfig"

from abilities._ability import Ability
from abilities._delegate import delegate_goal, delegate_result
from abilities._params import Keys
from abilities._result import ToolResult
from configs.channels.web_search import WebSearchConfig


class WebSearchAbility(Ability):
    def get_name(self) -> str:
        return "web_search"

    def get_summary(self) -> str:
        return (
            "Delegate a web-research task to a focused agent that searches the web, "
            "reads the best sources, and returns a grounded synthesis with citations."
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
            Keys.query: {
                "type": "string",
                "description": "What to research on the web.",
            },
        },
        "required": [Keys.query],
    }

    # An action-less delegate: the dispatcher's ACTION_REQUIRED pre-gate (the
    # ``""`` key covers action-less tools) rejects a missing/empty ``query`` with
    # ``code=missing-params`` BEFORE the policy gate and BEFORE run() — so an empty
    # query never spawns an expensive delegate on an empty goal.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        result = MessageProcessor.process(
            delegate_goal(params),
            WebSearchConfig(cast("_MpWithConfig", self.mp).config.policy_channel),
        )
        return delegate_result(
            result, hint="Narrow the query or split it into smaller searches, then retry."
        )

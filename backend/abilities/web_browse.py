# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseAbility — delegate an interactive web-browsing task to a focused agent.

Pairs with a typed ``ProcessorConfig`` subclass (``WebBrowseConfig``, in
``configs/channels/web_browse.py`` with the other channel configs).  Spec §5b /
§10f.  Replaces the browsing role the former ``web_surfer`` subagent performed,
scoped down to a clean-context agent that drives the raw ``browser`` tool
(open / read / find / click / fill / select / scroll / back / screenshot) and reads what it finds.

Named ``web_browse`` (not ``browser``) to avoid a flat-registry collision with
the raw ``browser`` ability — its tool *surface* still uses the raw ``browser``
tool (spec amendment 2026-06-02, Dylan, AC-3).

Permission boundary — ``policy_channel`` is inherited from the caller that
invoked the ``web_browse`` tool (``self.mp.config.policy_channel``):
the delegate's internal tool calls are gated under the SAME policy channel as
the caller, not a hardcoded value.  The user-facing permission check still
happens at the outer ``web_browse`` tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._delegate import delegate_result
from abilities._params import Keys
from abilities._result import ToolResult
from configs.channels.web_browse import WebBrowseConfig

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig


class WebBrowseAbility(Ability):
    def get_name(self) -> str:
        return "web_browse"

    def get_summary(self) -> str:
        return (
            "Spawn a subagent with full web browser control to perform an action on "
            "one or more websites. It drives a real browser — rendering pages, "
            "filling forms, navigating multi-step flows — and inspects screenshots "
            "with its own vision. Screenshots are saved as documents whose doc_id "
            "any vision tool can view later. Use for acting on a specific site — "
            "not for general lookups."
        )

    def get_examples(self) -> list[str]:
        return [
            "browse this site and book the earliest available appointment",
            "log into my account on this page and read my current balance",
            "navigate this multi-step checkout and tell me the final total",
            "open the dashboard and extract the table after it finishes loading",
            "click through this cookie banner and read the article behind it",
            "fill in this web form with my details and submit it",
            "step through this site's signup flow and report where it fails",
            "take a screenshot of this page and tell me what it shows",
        ]

    def get_search_tooltip(self) -> str:
        return "delegate an interactive web-browsing task"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.goal: {
                "type": "string",
                "description": (
                    "What to accomplish in the browser. Include everything the "
                    "answer needs — e.g. 'take a screenshot of the page and "
                    "describe what it shows', not just 'screenshot the page'."
                ),
            },
        },
        "required": [Keys.goal],
    }

    # An action-less delegate: the dispatcher's ACTION_REQUIRED pre-gate (the
    # ``""`` key covers action-less tools) rejects a missing/empty ``goal`` with
    # ``code=missing-params`` BEFORE the policy gate and BEFORE run() — so an empty
    # goal never spawns an expensive browser delegate on nothing.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.goal,)}

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        cfg = WebBrowseConfig(cast("ProcessorConfig", getattr(self.mp, "config", None)).policy_channel)
        result = MessageProcessor.process(cast(str, self.param(params, Keys.goal, required=True)), cfg)
        tr = delegate_result(
            result, hint="Restate the goal more concretely or break it into steps, then retry."
        )
        return self._with_screenshots(tr, cfg.final_screenshots())

    @staticmethod
    def _with_screenshots(tr: ToolResult, shots: list[tuple[str, str]]) -> ToolResult:
        """Mechanical, not prompt-dependent: the caller receives every doc_id
        even when the delegate forgets to mention its screenshots."""
        if tr.status != "success" or not shots:
            return tr
        lines = "\n".join(f"- doc_id={doc_id} ({url})" for doc_id, url in shots)
        return ToolResult.ok(
            f"{tr.body}\n\nScreenshots saved as documents "
            f"(view one with vision(image=<doc_id>)):\n{lines}",
            screenshots=len(shots),
        )

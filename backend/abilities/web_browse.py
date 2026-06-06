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
(render / screenshot / interact / monitor) and reads what it finds.

Named ``web_browse`` (not ``browser``) to avoid a flat-registry collision with
the raw ``browser`` ability — its tool *surface* still uses the raw ``browser``
tool (spec amendment 2026-06-02, Dylan, AC-3).

Permission boundary — ``policy_channel`` is inherited from the caller that
invoked the ``web_browse`` tool (``self.mp.config.policy_channel``):
the delegate's internal tool calls are gated under the SAME policy channel as
the caller, not a hardcoded value.  The user-facing permission check still
happens at the outer ``web_browse`` tool.
"""

from typing import ClassVar

from abilities._ability import Ability
from abilities._delegate import delegate_goal
from configs.channels.web_browse import WebBrowseConfig


class WebBrowseAbility(Ability):
    def get_name(self) -> str:
        return "web_browse"

    def get_summary(self) -> str:
        return (
            "Delegate an interactive web-browsing task to a focused agent that "
            "drives a real browser — rendering pages, taking screenshots, filling "
            "forms, and navigating flows — and reports what it finds."
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
            "open this JavaScript app and tell me what the page shows",
        ]

    def get_search_tooltip(self) -> str:
        return "delegate an interactive web-browsing task"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What to accomplish in the browser.",
            },
        },
        "required": ["goal"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        result = MessageProcessor.process(
            delegate_goal(params),
            WebBrowseConfig(self.mp.config.policy_channel),
        )
        return {"status": "success", "result": result}

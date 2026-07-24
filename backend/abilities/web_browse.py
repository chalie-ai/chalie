# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseAbility — delegate an interactive web-browsing task to a focused agent.

Pairs with a typed ``ProcessorConfig`` subclass (``WebBrowseConfig``, in
``configs/channels/web_browse.py`` with the other channel configs).  Replaces the browsing role the former ``web_surfer`` subagent performed,
scoped down to a clean-context agent that drives the raw ``browser`` tool
(open / read / find / click / fill / select / scroll / back / screenshot) and reads what it finds.

Named ``web_browse`` (not ``browser``) to avoid a flat-registry collision with
the raw ``browser`` ability — its tool *surface* still uses the raw ``browser``
tool.

Permission boundary — ``policy_channel`` is inherited from the caller that
invoked the ``web_browse`` tool (``self.mp.config.policy_channel``):
the delegate's internal tool calls are gated under the SAME policy channel as
the caller, not a hardcoded value.  The user-facing permission check still
happens at the outer ``web_browse`` tool.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._delegate import DelegateAbility, delegate_result
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from configs.channels.web_browse import WebBrowseConfig
from contracts.params.param_bag import ParamBag
from contracts.params.delegate_params_bag import DelegateParamsBag


class WebBrowseAbility(DelegateAbility[DelegateParamsBag]):
    def get_name(self) -> str:
        return "web_browse"

    def get_summary(self) -> str:
        return (
            "Spawn a subagent with full web browser control to perform an action on "
            "one or more websites. It drives a real browser — rendering pages, "
            "filling forms, navigating multi-step flows — and inspects screenshots "
            "with its own vision. Screenshots are saved as documents whose doc_id "
            "any vision tool can view later. Use for acting on a specific site — "
            "not for general lookups. Give the subagent one narrow, concrete goal; "
            "fire several in parallel for different tasks on the same site rather "
            "than handing one subagent a broad, generic goal."
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

    def get_follow_up(self, tr: ToolResult) -> str:
        """Pull the full text of a page the agent reached.

        Screenshots already ride back described (see :meth:`_with_screenshots`),
        so the only standing next step is reading a page's full text."""
        return "To get a full textual version of this page call the `read` tool."

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.instructions: {
                "type": "string",
                "description": (
                    "What to accomplish in the browser. Include everything the "
                    "answer needs — e.g. 'take a screenshot of the page and "
                    "describe what it shows', not just 'screenshot the page'."
                ),
            },
        },
        "required": [Keys.instructions],
    }

    # An action-less delegate: the dispatcher's ACTION_REQUIRED pre-gate (the
    # ``""`` key covers action-less tools) rejects missing/empty ``instructions``
    # with ``code=missing-params`` BEFORE the policy gate and BEFORE run() — so an
    # empty instruction never spawns an expensive browser delegate on nothing.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.instructions,)}

    # The typed input contract: the dispatch seam builds the shared delegate
    # bag via DelegateParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = DelegateParamsBag

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: DelegateParamsBag) -> ToolResult:
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415
        from tools.browser.session import screenshot_ledger  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError("web_browse.run() dispatched without a bound MessageProcessor")

        cfg = WebBrowseConfig(mp.config.policy_channel)
        delegate_mp = MessageProcessor.process(cfg, raw_input=params.instructions)
        result = delegate_mp.result()
        tr = delegate_result(
            result, hint="Restate the goal more concretely or break it into steps, then retry."
        )
        # Screenshots ride back off the sub-turn's uid-keyed ledger (the frozen
        # config no longer carries per-run state, §2.5) — the same key
        # browser._session_key() records under and PromptService reads.
        return self._with_screenshots(tr, screenshot_ledger(delegate_mp.uid or 0))

    @staticmethod
    def _with_screenshots(tr: ToolResult, shots: list[tuple[str, str]]) -> ToolResult:
        """Mechanical, not prompt-dependent: each screenshot rides back already
        DESCRIBED. Ingest ran every capture through vision (browser._screenshot →
        document pipeline), storing the description as the document's clean_text;
        we surface it inline here — exactly as the raw browser tool hands its own
        caller a described page — so the model sees what each page shows without a
        separate vision call, even when the delegate forgot to mention a shot."""
        if tr.status != "success" or not shots:
            return tr
        from services.document_service import DocumentService  # noqa: PLC0415

        doc_svc = DocumentService()
        blocks = []
        for doc_id, url in shots:
            doc = doc_svc.get_document(doc_id)
            vision = (doc.get("clean_text") if doc else "") or "(description unavailable)"
            blocks.append(f"- {url} (doc_id={doc_id}):\n{vision}")
        joined = "\n\n".join(blocks)
        return ToolResult.ok(
            f"{tr.body}\n\nScreenshots captured this run (each already described):\n{joined}",
            screenshots=len(shots),
        )

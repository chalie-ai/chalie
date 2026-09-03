"""Recall — search existing memory before writing (Memory v3).

Fuses Graph FTS (subject) + Map vector (situational cues, most-consolidated
first — a memory that absorbed others outranks any of them). Both
halves render in one shape — ``{"source": …, "memory": …}`` — where a fact's
source is its subject and an episode's is the origin it was distilled from.
``cues`` are the search key only: they are never fetched into this render path,
so no model ever reads back the tags it wrote. The one memory surface the
in-conversation model may use, and the memory step's recall-first step. Never
writes. A query longer than 500 characters after stop-word removal is refused
with an ``invalid-param`` error — a loud rejection, never a silent trim — so
the model asks a more fine-grained question.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.recall_params_bag import RecallParamsBag
from services.memory_recall_service import MemoryRecallService


class Recall(Ability[RecallParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "recall"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("query",)}
    PARAMS: ClassVar[type[ParamBag] | None] = RecallParamsBag

    def get_summary(self) -> str:
        return (
            "Search existing memory before deciding to store/update/augment/"
            "forget. Returns a list of {\"source\", \"memory\"} — the origin of "
            "each memory and the memory itself. Episodes are matched on the "
            "SITUATION they are about, so describe the situation, not the wording "
            "you expect. Always recall first; never store blindly. The query is "
            "capped at 500 characters after stop-word removal — a longer query "
            "is rejected, so ask a more fine-grained question."
        )

    def get_examples(self) -> list[str]:
        return ["query='where does the user live'", "query='going shopping later'"]

    def get_search_tooltip(self) -> str:
        return "search the memory graph and map"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: RecallParamsBag) -> ToolResult:
        k = self.mp.config.recall_k if self.mp is not None else 3
        try:
            result = MemoryRecallService().recall(params.query, k_graph=k, k_map=k)
        except ValueError as exc:
            # An oversized query is a bad parameter, not a data problem: refuse
            # loudly (save_map's >5-cues precedent) instead of silently trimming.
            return ToolResult.err(str(exc), code="invalid-param")
        hits: list[dict[str, object]] = [
            {"source": hit.get("subject"), "memory": hit.get("contents")}
            for hit in result.get("graph", [])
        ]
        hits.extend(
            {"source": hit.get("source"), "memory": hit.get("contents")}
            for hit in result.get("map", [])
        )
        if not hits:
            return ToolResult.ok("No existing memory found.")
        # The dispatcher renders a list body as compact JSON, verbatim and
        # uncapped — a memory is never trimmed on its way to the model.
        return ToolResult.ok(hits)

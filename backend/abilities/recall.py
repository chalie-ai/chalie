"""Recall — search existing memory before writing (Memory v3).

Fuses Graph FTS (subject) + Map vector (contents, iteration-ranked). The one
memory surface the in-conversation model may use, and the consolidator's
recall-first step. Never writes.
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
            "forget. Returns top facts (Graph) and episodes (Map). Always recall "
            "first; never store blindly."
        )

    def get_examples(self) -> list[str]:
        return ["query='where does the user live'", "query='pet history'"]

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
        result = MemoryRecallService().recall(params.query, k_graph=k, k_map=k)
        graph = result.get("graph", [])
        episodes = result.get("map", [])
        lines: list[str] = []
        if graph:
            lines.append("Facts:")
            for hit in graph:
                lines.append(f"- {hit.get('subject')}: {hit.get('contents')}")
        if episodes:
            lines.append("Episodes:")
            for hit in episodes:
                lines.append(f"- {hit.get('contents')}")
        body = "\n".join(lines) if lines else "No existing memory found."
        return ToolResult.ok(body)

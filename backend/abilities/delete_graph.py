"""DeleteGraph — hard-delete a living fact from the memory graph (Memory v3).

A consolidator-only tool. Map episodes are never deleted — they become
irrelevant by ranking, never removed.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.delete_graph_params_bag import DeleteGraphParamsBag
from contracts.params.param_bag import ParamBag
from models.memory_graph import MemoryGraphRow


class DeleteGraph(Ability[DeleteGraphParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "delete_graph"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("subject",)}
    PARAMS: ClassVar[type[ParamBag] | None] = DeleteGraphParamsBag

    def get_summary(self) -> str:
        return (
            "Hard-delete a living fact by `subject`. Use only when a fact is "
            "obsolete with nothing worth keeping. Map episodes are never deleted."
        )

    def get_examples(self) -> list[str]:
        return ["subject='pet'"]

    def get_search_tooltip(self) -> str:
        return "delete a living fact from the memory graph"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {"subject": {"type": "string"}},
        "required": ["subject"],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: DeleteGraphParamsBag) -> ToolResult:
        deleted = MemoryGraphRow.delete_by_subject(params.subject)
        return ToolResult.ok({"subject": params.subject, "deleted": deleted})

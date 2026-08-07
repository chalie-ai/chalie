"""SaveMap — save a distilled episodic memory to the memory map (Memory v3).

``derived_from`` (map ids) retires those parents from the searchable pool.
Iteration (salience) is computed deterministically as max(parent iterations) + 1
(fresh episodes start at 1). A consolidator-only tool.
"""

from __future__ import annotations

import json
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.save_map_params_bag import SaveMapParamsBag
from models.memory_map import MemoryMapRow


class SaveMap(Ability[SaveMapParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "save_map"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("contents",)}
    PARAMS: ClassVar[type[ParamBag] | None] = SaveMapParamsBag

    def get_summary(self) -> str:
        return (
            "Save a distilled episodic memory. `contents` is the terse "
            "distillation of what happened. `derived_from` (optional, map ids) "
            "retires those parents from search. Distil — never store prose."
        )

    def get_examples(self) -> list[str]:
        return [
            "contents='moved from Berlin to Lisbon in March'",
            "contents='switched jobs to Acme' derived_from=[12]",
        ]

    def get_search_tooltip(self) -> str:
        return "save a distilled episode to the memory map"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            "contents": {"type": "string"},
            "derived_from": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["contents"],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SaveMapParamsBag) -> ToolResult:
        derived = list(params.derived_from)
        iteration = MemoryMapRow.max_iteration(derived) + 1 if derived else 1
        MemoryMapRow(
            contents=params.contents,
            derived_from=json.dumps(derived) if derived else "[]",
            iteration=iteration,
        ).save()
        return ToolResult.ok(
            {"saved": 1, "iteration": iteration, "derived_from": derived}
        )

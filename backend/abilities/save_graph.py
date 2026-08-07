"""SaveGraph — save/update a living fact in the memory graph (Memory v3).

Subject-keyed: saving an existing subject overwrites ``contents`` and bumps
``last_updated_at``. A consolidator-only tool — the in-conversation model only
recalls; it never writes memory.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.save_graph_params_bag import SaveGraphParamsBag
from models.memory_graph import MemoryGraphRow


class SaveGraph(Ability[SaveGraphParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "save_graph"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("subject", "contents")}
    PARAMS: ClassVar[type[ParamBag] | None] = SaveGraphParamsBag

    def get_summary(self) -> str:
        return (
            "Save or update a living fact. `subject` is the stable key "
            "(e.g. 'user.residence'); `contents` is the current terse value. "
            "Saving an existing subject overwrites it. Facts only — never prose."
        )

    def get_examples(self) -> list[str]:
        return [
            "subject='user.residence' contents='Lisbon'",
            "subject='partner' contents='Ana'",
            "subject='pet' contents='none'",
        ]

    def get_search_tooltip(self) -> str:
        return "save a living fact to the memory graph"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "contents": {"type": "string"},
        },
        "required": ["subject", "contents"],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SaveGraphParamsBag) -> ToolResult:
        MemoryGraphRow(subject=params.subject, contents=params.contents).save()
        return ToolResult.ok({"subject": params.subject, "saved": 1})

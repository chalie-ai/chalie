"""SaveMap — save a distilled episodic memory to the memory map (Memory v3).

``cues`` are the situational tags the episode is indexed by — the ONLY searchable
field, so an episode saved without them can never be recalled. ``source`` names
the episode's origin in one sentence and is rendered back on recall.
``derived_from`` (map ids) retires those parents from the searchable pool.
Iteration (salience) is computed deterministically as max(parent iterations) + 1
(fresh episodes start at 1). A memory-step-only tool.
"""

from __future__ import annotations

import json
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities.save_graph import _source_transcript_ids
from contracts.params.param_bag import ParamBag
from contracts.params.save_map_params_bag import SaveMapParamsBag
from models.memory_map import MemoryMapRow


class SaveMap(Ability[SaveMapParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "save_map"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": ("contents", "source", "cues")
    }
    PARAMS: ClassVar[type[ParamBag] | None] = SaveMapParamsBag

    def get_summary(self) -> str:
        return (
            "Save a distilled episodic memory. `contents` is the terse "
            "distillation of what happened. `source` is the origin in one "
            "sentence (e.g. 'Grocery trip in Zabbar in May 2026'). `cues` are up "
            "to 5 situational tags of 1-3 words each — the only way this memory "
            "is ever found again, so tag the SITUATION, not the story. "
            "`derived_from` (optional, map ids) retires those parents from "
            "search. Distil — never store prose."
        )

    def get_examples(self) -> list[str]:
        return [
            "contents='forgot half the list because his wife never sent it' "
            "source='Grocery trip in Zabbar in May 2026' "
            "cues=['shopping', 'incomplete shopping list']",
            "contents='gets frustrated shopping in person — lists and phone "
            "batteries keep failing him' source='Shopping trips in Malta in 2026' "
            "cues=['shopping', 'dead phone battery'] derived_from=[12, 14]",
        ]

    def get_search_tooltip(self) -> str:
        return "save a distilled episode to the memory map"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            "contents": {"type": "string"},
            "source": {"type": "string"},
            "cues": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": SaveMapParamsBag.MAX_CUES,
            },
            "derived_from": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["contents", "source", "cues"],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SaveMapParamsBag) -> ToolResult:
        derived = list(params.derived_from)
        iteration = MemoryMapRow.max_iteration(derived) + 1 if derived else 1
        row = MemoryMapRow(
            contents=params.contents,
            source=params.source,
            cues=", ".join(params.cues),
            derived_from=json.dumps(derived) if derived else "[]",
            iteration=iteration,
        )
        ids = _source_transcript_ids(self.mp)
        if ids:
            row.sourced_from = json.dumps(ids)
        row.save()
        return ToolResult.ok(
            {"saved": 1, "iteration": iteration, "derived_from": derived}
        )

"""SaveGraph — save/update a living fact in the memory graph (Memory v3).

Subject-keyed: saving an existing subject overwrites ``contents`` and bumps
``last_updated_at``. A memory-step-only tool — the in-conversation model only
recalls; it never writes memory.
"""

from __future__ import annotations

import json
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.save_graph_params_bag import SaveGraphParamsBag
from models.memory_graph import MemoryGraphRow


def _source_transcript_ids(mp: object) -> list[int]:
    """Provenance transcript ids the memory step stamps on its writes, carried
    on the invoking processor's config (``_source_transcript_ids``). Empty for
    other callers (e.g. build-time introspection)."""
    cfg = getattr(mp, "config", None) if mp is not None else None
    ids = getattr(cfg, "_source_transcript_ids", None)
    return list(ids) if ids else []


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
        row = MemoryGraphRow(subject=params.subject, contents=params.contents)
        ids = _source_transcript_ids(self.mp)
        if ids:
            row.sourced_from = json.dumps(ids)
        row.save()
        return ToolResult.ok({"subject": params.subject, "saved": 1})

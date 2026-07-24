"""SavePattern — record a repeating behavioural pattern in the data graph."""
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.save_pattern_params_bag import SavePatternParamsBag

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class SavePattern(Ability[SavePatternParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False  # pattern-write tool; pinned on the pattern configs only

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty name/frequency/summary/evidence_transcript_ids as
    # code=missing-params before run() is reached (precedent: save_graph.py).
    # The pre-gate is truthiness-based, so an empty evidence list is rejected
    # there too; the bag's from_params rejects the whitespace-only name/summary
    # residue that slips past it.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.name_, Keys.frequency, Keys.summary, Keys.evidence_transcript_ids)
    }

    # The typed input contract: the dispatch seam builds the bag via
    # SavePatternParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SavePatternParamsBag

    def get_name(self) -> str:
        return "save_pattern"

    def get_summary(self) -> str:
        return (
            "Record a repeating behavioural pattern observed in the user's "
            "transcripts. Use snake_case names; reuse existing names exactly "
            "when reinforcing (case-sensitive). Requires at least 2 evidence "
            "transcript ids."
        )

    def get_examples(self) -> list[str]:
        return [
            "user goes for a run every weekday morning",
            "user reads before bed most nights",
            "user checks email first thing each morning",
            "user has coffee around 07:30 on workdays",
            "user meditates on weekends",
            "user takes a walk after lunch on weekdays",
        ]

    def get_search_tooltip(self) -> str:
        return "record behavioural patterns"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.name_: {
                "type": "string",
                "description": "snake_case identifier; mirror existing names when reinforcing.",
            },
            Keys.frequency: {
                "type": "string",
                "enum": ["daily", "weekly", "weekday", "weekend", "ad-hoc"],
            },
            Keys.time_anchor: {
                "type": "string",
                "description": "Optional anchor: '07:00' | 'evening' | 'weekends' | '' if not applicable.",
            },
            Keys.summary: {
                "type": "string",
                "description": "One concise sentence describing the habitual behavior. Not a narrative or episode summary.",
            },
            Keys.evidence_transcript_ids: {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [Keys.name_, Keys.frequency, Keys.summary, Keys.evidence_transcript_ids],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SavePatternParamsBag) -> ToolResult:
        validated: dict[str, object] = {
            "name": params.name_,
            "frequency": params.frequency,
            "summary": params.summary,
            "evidence": params.evidence,
            "time_anchor": params.time_anchor,
        }
        row_id, confidence_out, reinforced = cast(
            "MessageProcessor", self.mp
        ).behavioral_pattern_service.store(validated)

        body: dict[str, object] = {"saved": 1}
        if reinforced:
            body["reinforced"] = 1
        body["name"] = params.name_
        body["confidence"] = confidence_out
        body["row_id"] = row_id
        return ToolResult.ok(body)

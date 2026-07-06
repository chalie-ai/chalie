"""SavePattern — record a repeating behavioural pattern in the data graph."""
import re
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._budget import BudgetCappedAbility
from abilities._params import Keys
from abilities._result import ToolResult

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})

# One-line recovery hint carrying a minimal, fully-valid example so a weak model
# can self-correct without re-reading the schema (parity with save_graph).
_EXAMPLE_HINT = (
    "use a snake_case name, e.g. name=morning_run frequency=weekday "
    "summary=user runs on weekdays evidence_transcript_ids=[12, 18]"
)


class SavePattern(BudgetCappedAbility):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False  # pattern-write tool; pinned on the pattern configs only

    BUDGET_CAP: ClassVar[int] = 20

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty name/frequency/summary/evidence_transcript_ids as
    # code=missing-params before run() is reached (precedent: save_graph.py,
    # file_permissions.py). The pre-gate is truthiness-based, so an empty
    # evidence list is rejected there too, while whitespace-only name/summary
    # residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.name, Keys.frequency, Keys.summary, Keys.evidence_transcript_ids)
    }

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
            Keys.name: {
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
        "required": [Keys.name, Keys.frequency, Keys.summary, Keys.evidence_transcript_ids],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        capped = self.budget_exceeded()
        if capped is not None:
            return capped

        # The validation ORDER and rules are frozen (regex → frequency set →
        # min-2 evidence); only the failure shape changes from a phantom-success
        # dict to a loud ToolResult error, in parity with save_graph.
        name = (cast("str", params.get(Keys.name) or "")).strip()
        # The dispatcher pre-gate is truthiness-based, so a whitespace-only name
        # slips past it as a non-empty string and must be rejected here
        # (precedent: save_graph.py, file_permissions.py).
        if not name:
            return ToolResult.err(
                "Missing required parameter(s): name.",
                code="missing-params",
                valid=("name", "frequency", "summary", "evidence_transcript_ids"),
            )
        if not _NAME_PATTERN.fullmatch(name):
            return ToolResult.err(
                f"Invalid pattern name {name!r}; must be snake_case.",
                code="invalid-param",
                hint=_EXAMPLE_HINT,
            )

        frequency = params.get(Keys.frequency, "")
        if frequency not in _VALID_FREQUENCIES:
            return ToolResult.err(
                f"Unknown frequency {frequency!r}; not a recognised cadence.",
                code="invalid-param",
                valid=tuple(sorted(_VALID_FREQUENCIES)),
                hint=_EXAMPLE_HINT,
            )

        summary = (cast("str", params.get(Keys.summary) or "")).strip()
        if not summary:
            return ToolResult.err(
                "Missing required parameter(s): summary.",
                code="missing-params",
                valid=("name", "frequency", "summary", "evidence_transcript_ids"),
            )

        evidence = params.get(Keys.evidence_transcript_ids)
        n_evidence = len(evidence) if isinstance(evidence, list) else 0
        if n_evidence < 2:
            return ToolResult.err(
                f"At least 2 evidence transcript ids are required; got {n_evidence}.",
                code="invalid-param",
                hint="provide at least 2 evidence_transcript_ids from the transcript window",
            )

        validated = {
            "name": name,
            "frequency": frequency,
            "summary": summary,
            "evidence": evidence,
            "time_anchor": params.get(Keys.time_anchor, "") or "",
        }
        row_id, confidence_out, reinforced = cast(
            "MessageProcessor", self.mp
        ).behavioral_pattern_service.store(validated)

        body: dict[str, object] = {"saved": 1}
        if reinforced:
            body["reinforced"] = 1
        body["name"] = validated["name"]
        body["confidence"] = confidence_out
        body["row_id"] = row_id
        return ToolResult.ok(body)

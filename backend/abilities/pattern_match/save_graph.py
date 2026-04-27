"""SaveGraphAbility — processor-internal ability for PatternMatchProcessor.

NOT registered in AbilityRegistry. Only PatternMatchProcessor imports this.
The normal Ability.execute() path is not used; callers must use
execute_with_processor() which carries the processor context needed for budget
tracking.
"""
from abilities._base import Ability
from services.data_graph_service import VALID_KINDS, get_data_graph_service

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})
# Yields (alphabetised): ['document', 'misc', 'moment', 'user_specific']


class SaveGraphAbility(Ability):
    NAME = "save_graph"
    SUMMARY = (
        "Record a durable fact about the user that is NOT a repeating behaviour. "
        "Processor-internal: only PatternMatchProcessor may call this."
    )
    EXAMPLES = [
        "user prefers dark mode",
        "user lives in Malta",
        "user's partner is named Sara",
        "user works as a software engineer",
        "user has a dog named Max",
        "user's favourite cuisine is Italian",
        "user is vegetarian",
        "user reads science fiction",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ALLOWED_KINDS},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["kind", "key", "value"],
    }
    ALWAYS_AVAILABLE = False
    INTERNAL = True
    TIMEOUT = 10

    # Canonical tool schema consumed by PatternMatchProcessor.getDynamicTools().
    TOOL_SCHEMA: dict = {
        "name": NAME,
        "description": (
            "Record a durable fact about the user that is NOT a repeating "
            "behaviour (preferences, identity, relationships, places, "
            "documents, timestamped events). Pick the right `kind`. Use "
            "save_pattern for repeating behaviours. Allowed kinds: "
            f"{', '.join(ALLOWED_KINDS)}."
        ),
        "input_schema": INPUT_SCHEMA,
    }

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        """Not used via normal ability dispatch — call execute_with_processor() instead."""
        raise NotImplementedError(
            "SaveGraphAbility is processor-internal. Use execute_with_processor()."
        )

    def execute_with_processor(self, args: dict, processor: object) -> dict:
        """Route a durable fact into DataGraphService.

        processor must be a PatternMatchProcessor instance — reads/writes
        _save_graph_calls.
        """
        # 1. Budget cap
        if processor._save_graph_calls >= 50:
            return {"budget_exceeded": True, "tool": "save_graph"}

        # 2. Validate
        kind = args.get("kind", "")
        if kind not in ALLOWED_KINDS:
            return {"error": "invalid_kind", "kind": kind}
        key = args.get("key", "")
        value = args.get("value", "")
        if not key:
            return {"error": "empty_key"}
        if not value:
            return {"error": "empty_value"}

        # 3. Route through DataGraphService
        try:
            get_data_graph_service().store(
                kind=kind,
                key=key,
                value=value,
                source="pattern_match",
            )
        except Exception as exc:
            return {"error": "store_failed", "message": str(exc)}

        # 4. Bookkeep
        processor._save_graph_calls += 1
        return {"ok": True}

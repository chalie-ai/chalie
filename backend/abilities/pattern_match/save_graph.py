"""SaveGraph — processor-internal helper for PatternMatchProcessor.

Not an Ability subclass and not registered in AbilityRegistry. Lives in a
subdirectory so the registry's shallow ``glob("*.py")`` walk skips it. Only
PatternMatchProcessor imports this module.
"""
from services.data_graph_service import VALID_KINDS, get_data_graph_service

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})
# Yields (alphabetised): ['document', 'misc', 'moment', 'user_specific']


class SaveGraph:
    NAME = "save_graph"
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ALLOWED_KINDS},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["kind", "key", "value"],
    }

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

    def execute(self, args: dict, processor: object) -> dict:
        """Route a durable fact into DataGraphService.

        processor must be a PatternMatchProcessor instance — reads/writes
        _save_graph_calls.
        """
        if processor._save_graph_calls >= 50:
            return {"budget_exceeded": True, "tool": "save_graph"}

        kind = args.get("kind", "")
        if kind not in ALLOWED_KINDS:
            return {"error": "invalid_kind", "kind": kind}
        key = args.get("key", "")
        value = args.get("value", "")
        if not key:
            return {"error": "empty_key"}
        if not value:
            return {"error": "empty_value"}

        try:
            get_data_graph_service().store(
                kind=kind,
                key=key,
                value=value,
                source="pattern_match",
            )
        except Exception as exc:
            return {"error": "store_failed", "message": str(exc)}

        processor._save_graph_calls += 1
        return {"ok": True}

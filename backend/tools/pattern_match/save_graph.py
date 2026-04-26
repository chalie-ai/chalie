from services.data_graph_service import VALID_KINDS, get_data_graph_service

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})
# Yields (alphabetised): ['document', 'misc', 'moment', 'user_specific']

TOOL_SCHEMA = {
    "name": "save_graph",
    "description": (
        "Record a durable fact about the user that is NOT a repeating "
        "behaviour (preferences, identity, relationships, places, "
        "documents, timestamped events). Pick the right `kind`. Use "
        "save_pattern for repeating behaviours. Allowed kinds: "
        f"{', '.join(ALLOWED_KINDS)}."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ALLOWED_KINDS},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["kind", "key", "value"],
    },
}


def execute(args: dict, ctx: dict) -> dict:
    """Route a durable fact into DataGraphService.

    ctx must contain:
      - 'processor': PatternMatchProcessor instance — reads/writes
        `_save_graph_calls`.
    """
    processor = ctx["processor"]

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

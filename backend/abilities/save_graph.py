"""SaveGraph — record a durable, non-behavioural fact in the data graph.

Reachable when a processor lists ``"save_graph"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget state lives on the calling processor (read via
``current_processor()``).  PMP initialises ``_save_graph_calls = 0`` in
``__init__``; this Ability uses ``getattr`` defaults so it remains usable
from any processor that opts it in.
"""
import logging

from abilities._base import Ability
from services.data_graph_service import VALID_KINDS, get_data_graph_service
from services.message_processor import current_processor

logger = logging.getLogger(__name__)

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})

_BUDGET_CAP = 50


class SaveGraph(Ability):
    NAME = "save_graph"
    SYSTEM = True
    SUMMARY = (
        "Record a durable fact about the user that is NOT a repeating "
        "behaviour (preferences, identity, relationships, places, "
        "documents, timestamped events). Pick the right `kind`. Use "
        "save_pattern for repeating behaviours. Allowed kinds: "
        f"{', '.join(ALLOWED_KINDS)}."
    )
    EXAMPLES = [
        "user's favourite colour is blue",
        "user lives in Lisbon",
        "user's partner is called Ana",
        "user prefers dark mode interfaces",
        "user bought a new laptop on 2026-04-15",
        "user's workplace is downtown",
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
    TIMEOUT = 10

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        proc = current_processor()
        count = getattr(proc, "_save_graph_calls", 0) if proc is not None else 0
        if count >= _BUDGET_CAP:
            return {"budget_exceeded": True, "tool": "save_graph"}

        kind = params.get("kind", "")
        if kind not in ALLOWED_KINDS:
            return {"error": "invalid_kind", "kind": kind}
        key = params.get("key", "")
        value = params.get("value", "")
        if not key:
            return {"error": "empty_key"}
        if not value:
            return {"error": "empty_value"}

        dedup_key = (kind, key.lower().strip(), value.lower().strip())
        seen: set | None = getattr(proc, "_save_graph_seen", None) if proc else None
        if seen is not None and dedup_key in seen:
            return {"already_stored": True, "key": key}

        try:
            result = get_data_graph_service().store(
                kind=kind,
                key=key,
                value=value,
                source="pattern_match",
            )
        except Exception as exc:
            return {"error": "store_failed", "message": str(exc)}

        if proc is not None:
            proc._save_graph_calls = count + 1
            if seen is None:
                proc._save_graph_seen = set()
            proc._save_graph_seen.add(dedup_key)

        if result and result.get("status") == "reinforced":
            return {"already_stored": True, "key": key}

        return {"ok": True}

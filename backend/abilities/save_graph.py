"""SaveGraph — record a durable, non-behavioural fact in the data graph.

Reachable when a processor lists ``"save_graph"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget state lives on the calling processor (read via
``self.mp``).  PMP initialises ``_save_graph_calls = 0`` in
``__init__``; this Ability uses ``getattr`` defaults so it remains usable
from any processor that opts it in.
"""
import logging
from typing import ClassVar

from abilities._budget import BudgetCappedAbility
from abilities._result import ToolResult
from services.data_graph_service import VALID_KINDS, get_data_graph_service

logger = logging.getLogger(__name__)

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})


class SaveGraph(BudgetCappedAbility):
    SYSTEM = True

    BUDGET_COUNTER_ATTR: ClassVar[str] = "_save_graph_calls"
    BUDGET_CAP: ClassVar[int] = 50

    def get_name(self) -> str:
        return "save_graph"

    def get_summary(self) -> str:
        return (
            "Record a durable fact about the user that is NOT a repeating "
            "behaviour (preferences, identity, relationships, places, "
            "documents, timestamped events). Pick the right `kind`. Use "
            "save_pattern for repeating behaviours. Allowed kinds: "
            f"{', '.join(ALLOWED_KINDS)}."
        )

    def get_examples(self) -> list[str]:
        return [
            "user's favourite colour is blue",
            "user lives in Lisbon",
            "user's partner is called Ana",
            "user prefers dark mode interfaces",
            "user bought a new laptop on 2026-04-15",
            "user's workplace is downtown",
        ]

    def get_search_tooltip(self) -> str:
        return "store user facts to knowledge graph"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ALLOWED_KINDS},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["kind", "key", "value"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        capped = self.budget_exceeded()
        if capped is not None:
            return capped

        proc = self.mp
        kind = params.get("kind", "")
        if kind not in ALLOWED_KINDS:
            return ToolResult.ok({"error": "invalid_kind", "kind": kind})
        key = params.get("key", "")
        value = params.get("value", "")
        if not key:
            return ToolResult.ok({"error": "empty_key"})
        if not value:
            return ToolResult.ok({"error": "empty_value"})

        dedup_key = (kind, key.lower().strip(), value.lower().strip())
        seen: set | None = getattr(proc, "_save_graph_seen", None) if proc else None
        if seen is not None and dedup_key in seen:
            return ToolResult.ok({"already_stored": True, "key": key})

        try:
            result = get_data_graph_service().store(
                kind=kind,
                key=key,
                value=value,
                source="pattern_match",
            )
        except Exception as exc:
            return ToolResult.ok({"error": "store_failed", "message": str(exc)})

        self.bump_budget()
        if proc is not None:
            if seen is None:
                proc._save_graph_seen = set()
            proc._save_graph_seen.add(dedup_key)

        if result and result.get("status") == "reinforced":
            return ToolResult.ok({"already_stored": True, "key": key})

        return ToolResult.ok({"ok": True})

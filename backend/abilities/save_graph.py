"""SaveGraph — record a durable, non-behavioural fact in the data graph.

Reachable when a processor lists ``"save_graph"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget state lives on the calling processor (read via
``self.mp``).  PMP initialises ``_save_graph_calls = 0`` in
``__init__``; this Ability uses ``getattr`` defaults so it remains usable
from any processor that opts it in.
"""
from typing import ClassVar

from abilities._budget import BudgetCappedAbility
from abilities._pattern_provenance import pattern_provenance
from abilities._result import ToolResult
from services.data_graph_service import VALID_KINDS, get_data_graph_service

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})

# One-line invalid-kind recovery hint carrying a minimal, fully-valid example so
# a weak model can self-correct without re-reading the schema.
_INVALID_KIND_HINT = "pick one of the allowed kinds, e.g. kind=user_specific key=residence value=Lisbon"


class SaveGraph(BudgetCappedAbility):
    SYSTEM = True

    BUDGET_COUNTER_ATTR: ClassVar[str] = "_save_graph_calls"
    BUDGET_CAP: ClassVar[int] = 50

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty kind/key/value as code=missing-params before run() is reached
    # (precedent: _delegate.py, file_permissions.py). The pre-gate is
    # truthiness-based, so whitespace-only residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict] = {"": ("kind", "key", "value")}

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
            return ToolResult.err(
                f"Unknown kind {kind!r}; not a storable fact kind.",
                code="invalid-param",
                valid=tuple(ALLOWED_KINDS),
                hint=_INVALID_KIND_HINT,
            )

        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only key/value slips past it and must be rejected here
        # (precedent: file_permissions.py).
        key = params.get("key", "").strip()
        value = params.get("value", "").strip()
        if not key or not value:
            missing = ", ".join(
                name for name, val in (("key", key), ("value", value)) if not val
            )
            return ToolResult.err(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("kind", "key", "value"),
            )

        dedup_key = (kind, key.lower(), value.lower())
        seen: set | None = getattr(proc, "_save_graph_seen", None) if proc else None
        if seen is not None and dedup_key in seen:
            return ToolResult.ok({"saved": 0, "deduped": 1, "key": key})

        # DataGraphService.store() never raises — it swallows every failure
        # internally and returns None (data_graph_service.store: try/except →
        # logger.error → return None). A None result is therefore the only
        # store-failure signal; surface it loudly instead of as a phantom save.
        result = get_data_graph_service().store(
            kind=kind,
            key=key,
            value=value,
            source=pattern_provenance(proc),
        )
        if result is None:
            return ToolResult.err(
                f"Could not store {kind} fact {key!r}.",
                code="store-failed",
                hint="check the data graph service logs; nothing was persisted",
            )

        self.bump_budget()
        if proc is not None:
            if seen is None:
                proc._save_graph_seen = set()
            proc._save_graph_seen.add(dedup_key)

        if result.get("status") == "reinforced":
            return ToolResult.ok({"saved": 0, "deduped": 1, "key": key})

        return ToolResult.ok({"saved": 1, "kind": kind, "key": key})

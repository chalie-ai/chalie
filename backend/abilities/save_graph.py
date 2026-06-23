"""SaveGraph — record a durable, non-behavioural fact in the data graph."""
import json
from typing import ClassVar, cast

from abilities._budget import BudgetCappedAbility
from abilities._params import Keys
from abilities._pattern_provenance import pattern_provenance
from abilities._result import ToolResult
from services.act_trail import ActTrail
from services.data_graph_service import VALID_KINDS, get_data_graph_service

# Subset: exclude behavioral_pattern (own tool) and system (internal use).
ALLOWED_KINDS = sorted(VALID_KINDS - {"behavioral_pattern", "system"})

# One-line invalid-kind recovery hint carrying a minimal, fully-valid example so
# a weak model can self-correct without re-reading the schema.
_INVALID_KIND_HINT = "pick one of the allowed kinds, e.g. kind=user_specific key=residence value=Lisbon"


class SaveGraph(BudgetCappedAbility):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False  # pattern-write tool; pinned on the pattern configs only

    BUDGET_CAP: ClassVar[int] = 50

    def _seen_from_trail(self) -> set[tuple[str, str, str]]:
        """Build the dedup set from this turn's prior save_graph rows."""
        proc = self.mp
        if proc is None:
            return set()
        channel = getattr(getattr(proc, "config", None), "channel", None)
        turn_id = getattr(proc, "turn_id", None)
        if channel is None or turn_id is None:
            return set()
        seen: set[tuple[str, str, str]] = set()
        for row in ActTrail().fetch_by_turn(channel, turn_id):
            if row.get("tool_name") != "save_graph":
                continue
            try:
                p = json.loads(cast("str", row.get("params") or "{}"))
            except (ValueError, TypeError):
                continue
            k = p.get(Keys.kind, "")
            key = p.get(Keys.key, "")
            val = p.get(Keys.value, "")
            if k and key and val:
                seen.add((k, key.lower(), val.lower()))
        return seen

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty kind/key/value as code=missing-params before run() is reached
    # (precedent: _delegate.py, file_permissions.py). The pre-gate is
    # truthiness-based, so whitespace-only residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.kind, Keys.key, Keys.value)}

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

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.kind: {"type": "string", "enum": ALLOWED_KINDS},
            Keys.key: {"type": "string"},
            Keys.value: {"type": "string"},
        },
        "required": [Keys.kind, Keys.key, Keys.value],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        capped = self.budget_exceeded()
        if capped is not None:
            return capped

        proc = self.mp
        kind = params.get(Keys.kind, "")
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
        key = cast("str", params.get(Keys.key, "")).strip()
        value = cast("str", params.get(Keys.value, "")).strip()
        if not key or not value:
            missing = ", ".join(
                name for name, val in (("key", key), ("value", value)) if not val
            )
            return ToolResult.err(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("kind", "key", "value"),
            )

        # Derive the dedup set from the persisted trail — single DB round-trip.
        seen = self._seen_from_trail()
        dedup_key = (kind, key.lower(), value.lower())
        if dedup_key in seen:
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

        if result.get("status") == "reinforced":
            return ToolResult.ok({"saved": 0, "deduped": 1, "key": key})

        return ToolResult.ok({"saved": 1, "kind": kind, "key": key})

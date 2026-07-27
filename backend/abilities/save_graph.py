"""SaveGraph — record a durable, non-behavioural fact in the data graph."""
import json
from typing import ClassVar

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._pattern_provenance import pattern_provenance
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.save_graph_params_bag import ALLOWED_KINDS, SaveGraphParamsBag
from models.tool_call import ToolCall


class SaveGraph(Ability[SaveGraphParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False  # pattern-write tool; pinned on the pattern configs only
    NAME: ClassVar[str] = "save_graph"

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
        for row in ToolCall.by_turn(channel, turn_id):
            if row.tool_name != "save_graph":
                continue
            # The dispatcher opens this very call's trail row (result="") before
            # run() executes, so it surfaces here too. An empty result means the
            # row is still in flight (incl. this call) — only committed prior
            # writes count toward dedup.
            if not row.result:
                continue
            try:
                p = json.loads(row.params or "{}")
            except (ValueError, TypeError):
                continue
            k = p.get(Keys.kind, "")
            key = p.get(Keys.key, "")
            val = p.get(Keys.value_, "")
            if k and key and val:
                seen.add((k, key.lower(), val.lower()))
        return seen

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty kind/key/value as code=missing-params before run() is reached
    # (precedent: _delegate.py). The pre-gate is truthiness-based; the bag's
    # from_params rejects the whitespace-only residue that slips past it.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.kind, Keys.key, Keys.value_)}

    # The typed input contract: the dispatch seam builds the bag via
    # SaveGraphParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SaveGraphParamsBag

    def get_summary(self) -> str:
        from abilities.save_pattern import SavePattern  # noqa: PLC0415

        return (
            "Record a durable fact about the user that is NOT a repeating "
            "behaviour (preferences, identity, relationships, places, "
            "documents, timestamped events). Pick the right `kind`. Use "
            f"{SavePattern.NAME} for repeating behaviours. Allowed kinds: "
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
            Keys.value_: {"type": "string"},
        },
        "required": [Keys.kind, Keys.key, Keys.value_],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SaveGraphParamsBag) -> ToolResult:
        proc = self.mp
        kind, key, value = params.kind, params.key, params.value_

        # Derive the dedup set from the persisted trail — single DB round-trip.
        seen = self._seen_from_trail()
        dedup_key = (kind, key.lower(), value.lower())
        if dedup_key in seen:
            return ToolResult.ok({"saved": 0, "deduped": 1, "key": key})

        # Every ALLOWED_KIND routes to its vertical (rich status →
        # reinforced-dedup; place = supersede).
        source = pattern_provenance(proc)
        if kind == "user_specific":
            from services.fact_service import FactService
            status = FactService().store(key, value, source=source)["status"]
        elif kind == "place":
            from services.place_service import PlaceService
            status = PlaceService().store(key, value, source=source)["status"]
        elif kind == "discovery":
            from services.discovery_service import DiscoveryService
            status = DiscoveryService().store(key, value, source=source)["status"]
        elif kind == "misc":
            from services.misc_service import MiscService
            status = MiscService().store(key, value, source=source)["status"]
        elif kind == "contact":
            from services.contact_service import ContactService
            ContactService().store(key, value, source=source)  # returns ContactRow, NOT a status envelope
            status = "created"
        else:  # unreachable — the bag gates kind ∈ ALLOWED_KINDS, all of which branch above
            raise RuntimeError(f"save_graph: no vertical wired for kind {kind!r}")

        if status == "reinforced":
            return ToolResult.ok({"saved": 0, "deduped": 1, "key": key})

        return ToolResult.ok({"saved": 1, "kind": kind, "key": key})

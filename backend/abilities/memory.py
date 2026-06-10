"""
MemoryAbility — Store, recall, and forget first-party facts about the user.

Covers all actions: store, recall, reflect, forget. This module is a **thin
adapter**: it owns only the tool's metadata (name, summary, examples, parameter
schema) and the action dispatch in ``run()``. Every retrieval/mutation handler,
the episode recall engine with its dynamic radius, data-graph search, reflection
layer expansion, response formatting and recall telemetry live in
``services.memory_retrieval`` so the same engine is reachable from non-ability
callers (e.g. the ``/api/updates/memory`` REST endpoint) without importing an
ability.

The dynamic-radius tuning constants are the service's source of truth; they are
re-exposed here as ``MemoryAbility`` ClassVars so they remain mechanically
tunable and inspectable via the ability surface. The engine helpers the tests
drive through this module (``_compute_radius_factors``, ``_render_behavioral_pattern``,
``_format_results``, ``_search_data_graph``) are imported from the service — the
ability module IS the public surface those harnesses exercise.
"""

import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from services import memory_retrieval
from services.memory_retrieval import (  # noqa: F401  (re-exported engine surface)
    _compute_radius_factors,
    _format_results,
    _render_behavioral_pattern,
    _search_data_graph,
    recall_episodes,
)

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"


class MemoryAbility(Ability):
    def get_name(self) -> str:
        return "memory"

    def get_summary(self) -> str:
        return "Store, recall, or forget first-party facts about the user — traits, preferences, relationships, goals, and habits."

    def get_examples(self) -> list[str]:
        return [
            "please remember that my wifi password is BlueSky42",
            "what's my wifi password again",
            "I want you to forget that I told you my salary",
            "do you know anything about my dietary preferences",
            "save the fact that I have a dog named Biscuit",
            "reflect on everything you know about my exercise habits",
            "what happened at home last week",
            "recall conversations in Zabbar",
        ]

    def get_search_tooltip(self) -> str:
        return "personal memory store"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "recall", "reflect", "forget"],
                "description": (
                    "store: save a fact. recall: search memory - fast. "
                    "reflect: deep search about a topic. "
                    "forget: permanently remove a memory."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["user_specific", "system", "misc"],
                "description": (
                    "user_specific: first-party facts about the human you "
                    "are talking to (traits, preferences, relationships, "
                    "secrets, goals). system: about how Chalie operates "
                    "(rules, decisions, analysis). misc: short-lived "
                    "scratchpad for Chalie's own working notes — NOT a "
                    "dumping ground for user-supplied bulk content (use the "
                    "`document` tool for that)."
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "For store/forget: the canonical key from the list in the "
                    "tool description (e.g. 'residence', 'food_and_drink', "
                    "'employment'). Use the exact canonical key when the fact "
                    "fits one of the 27 concepts."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "For store: the fact itself, atomic — a single value. "
                    "'Valletta' (not 'lives in Valletta'). 'pasta' (not "
                    "'loves pasta and pizza'). "
                    "For forget: required when removing one value from a "
                    "multi-value key."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "For recall/reflect: what to search for. One topic per "
                    "call — to fetch memories about different topics, call "
                    "this tool once for each topic. If results are broad or "
                    "sparse, try searching again with more narrow queries."
                ),
            },
            "location": {
                "type": "string",
                "description": (
                    "A location to filter memories by. Use a city, country, "
                    "or a saved place name like 'home' or 'work'. "
                    "When set, only memories from that location are returned. "
                    "You can use location without query to get all memories "
                    "from a place regardless of topic."
                ),
            },
        },
        "required": ["action"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    # SYSTEM tool: always allowed in every context and never shown in the Policy
    # Manager. Memory is core to Chalie's operation, so it bypasses policy like
    # skill_manager.
    SYSTEM = True

    # Dynamic-radius tuning constants — the service holds the source of truth;
    # these ClassVars re-expose them on the ability surface so they stay
    # inspectable and mechanically tunable. (TKT-878: RECALL=0.5, SEED=0.35.)
    RECALL_RADIUS_BASELINE: ClassVar[float] = memory_retrieval.RECALL_RADIUS_BASELINE
    SEED_RADIUS_BASELINE: ClassVar[float] = memory_retrieval.SEED_RADIUS_BASELINE

    NARROW_MIN_DIST: ClassVar[float] = memory_retrieval.NARROW_MIN_DIST
    NARROW_MAX_DIST: ClassVar[float] = memory_retrieval.NARROW_MAX_DIST
    NARROW_FACTOR_FLOOR: ClassVar[float] = memory_retrieval.NARROW_FACTOR_FLOOR

    EXPAND_MIN_DIST: ClassVar[float] = memory_retrieval.EXPAND_MIN_DIST
    EXPAND_MAX_DIST: ClassVar[float] = memory_retrieval.EXPAND_MAX_DIST
    EXPAND_FACTOR_CEILING: ClassVar[float] = memory_retrieval.EXPAND_FACTOR_CEILING

    def run(self, params: dict) -> ToolResult:
        action = params.get("action", "recall")
        mp = self.mp
        channel = getattr(getattr(mp, "config", None), "channel", "") or ""

        try:
            if action == "store":
                return memory_retrieval.handle_store(channel, params)
            if action == "recall":
                return memory_retrieval.handle_recall(mp, channel, params)
            if action == "reflect":
                return memory_retrieval.handle_reflect(mp, channel, params)
            if action == "forget":
                return memory_retrieval.handle_forget(params)
            return ToolResult.err(
                f"unknown-action:{action}",
                code="error",
                valid=("store", "recall", "reflect", "forget"),
            )
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Error in {action}: {e}")
            return ToolResult.err(str(e)[:200], code="error", action=action)

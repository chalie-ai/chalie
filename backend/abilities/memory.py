"""
MemoryAbility — Store, recall, and forget first-party facts about the user.

Covers all actions: store, recall, reflect, forget. This module is a **thin
adapter**: it owns only the tool's metadata (name, summary, examples, parameter
schema) and the action dispatch in ``run()`` — ``params → service handler →
ToolResult``. Every retrieval/mutation handler, the episode recall engine,
data-graph search, reflection layer expansion, response formatting and recall
telemetry live in ``services.memory_retrieval`` so the same engine is reachable
from non-ability callers without importing an ability.

The engine functions live ONLY in the service module — there are no re-exports
from here: callers and tests import them from ``services.memory_retrieval``
directly.
"""

import logging
from typing import ClassVar

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from services import memory_retrieval

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"


class MemoryAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # framework tool; always pinned (DEFAULT_ALWAYS_AVAILABLE), never discovered

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

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["store", "recall", "reflect", "forget"],
                "description": (
                    "store: save a fact. recall: search memory - fast. "
                    "reflect: deep search about a topic. "
                    "forget: permanently remove a memory."
                ),
            },
            Keys.kind: {
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
            Keys.key: {
                "type": "string",
                "description": (
                    "For store/forget: the canonical key naming the fact "
                    "(e.g. 'residence', 'food_and_drink', 'employment'). "
                    "Use the exact canonical key when the fact fits one of the canonical concepts."
                ),
            },
            Keys.value_: {
                "type": "string",
                "description": (
                    "For store: the fact itself, atomic — a single value. "
                    "'Valletta' (not 'lives in Valletta'). 'pasta' (not "
                    "'loves pasta and pizza'). "
                    "For forget: required when removing one value from a "
                    "multi-value key."
                ),
            },
            Keys.replaces: {
                "type": "string",
                "description": (
                    "For store: when correcting a fact already stored under "
                    "this key, the OLD value being replaced. The old value is "
                    "demoted and the new value takes its place. Omit when "
                    "adding a genuinely new fact."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": (
                    "For recall/reflect: what to search for. One topic per "
                    "call — to fetch memories about different topics, call "
                    "this tool once for each topic. If results are broad or "
                    "sparse, try searching again with more narrow queries."
                ),
            },
            Keys.location: {
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
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    # SYSTEM tool: always allowed in every context and never shown in the Policy
    # Manager. Memory is core to Chalie's operation, so it bypasses policy like
    # skill_manager.
    SYSTEM = True

    # Action → required params, pre-validated by the dispatcher BEFORE run(): an
    # unknown action yields one ``unknown-action`` error listing these keys; a
    # known action missing a param yields one ``missing-params`` error naming it.
    # Every action MUST appear (the pre-gate rejects any action not in this map),
    # so ``recall`` is listed with NO required params — it accepts EITHER
    # ``query`` OR ``location`` (an OR the flat map cannot express), and the
    # handler validates that pair itself, returning ``code=no-query-or-location``.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "store": (Keys.key, Keys.value_),
        "recall": (),
        "reflect": (Keys.query,),
        "forget": (Keys.key,),
    }

    def run(self, params: dict[str, object]) -> ToolResult:
        action = params.get(Keys.action, "recall")
        mp = self.mp
        if mp is None:
            raise RuntimeError("memory.run() dispatched without a bound MessageProcessor")
        channel = mp.config.channel

        # The dispatcher's ACTION_REQUIRED pre-gate has already rejected unknown
        # actions and missing required params before run() is reached; this maps
        # the validated action to its service handler. The unknown-action branch
        # below is defensive belt-and-braces for direct callers that bypass the
        # gate (the engine is also reachable outside the ACT loop).
        try:
            if action == "store":
                return memory_retrieval.handle_store(channel, params)
            if action == "recall":
                return memory_retrieval.handle_recall(mp, channel, params)
            if action == "reflect":
                return memory_retrieval.handle_reflect(mp, params)
            if action == "forget":
                return memory_retrieval.handle_forget(params)
            return ToolResult.err(
                f"Unknown memory action: {action!r}.",
                code="unknown-action",
                valid=("store", "recall", "reflect", "forget"),
            )
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Error in {action}: {e}")
            return ToolResult.err(
                f"Memory {action} failed: {str(e)[:200]}",
                code="unhandled-exception",
                action=action,
            )

"""The ``Ability`` ABC — describes and runs one tool, nothing more.

An Ability declares what a tool *is* through five zero-arg getters
(``get_name`` / ``get_summary`` / ``get_examples`` / ``get_search_tooltip`` /
``get_parameters``) and how it *runs* (``run()``). The full LLM-facing tool
descriptor is assembled in ONE place — the ``final`` ``get_input_schema()`` — which
is also the SINGLE site that injects the ``act_summary`` framework field.
Subclasses cannot override it; they only fill in the getters. The ``async``
backgrounding flag is NOT injected here: it is a delegate-only primitive added by
``DelegateAbility`` (``abilities/_delegate.py``), so plain tools never carry it.

Dispatch — matching, binding, policy gating, execution, recording — lives in
``DispatchService`` (``services/dispatch_service.py``), the single chokepoint
every ACT-loop tool call takes. This module imports nothing from the registry /
policy / dispatcher, so ``_registry`` can import ``Ability`` here with no
circular-import dance.
"""

from __future__ import annotations

import copy
import typing
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast

from typing_extensions import TypeVar

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

# The act_summary framework field, injected into EVERY tool descriptor by
# get_input_schema — the one place it is declared. It is the per-call act-trail
# tooltip (always present, required). The ``async`` backgrounding flag lives with
# ``DelegateAbility`` in abilities/_delegate.py — it is delegate-only.
_ACT_SUMMARY_PROPERTY: dict[str, object] = {
    "type": "string",
    "description": (
        "A ~3-10 word summary of what this specific tool call does, shown to"
        ' the user as a tooltip (e.g. "Searching for laptops in Malta",'
        ' "Looking up the weather in London").'
    ),
}

# What run() receives: the ability's typed ParamBag. Every first-party ability
# declares ``Ability[ItsParamsBag]`` and mypy then enforces that its run()
# accepts exactly that bag. The PEP 696 default keeps bare ``Ability``
# annotations (registry, dispatcher) valid and covers the synthetic
# ``_MCPAbility`` proxies, whose run() receives the raw params dict because a
# remote MCP schema can never have a compile-time bag.
B = TypeVar("B", default=Any)


class Ability(ABC, Generic[B]):
    """The getters read ``self.mp`` (the invoking MessageProcessor,
    constructor-injected) when a value depends on the live request; at
    ``self.mp is None`` — introspection / search-index build — they MUST return
    deterministic base text.

    Tool *scope* is two flags and nothing else. ``DISCOVERABLE`` (below) decides
    whether ``find_tools`` may ever surface this ability — it is a single global
    trait of the ability, not a per-channel list. A MessageProcessor reaches a
    tool one of exactly two ways: the config injects it directly
    (``ProcessorConfig.always_available``), or — if the ability is
    ``DISCOVERABLE`` — the processor discovers it through ``find_tools`` (which
    the processor only carries when ``find_tools`` is in its always_available).
    There is no per-config discoverable/blocked allow-list. Whether a call blocks
    or runs in the background is a per-call decision (the framework ``async``
    flag, exposed ONLY on delegate tools via ``DelegateAbility``), not an
    ability-level trait.
    """

    # Global discovery flag. True (the default) means find_tools may surface this
    # ability — semantically via the abilities.sqlite index (query mode) and by
    # exact name (select mode). False removes it from the index AND the select
    # roster entirely: it can ONLY reach a processor by being pinned directly in
    # that processor's ProcessorConfig.always_available. Internal/framework tools
    # (the compactors, thinking, find_tools itself, the pattern writers, the raw
    # web tools, memory) set this False; user-facing tools leave it True.
    DISCOVERABLE: ClassVar[bool] = True

    # Settle flag. True (the default) means a tool_calls row for this ability
    # demotes its transcript row's settled=1 back to 0 — the row carries a
    # model-driven tool and is therefore NOT a settle0. Internal framework passes
    # (chat_history_compactor, thinking) set this False so they never demote a
    # settle: their tool_calls rows are implementation artefacts, not model tools.
    counts_as_settle: ClassVar[bool] = True

    # Constructor-injected, the invoking MessageProcessor (the "parent" of this
    # tool call). A tool reads ALL its context off this — self.mp.config.channel,
    # .config.policy_channel, ._uid, etc. This is the traceability spine: every
    # hop holds a real reference to its parent instead of reaching into a hidden
    # global. None only on a synthetic / introspection / build-time instance.
    # Maps action → required param names (key ``""`` for action-less tools). The
    # dispatcher validates this BEFORE run(): an unknown action → one error with
    # ``valid=`` the action keys; a known action missing params → one
    # ``missing-params`` error naming ALL of them. The default empty dict means
    # "no pre-validation" so unmigrated tools are untouched by the contract.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {}

    # The ability's typed input contract: every first-party ability sets its
    # ParamBag class here and run() receives an instance the dispatcher builds
    # via the bag's from_params factory — a bad input comes back from the bag
    # as an error ToolResult the dispatcher returns as-is, never an exception.
    # None is reserved for the synthetic _MCPAbility proxies (remote schema, no
    # compile-time bag): their run() receives the raw params dict.
    PARAMS: ClassVar["type[ParamBag] | None"] = None

    def __init__(self, mp: "MessageProcessor | None" = None) -> None:
        self.mp = mp
        # Set by DispatchService._run() immediately before run(): the flattened
        # client telemetry dict (location / locale / time / currency …) or None
        # when no client context is stored yet (fresh boot, no heartbeat).
        self.telemetry: "dict[str, object] | None" = None

    # ── Metadata getters — every concrete ability implements all five ──────────

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def get_summary(self) -> str:
        """Override to enrich at runtime (e.g. bash appends the cwd) — gate
        enrichment on ``self.mp is not None`` so the build-time index stays
        deterministic."""
        ...

    @abstractmethod
    def get_examples(self) -> list[str]:
        """6–8 entries, embedded + FTS-indexed for find_tools search."""
        ...

    @abstractmethod
    def get_search_tooltip(self) -> str:
        ...

    def get_follow_up(self, tr: "ToolResult") -> str:
        """A standing next-step nudge appended to this tool's SUCCESSFUL result.

        Default ``""`` — no nudge. Override on a tool whose success routinely leaves
        the model one obvious, loop-safe step short of a complete answer
        (search→read, find_tools→call the activated tool): return a short
        instruction and the dispatcher wraps it in a
        ``[follow_up_instruction]…[end:follow_up_instruction]`` block placed INSIDE
        the envelope (after any rich-media block, before the closing
        ``[end:tool]``), on success only.

        ``tr`` is the SUCCESS :class:`ToolResult` being rendered, so an override can
        interpolate live values straight from the result the model already sees
        (the downloaded ``path``, the activated tool ``name``, the anchor
        ``date_time``) — present-in-context data lifts compliance over a generic
        nudge. An override that needs data the result lacks MUST degrade to ``""``
        rather than assume a shape. No ``self.mp`` — the nudge is request-agnostic.
        """
        return ""

    @abstractmethod
    def get_parameters(self) -> dict[str, object]:
        """Override to enrich at runtime (e.g. bash folds the live cwd into its
        description). When the enrichment reads live request state, gate on
        ``self.mp`` so the build-time (mp=None) schema stays deterministic for
        the search-index/SHA build. (find_tools enriches from the global
        discoverable roster and is itself DISCOVERABLE=False, so it is never
        part of that build.)"""
        ...

    # ── Instance method — the actual tool logic ───────────────────────────────

    @abstractmethod
    def run(self, params: B) -> "ToolResult":
        """MUST return a :class:`abilities._result.ToolResult` — anything else
        HARD-FAILS as ``code=non-canonical-result``. ``params`` is this ability's
        declared ``PARAMS`` bag, constructed (= validated) by the dispatcher: a
        bad input never reaches run() — the bag returned the error ToolResult
        from ``from_params`` and the dispatcher passed it straight to the wire.
        The ability NEVER formats the ``[tool(...)]`` wire envelope — the
        dispatcher owns it."""
        ...

    # ── The single, final tool-descriptor assembler ───────────────────────────

    @typing.final
    def get_input_schema(self) -> dict[str, object]:
        """This is the ONE place a tool schema is built and the ONE place
        ``act_summary`` is declared (``async`` is added on top by
        ``DelegateAbility`` for delegate tools only). ``final`` — do not override;
        enrich ``get_parameters()`` / ``get_summary()`` instead."""
        return {
            "name": self.get_name(),
            "description": self.get_summary(),
            "input_schema": self._inject_framework_fields(self.get_parameters()),
        }

    def _inject_framework_fields(self, params: dict[str, object]) -> dict[str, object]:
        """Deep-copies so a getter that returns a shared dict is never mutated."""
        schema = copy.deepcopy(params)
        properties = cast("dict[str, object]", schema.setdefault("properties", {}))
        required = cast("list[object]", schema.setdefault("required", []))

        properties["act_summary"] = dict(_ACT_SUMMARY_PROPERTY)
        if "act_summary" not in required:
            required.append("act_summary")

        return schema

    def classify_action(self, params: dict[str, object]) -> "str | None":
        """Derive the risk class the policy gate keys on, from the inputs alone.

        The dispatcher calls this ONCE, BEFORE ``PolicyManager.wrap``, and prefers
        its return value over a model-supplied ``action`` param when building the
        ``<tool>.<action>`` permission. This is the framework seam that lets a
        tool's permission be computed from what the call WOULD DO (e.g. bash
        inspecting the command string) instead of from a self-declared, and thus
        prompt-injectable, ``action`` field.

        The default returns ``None`` — "no opinion, use the ``action`` param (or
        the bare tool name when there is none)". Override on a tool whose risk
        class must be inferred rather than trusted; the override OWNS the
        classification (the model never self-declares risk for such a tool).
        """
        return None

    @classmethod
    def enrich_rich_payload(cls, payload: dict[str, object]) -> dict[str, object]:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on a
        subclass whose card must reflect state that changed AFTER the tool ran —
        the persisted ``tool_calls.result`` is a frozen snapshot.

        Called by ``rich_media_parser.parse()`` exactly once per rich segment.
        The tool call's wall-clock anchor is NOT passed here: it is generic
        segment metadata (``segment["created_at"]``), not per-ability state.
        """
        return payload

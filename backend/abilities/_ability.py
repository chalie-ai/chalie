"""The ``Ability`` ABC — describes and runs one tool, nothing more.

An Ability declares what a tool *is* through the ``NAME`` class constant and
four zero-arg getters (``get_summary`` / ``get_examples`` /
``get_search_tooltip`` / ``get_parameters``) and how it *runs* (``run()``).
The full LLM-facing tool
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
from configs.enums.ability_category import AbilityCategory
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
    # ability — by exact name or alias lookup through the registry. False removes
    # it from the discoverable roster entirely: it can ONLY reach a processor by
    # being pinned directly in that processor's ProcessorConfig.always_available.
    # Internal/framework tools (the compactors, thinking, find_tools itself, the
    # pattern writers, the raw web tools, memory) set this False; user-facing
    # tools leave it True.
    DISCOVERABLE: ClassVar[bool] = True

    # Alternate names a model may use to load this tool by exact match.
    # Consumed by find_tools discovery; empty default means the canonical
    # name is the only alias.
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ()

    # The heading this tool is listed under in the find_tools menu. REQUIRED on
    # every DISCOVERABLE ability — the registry raises at load time when one is
    # missing, so a new tool cannot silently join the roster with no heading to
    # render under. None is correct (and enforced-as-fine) only for
    # DISCOVERABLE=False abilities, which never appear in the menu at all.
    CATEGORY: ClassVar[AbilityCategory | None] = None

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

    # Param names exempt from the dispatcher pre-gate's non-empty requirement:
    # they are still required to be PRESENT in the input (``p not in params``
    # triggers ``missing-params``), but an empty string (``""``) is a legitimate
    # value for them. The default is empty — every param is non-empty by
    # default — so an ability with no ``ALLOW_EMPTY`` behaves exactly as before.
    ALLOW_EMPTY: ClassVar[tuple[str, ...]] = ()

    # Param names whose VALUES reach run() exactly as the model sent them — the
    # dispatch seam's scrub (leaked provider sentinel tokens out, surrounding
    # whitespace trimmed) is skipped for them. Declare a param here when the
    # tool STORES its value rather than interpreting it: file content,
    # replacement text. For those, a leading tab or a trailing newline is data
    # the model chose byte by byte, not noise to tidy away — trimming it is a
    # silent rewrite of the user's file. The default is empty, so every param is
    # scrubbed unless it opts out.
    VERBATIM: ClassVar[tuple[str, ...]] = ()

    # Per-action steer for actions whose SUCCESS body can carry prose written
    # OUTSIDE this conversation — a web page, a file, an email, an image's text,
    # another agent's synthesis. Keyed exactly like ACTION_REQUIRED: on the
    # resolved action, or on "" for a tool that has none. The dispatcher appends
    # the matching entry to that result's follow-up block, so the warning travels
    # WITH the payload instead of sitting far away in the system prompt where a
    # long result pushes it out of mind.
    #
    # Write the steer for THIS action: name what the content actually is, who
    # controls it, and what an attack through this particular channel looks like.
    # A generic paragraph repeated across every tool is the thing a model learns
    # to skip — and a page title reads differently from an inbox message, so the
    # warning that earns its place says which one this is.
    #
    # An action absent from the map gets NOTHING. That is the common case and it
    # is deliberate: `email.send` returns Chalie's own confirmation, `browser.fill`
    # returns mechanical state. Warning about those trains the model to ignore the
    # warning on `email.read`, which is where it matters.
    UNTRUSTED_CONTENT: ClassVar[dict[str, str]] = {}

    # The ability's typed input contract: every first-party ability sets its
    # ParamBag class here and run() receives an instance the dispatcher builds
    # via the bag's from_params factory — a bad input comes back from the bag
    # as an error ToolResult the dispatcher returns as-is, never an exception.
    # None is reserved for the synthetic _MCPAbility proxies (remote schema, no
    # compile-time bag): their run() receives the raw params dict.
    PARAMS: ClassVar["type[ParamBag] | None"] = None

    # The ability's canonical, immutable identifier — the key used by the
    # registry, the dispatcher, the policy gate, and the tool descriptor.
    # Subclasses MUST set this; the registry raises on missing or empty NAME.
    NAME: ClassVar[str]

    def __init__(self, mp: "MessageProcessor | None" = None) -> None:
        self.mp = mp
        # Set by DispatchService._run() immediately before run(): the flattened
        # client telemetry dict (location / locale / time / currency …) or None
        # when no client context is stored yet (fresh boot, no heartbeat).
        self.telemetry: "dict[str, object] | None" = None

    # ── Metadata getters — every concrete ability implements all four ──────────

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
            "name": self.NAME,
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

    def _append_active(self, names: list[str]) -> None:
        """Append tool names to ``mp.active_tools`` (skipping dupes) so they are
        live for the rest of this ACT turn. The activation seam shared by the
        discovery tools (``find_tools``, ``mcp_tools``); a no-op off-spine
        (``mp is None``)."""
        if not names:
            return
        proc = self.mp
        if proc is None:
            return
        active = cast("list[str] | None", getattr(proc, "active_tools", None))
        if active is None:
            return
        for name in names:
            if name not in active:
                active.append(name)

    def classify_action(self, params: dict[str, object]) -> "str | None":
        """Derive the risk class the policy gate keys on, from the inputs alone.

        The dispatcher calls this ONCE, BEFORE ``PolicyManager.authorize``, and prefers
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

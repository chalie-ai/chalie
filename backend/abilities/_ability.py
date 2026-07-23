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
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, cast

from typing_extensions import TypeVar

from abilities._result import ToolResult
from exceptions import ToolParamError

if TYPE_CHECKING:
    from datetime import datetime

    from contracts.params.param_bag import ParamBag
    from controllers.message_processor import MessageProcessor

# Sentinel distinguishing "no default supplied" from an explicit default of None
# in Ability.param(): a required param with no value raises; an optional one with
# default=None returns None.
_MISSING = object()

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

# What run() receives: the ability's typed ParamBag once migrated, the raw params
# dict until then. The PEP 696 default keeps every bare ``Ability`` annotation
# (registry, dispatcher, unmigrated subclasses) valid while the migration is in
# flight; a migrated ability declares ``Ability[ItsParamsBag]`` and mypy then
# enforces that its run() accepts exactly that bag.
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

    # The ability's typed input contract: a migrated ability sets its ParamBag
    # class here and run() receives an instance the dispatcher builds via the
    # bag's from_params factory — validation happens there (ToolParamError →
    # canonical dispatcher rendering). None (the default) = unmigrated: run()
    # receives the raw params dict and ACTION_REQUIRED / self.param() keep
    # gating it.
    PARAMS: ClassVar["type[ParamBag] | None"] = None

    def __init__(self, mp: "MessageProcessor | None" = None) -> None:
        self.mp = mp
        # Set by DispatchService._run() immediately before run(): the flattened
        # client telemetry dict (location / locale / time / currency …) or None
        # when no client context is stored yet (fresh boot, no heartbeat).
        self.telemetry: "dict[str, object] | None" = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # get_input_schema / param are the single assembler and the one sanctioned
        # param path — sealed. A subclass that redefines either would silently fork
        # the contract, so it is rejected at import. Subclasses enrich via
        # get_parameters() / get_summary() instead.
        for sealed in ("get_input_schema", "param"):
            if sealed in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must not override Ability.{sealed} — "
                    f"enrich get_parameters()/get_summary() instead"
                )
        # _inject_framework_fields is the single injection site — sealed against
        # arbitrary subclasses (redefining it would fork the act_summary/async
        # contract). The lone sanctioned override is DelegateAbility, which adds
        # the delegate-only ``async`` property on top of super()'s act_summary.
        if "_inject_framework_fields" in cls.__dict__ and cls.__name__ != "DelegateAbility":
            raise TypeError(
                f"{cls.__name__} must not override Ability._inject_framework_fields — "
                f"enrich get_parameters()/get_summary() instead"
            )

        # Synthetic proxies (e.g. _MCPAbility) source their metadata from a remote
        # schema and are constructed with arguments — skip metadata validation.
        if getattr(cls, "_SYNTHETIC", False):
            return
        # Abstract base mix-ins (e.g. SearchableAbility, which lists ABC directly
        # in its bases) are not concrete abilities — skip their metadata probe.
        # We deliberately do NOT consult __abstractmethods__ here: ABCMeta sets it
        # AFTER __init_subclass__ returns, so it is always unset at this point. A
        # concrete-looking subclass that still omits a getter is caught downstream
        # instead — get_examples / get_search_tooltip by the probe below (the
        # inherited abstract getter returns None and fails the shape check), and
        # get_name / get_summary / get_parameters / run at registry instantiation
        # (ABC blocks it).
        if ABC in cls.__bases__:
            return

        # Concrete ability: validate its metadata shape AT IMPORT, through the
        # getters, on a throwaway mp=None instance (deterministic at build time).
        # This keeps the "won't import with bad metadata" guarantee for EVERY
        # ability — including the never-indexed ones (thinking, the compactors)
        # that the search-index builder skips.
        probe = cls()
        examples = probe.get_examples()
        if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
            raise TypeError(f"{cls.__name__}.get_examples() must return list[str]")
        if not (6 <= len(examples) <= 8):
            raise TypeError(
                f"{cls.__name__}.get_examples() must return 6–8 entries, got {len(examples)}"
            )
        if cls.__module__.startswith("abilities.") and not probe.get_search_tooltip():
            raise TypeError(f"{cls.__name__}.get_search_tooltip() must be non-empty")
        follow_up = probe.get_follow_up(ToolResult.ok(""))
        if not isinstance(follow_up, str):
            raise TypeError(f"{cls.__name__}.get_follow_up() must return str")

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
        nudge. An override that needs data the result lacks MUST degrade to ``""``;
        it is probed at import on an empty ``ToolResult`` and so must never assume a
        shape. No ``self.mp`` — the nudge is request-agnostic.
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
        declared ``PARAMS`` bag — constructed (= validated) by the dispatcher —
        once migrated, or the raw params dict until then. Raise ``ToolParamError``
        (via a bag validator / ``self.param``) for bad inputs; the dispatcher
        renders it canonically. The ability NEVER formats the ``[tool(...)]`` wire
        envelope — the dispatcher owns it."""
        ...

    # ── The single, final tool-descriptor assembler ───────────────────────────

    @typing.final
    def get_input_schema(self) -> dict[str, object]:
        """This is the ONE place a tool schema is built and the ONE place
        ``act_summary`` is declared (``async`` is added on top by
        ``DelegateAbility`` for delegate tools only). ``final`` — sealed at import
        by ``__init_subclass__``."""
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

    # ── The one sanctioned param-handling path ─────────────────────────────────

    @typing.final
    def param(
        self,
        params: dict[str, object],
        name: str,
        *,
        default: object = _MISSING,
        choices: "tuple[object, ...] | None" = None,
        clamp: "tuple[object, ...] | None" = None,
        required: bool = False,
    ) -> object:
        """All failures raise ``ToolParamError`` — the dispatcher catches it and
        renders ``code=invalid-param`` with a ``hint`` and ``valid=`` so the model
        fixes the call without re-reading the schema. ``final`` — sealed at import
        by ``__init_subclass__``."""
        value = params.get(name)
        if value is None:
            if required:
                raise ToolParamError(
                    f"Required parameter '{name}' is missing.",
                    code="invalid-param",
                    hint=f"pass '{name}'.",
                )
            if default is _MISSING:
                return None
            return default

        if choices is not None and value not in choices:
            raise ToolParamError(
                f"'{value}' is not a valid value for '{name}'.",
                code="invalid-param",
                hint=f"choose one of: {', '.join(str(c) for c in choices)}.",
                valid=tuple(str(c) for c in choices),
            )

        if clamp is not None:
            lo, hi = clamp
            try:
                numeric = cast("Callable[[str], float]", type(lo))(cast("str", value))
            except (TypeError, ValueError):
                raise ToolParamError(
                    f"'{name}' must be a number.",
                    code="invalid-param",
                    hint=f"pass a number between {lo} and {hi}.",
                ) from None
            value = max(cast("float", lo), min(cast("float", hi), numeric))

        return value

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
    def enrich_rich_payload(cls, payload: dict[str, object], created_at: "datetime | str | None") -> dict[str, object]:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on
        subclasses whose card needs data that does NOT live in the LLM-visible
        tool result.

        Called by ``rich_media_parser.parse()`` exactly once per rich segment.
        """
        return payload

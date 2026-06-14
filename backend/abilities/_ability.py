"""The ``Ability`` ABC — describes and runs one tool, nothing more.

An Ability declares what a tool *is* through five zero-arg getters
(``get_name`` / ``get_summary`` / ``get_examples`` / ``get_search_tooltip`` /
``get_parameters``) and how it *runs* (``run()``). The full LLM-facing tool
descriptor is assembled in ONE place — the ``final`` ``get_input_schema()`` — which
is also the SINGLE site that injects the two framework fields (``act_summary``
and ``async``). Subclasses cannot override it; they only fill in the getters.

Dispatch — matching, binding, policy gating, execution, recording — lives in
``ToolDispatcher`` (``abilities/_dispatcher.py``), the single chokepoint every
ACT-loop tool call takes. This module imports nothing from the registry / policy
/ dispatcher, so ``_registry`` can import ``Ability`` here with no circular-import
dance.
"""

from __future__ import annotations

import copy
import typing
from abc import ABC, abstractmethod
from typing import ClassVar

from abilities._result import ToolParamError, ToolResult

# Sentinel distinguishing "no default supplied" from an explicit default of None
# in Ability.param(): a required param with no value raises; an optional one with
# default=None returns None.
_MISSING = object()

# The framework fields injected into EVERY tool descriptor by get_input_schema —
# the one place either is declared. act_summary is the per-call act-trail tooltip
# (always present, required); async is the per-call backgrounding flag, injected
# only on channels whose config sets SUPPORTS_ASYNC.
_ACT_SUMMARY_PROPERTY: dict = {
    "type": "string",
    "description": (
        "A ~3-10 word summary of what this specific tool call does, shown to"
        ' the user as a tooltip (e.g. "Searching for laptops in Malta",'
        ' "Looking up the weather in London").'
    ),
}
_ASYNC_PROPERTY: dict = {
    "type": "boolean",
    "default": False,
    "description": (
        "Run in the background instead of blocking this step. You get an "
        "immediate acknowledgement and the result is delivered on this channel "
        "when it completes. Use for long-running calls."
    ),
}


class Ability(ABC):
    """Base class for every dispatchable tool.

    A concrete Ability implements five getters (the metadata) plus ``run()`` (the
    behaviour). The getters read ``self.mp`` (the invoking MessageProcessor,
    constructor-injected) when a value depends on the live request; at
    ``self.mp is None`` — introspection / search-index build — they MUST return
    deterministic base text.

    Tool *scope* (always-available vs discoverable) lives on ProcessorConfig
    (always_available / discoverable / blocked). Whether a call blocks or runs in
    the background is a per-call decision (the framework ``async`` flag), not an
    ability-level trait.
    """

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

    def __init__(self, mp: "object | None" = None) -> None:
        self.mp = mp
        # Set by ToolDispatcher._run() immediately before run(): the flattened
        # client telemetry dict (location / locale / time / currency …) or None
        # when no client context is stored yet (fresh boot, no heartbeat).
        self.telemetry: "dict | None" = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # get_input_schema / _inject_framework_fields are the single assembler and
        # the single injection site — sealed. A subclass that redefines either
        # (e.g. to "also enrich the schema") would silently fork the async /
        # act_summary contract, so it is rejected at import. Subclasses enrich via
        # get_parameters() / get_summary() instead.
        for sealed in ("get_input_schema", "_inject_framework_fields", "param"):
            if sealed in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must not override Ability.{sealed} — "
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

    # ── Metadata getters — every concrete ability implements all five ──────────

    @abstractmethod
    def get_name(self) -> str:
        """The tool's stable identifier (the name the model calls)."""
        ...

    @abstractmethod
    def get_summary(self) -> str:
        """The tool description shown to the model AND the base text embedded for
        semantic search. Override to enrich at runtime (e.g. bash appends the cwd)
        — but gate enrichment on ``self.mp is not None`` so the build-time index
        stays deterministic."""
        ...

    @abstractmethod
    def get_examples(self) -> list[str]:
        """6–8 natural-language example invocations, embedded + FTS-indexed for
        find_tools search."""
        ...

    @abstractmethod
    def get_search_tooltip(self) -> str:
        """A short search-facing label for the tool."""
        ...

    @abstractmethod
    def get_parameters(self) -> dict:
        """The bare JSON-schema body for the tool's inputs
        (``{"type": "object", "properties": {...}, "required": [...]}``) WITHOUT
        the framework fields. Override to enrich at runtime (e.g. find_tools folds
        the discoverable-tools index into the ``select`` description) — gate on
        ``self.mp`` so the build-time schema stays deterministic."""
        ...

    # ── Instance method — the actual tool logic ───────────────────────────────

    @abstractmethod
    def run(self, params: dict) -> "ToolResult":
        """Execute the ability. Called by ToolDispatcher._run().

        The invoking MessageProcessor is available as ``self.mp`` (read
        ``self.mp.config.channel`` where the old signature passed ``channel``),
        and the flattened client telemetry as ``self.telemetry`` (or None).

        MUST return an :class:`abilities._result.ToolResult`, built ONLY via
        ``ToolResult.ok(body, *, rich=None, **meta)`` /
        ``ToolResult.err(message, *, code, hint=None, valid=(), **meta)``:
        - ``ok`` body is prose (``str``, shown verbatim) or structured data
          (``dict``/``list``, rendered as compact JSON); ``rich=`` is an optional
          card payload; ``**meta`` is the flat tag map.
        - ``err`` carries a stable kebab-case ``code``, a one-line ``hint``, and
          a ``valid`` tuple of acceptable actions/values.

        The ability NEVER formats the ``[tool(...)]`` wire envelope — the
        dispatcher (``ToolDispatcher._render``) owns it. Returning anything that
        is not a ``ToolResult`` HARD-FAILS as ``code=non-canonical-result``.
        Raise ``ToolParamError`` (via ``self.param``) for bad inputs; the
        dispatcher renders it canonically.

        Args:
            params: Input parameters from the LLM, framework keys stripped.

        Spec §5.
        """
        ...

    # ── The single, final tool-descriptor assembler ───────────────────────────

    @typing.final
    def get_input_schema(self) -> dict:
        """Assemble the full LLM-facing tool descriptor from the getters and inject
        the framework fields. This is the ONE place a tool schema is built and the
        ONE place ``act_summary`` + ``async`` are declared. ``final`` — sealed at
        import by ``__init_subclass__``.

        Spec §4.3.
        """
        return {
            "name": self.get_name(),
            "description": self.get_summary(),
            "input_schema": self._inject_framework_fields(self.get_parameters()),
        }

    def _inject_framework_fields(self, params: dict) -> dict:
        """Return a copy of *params* with the framework fields added: ``act_summary``
        (always, required) and ``async`` (iff this channel backgrounds — i.e. the
        invoking ``mp``'s config sets ``SUPPORTS_ASYNC``). Deep-copies so a getter
        that returns a shared dict is never mutated."""
        schema = copy.deepcopy(params)
        properties = schema.setdefault("properties", {})
        required = schema.setdefault("required", [])

        properties["act_summary"] = dict(_ACT_SUMMARY_PROPERTY)
        if "act_summary" not in required:
            required.append("act_summary")

        if getattr(getattr(self.mp, "config", None), "SUPPORTS_ASYNC", False):
            properties["async"] = dict(_ASYNC_PROPERTY)

        return schema

    # ── The one sanctioned param-handling path ─────────────────────────────────

    @typing.final
    def param(
        self,
        params: dict,
        name: str,
        *,
        default: object = _MISSING,
        choices: "tuple | None" = None,
        clamp: "tuple | None" = None,
        required: bool = False,
    ) -> object:
        """Resolve one input the model passed, with self-correcting failures.

        The canonical ``name`` (a :class:`abilities._params.Keys` constant) is
        matched; the first key present with a non-None value wins. Argument-key
        healing — a model addressing the parameter by a mangled (``source"``) or
        alias (``url`` for ``source``) spelling — happens ONCE upstream at the
        dispatch seam (``abilities._params.KeyHealer.heal``, fed by the shared
        :data:`abilities._params.VARIANTS` ladders), so by the time ``run()`` calls
        ``param`` the key is already canonical and this method only matches *name*.
        Missing → *default* (or a ``ToolParamError`` when *required* / no default).
        ``choices`` validates membership; ``clamp`` ``(lo, hi)`` bounds a numeric.
        All failures raise ``ToolParamError`` — the dispatcher catches it and
        renders ``code=invalid-param`` with a ``hint`` and ``valid=`` so the model
        fixes the call without re-reading the schema. ``final`` — sealed at import
        by ``__init_subclass__``.
        """
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
                numeric = type(lo)(value)
            except (TypeError, ValueError):
                raise ToolParamError(
                    f"'{name}' must be a number.",
                    code="invalid-param",
                    hint=f"pass a number between {lo} and {hi}.",
                ) from None
            value = max(lo, min(hi, numeric))

        return value

    def classify_action(self, params: dict) -> "str | None":
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
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on
        subclasses whose card needs data that does NOT live in the LLM-visible
        tool result.

        Called by ``rich_media_parser.parse()`` exactly once per rich segment.
        """
        return payload

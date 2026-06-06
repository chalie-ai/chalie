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

Spec: docs/superpowers/specs/2026-06-06-ability-schema-getters-design.md (TKT-837);
ACT Loop Orchestrator Refactor §5; eliminate-_base §4.1.
"""

from __future__ import annotations

import copy
import typing
from abc import ABC, abstractmethod

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

    Spec: §5 / AC-4; TKT-837.
    """

    # Constructor-injected, the invoking MessageProcessor (the "parent" of this
    # tool call). A tool reads ALL its context off this — self.mp.config.channel,
    # .config.policy_channel, ._uid, etc. This is the traceability spine: every
    # hop holds a real reference to its parent instead of reaching into a hidden
    # global. None only on a synthetic / introspection / build-time instance.
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
        for sealed in ("get_input_schema", "_inject_framework_fields"):
            if sealed in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must not override Ability.{sealed} — "
                    f"enrich get_parameters()/get_summary() instead"
                )

        # Synthetic proxies (e.g. _MCPAbility) source their metadata from a remote
        # schema and are constructed with arguments — skip metadata validation.
        if getattr(cls, "_SYNTHETIC", False):
            return
        # Abstract subclasses (still missing a getter or run) cannot be
        # instantiated and never reach the registry — skip them.
        if ABC in cls.__bases__ or getattr(cls, "__abstractmethods__", None):
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
    def run(self, params: dict) -> "dict | str":
        """Execute the ability. Called by ToolDispatcher._run().

        The invoking MessageProcessor is available as ``self.mp`` (read
        ``self.mp.config.channel`` where the old signature passed ``channel``),
        and the flattened client telemetry as ``self.telemetry`` (or None).

        Must return either:
        - A dict with 'status' and 'result' keys (canonical form).
        - A dict without 'status' (legacy — treated as success).
        - A plain string (treated as success result text).

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

        Spec §4.3 / TKT-837.
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

    @classmethod
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on
        subclasses whose card needs data that does NOT live in the LLM-visible
        tool result.

        Called by ``rich_media_parser.parse()`` exactly once per rich segment.
        """
        return payload

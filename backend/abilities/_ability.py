"""The ``Ability`` ABC — describes and runs one tool, nothing more.

An Ability declares what a tool *is* (NAME / SUMMARY / EXAMPLES / INPUT_SCHEMA /
SEARCH_TOOLTIP) and how it *runs* (``run()``). Dispatch — matching, binding,
policy gating, execution, recording — lives in ``ToolDispatcher``
(``abilities/_dispatcher.py``), the single chokepoint every ACT-loop tool call
takes. This module imports nothing from the registry / policy / dispatcher, so
``_registry`` can import ``Ability`` here with no circular-import dance.

Spec: ACT Loop Orchestrator Refactor §5; eliminate-_base §4.1.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import ClassVar


class Ability(ABC):
    """Base class for every dispatchable tool.

    Tool *scope* (always-available vs discoverable) lives on ProcessorConfig
    (always_available / discoverable / blocked fields).  Abilities only describe
    what the tool is (NAME / SUMMARY / EXAMPLES / INPUT_SCHEMA).  Whether a call
    blocks or runs in the background is a per-call decision (the framework
    ``async`` flag), not an ability-level trait.

    Spec: §5 / AC-4.
    """

    NAME: ClassVar[str]
    SUMMARY: ClassVar[str]
    EXAMPLES: ClassVar[list[str]]
    INPUT_SCHEMA: ClassVar[dict]
    SEARCH_TOOLTIP: ClassVar[str] = ""

    # Bound per-call by ToolDispatcher._bind(): the invoking MessageProcessor
    # (the "parent" of this tool call). A tool reads ALL its context off this —
    # self.MessageProcessor.config.channel, .config.policy_channel, ._uid, etc.
    # This is the traceability spine: every hop holds a real reference to its
    # parent instead of reaching into a hidden global. None only on a synthetic
    # / never-bound instance.
    MessageProcessor: "object | None" = None
    # Set by ToolDispatcher._run() immediately before run(): the flattened
    # client telemetry dict (location / locale / time / currency …) or None when
    # no client context is stored yet (fresh boot, no heartbeat).
    telemetry: "dict | None" = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Synthetic proxies (e.g. _MCPAbility) carry no EXAMPLES/SEARCH_TOOLTIP
        # and are constructed with arguments — skip all validation for them.
        if getattr(cls, "_SYNTHETIC", False):
            return
        # Abstract subclasses (marked with abstractmethod) are exempt.
        if ABC in cls.__bases__:
            return
        # Skip intermediate abstract classes that still have abstract methods.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("NAME", "INPUT_SCHEMA"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        for attr in ("SUMMARY", "EXAMPLES"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if not isinstance(cls.EXAMPLES, list) or not all(
            isinstance(e, str) for e in cls.EXAMPLES
        ):
            raise TypeError(f"{cls.__name__}.EXAMPLES must be list[str]")
        if not (6 <= len(cls.EXAMPLES) <= 8):
            raise TypeError(
                f"{cls.__name__}.EXAMPLES must have 6–8 entries, got {len(cls.EXAMPLES)}"
            )
        if (
            getattr(cls, "__module__", "").startswith("abilities.")
            and not getattr(cls, "SEARCH_TOOLTIP", "")
        ):
            raise TypeError(f"{cls.__name__} must define a non-empty SEARCH_TOOLTIP")

    # ── Instance method — the actual tool logic ───────────────────────────────

    @abstractmethod
    def run(self, params: dict) -> "dict | str":
        """Execute the ability.  Called by ToolDispatcher._run().

        Override this method on every Ability subclass — it is the single tool
        entrypoint. The invoking MessageProcessor is available as
        ``self.MessageProcessor`` (read ``self.MessageProcessor.config.channel``
        where the old signature passed ``channel``), and the flattened client
        telemetry as ``self.telemetry`` (or None).

        Must return either:
        - A dict with 'status' and 'result' keys (canonical form).
        - A dict without 'status' (legacy — treated as success).
        - A plain string (treated as success result text).

        Args:
            params: Input parameters from the LLM, framework keys stripped.

        Returns:
            dict when the result is structured data, or str for plain text.

        Spec §5.
        """
        ...

    # ── Schema hooks ──────────────────────────────────────────────────────────

    def get_description(self) -> str:
        """Return the tool description for LLM tool presentation.

        Override to enrich the description at runtime (e.g. find_tools
        appends a discoverable-tools index).  The default returns SUMMARY.
        """
        return self.SUMMARY

    def get_input_schema(self, mp=None) -> dict:
        """Return the INPUT_SCHEMA for LLM tool presentation, with the framework
        ``async`` property injected on channels that support backgrounding.

        ``mp`` is the invoking MessageProcessor (threaded by ``build_tools``
        because that path operates on the unbound registry singleton).  The
        ``async`` boolean is injected **iff** ``mp`` is known and its config sets
        ``SUPPORTS_ASYNC`` (only the user channel today, §4.8d); elsewhere — and
        when ``mp`` is None (synchronous is the safe default) — the property is
        omitted, so the model cannot pick async and every tool is synchronous.

        This is the ONLY place ``async`` is declared.  Overrides that enrich the
        schema MUST start from ``super().get_input_schema(mp)`` so this gate
        applies uniformly (§4.1).

        Spec §4.0 / §4.1.
        """
        if mp is None or not getattr(getattr(mp, "config", None), "SUPPORTS_ASYNC", False):
            return self.INPUT_SCHEMA
        schema = copy.deepcopy(self.INPUT_SCHEMA)
        schema.setdefault("properties", {})["async"] = {
            "type": "boolean",
            "default": False,
            "description": (
                "Run in the background instead of blocking this step. You get an "
                "immediate acknowledgement and the result is delivered on this "
                "channel when it completes. Use for long-running calls."
            ),
        }
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

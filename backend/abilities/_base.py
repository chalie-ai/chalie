"""Ability base class — the single entry point for all tool calls.

Every tool call from the ACT loop enters through ``Ability.use()``, which
matches the handler, gates it through ``PolicyManager.wrap``, runs it via
``Ability.execute``, and records the outcome.  This is the ONLY path from
MessageProcessor._loop() to ability execution.

Spec: ACT Loop Orchestrator Refactor §5, §7b; PolicyManager redesign (TKT-797).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import uuid4

# ClientContext owns the per-request client-telemetry snapshot that
# _load_tool_telemetry() flattens onto ability.telemetry before run().
from services.client_context import ClientContext
# ActEventEmitter owns the broadcast_to gate for ACT tool-start/end events.
from abilities._event_emitter import ActEventEmitter


# These module-level names are populated at the bottom of this file (after the
# Ability class is defined) so that unittest.mock.patch("abilities._base.X")
# can intercept them from tests.  The imports are deferred to the END of the
# module to avoid circular-import issues (_registry imports Ability from here).
AbilityRegistry: object = None  # type: ignore[assignment]
PolicyManager: object = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _load_tool_telemetry() -> "dict | None":
    """Pull a flattened telemetry dict for tool dispatch.

    Returns None when no client context is stored yet (fresh boot, no
    heartbeat) so abilities can fall back gracefully. Delegates the snapshot to
    the ``ClientContext`` value object — this is just the dict adapter.
    """
    ctx = ClientContext.current()
    return ctx.as_dict() if ctx else None


def _run_ability(ability: "Ability", params: dict) -> dict:
    """Execute ability.run() synchronously and normalise the result.

    Loads the flattened client telemetry onto ``ability.telemetry`` just before
    run(); the ability reads its parent off ``self.MessageProcessor``.

    There is no wall-clock bound — an ability runs to completion. The framework
    never abandons a tool call: the only things that stop a running tool are
    cooperative cancellation (cancel_event) and the network-level I/O timeouts
    inside individual abilities.

    Returns a result dict with 'status' and 'result' keys.

    Spec §5 / I13.
    """
    try:
        ability.telemetry = _load_tool_telemetry()
    except Exception:  # noqa: BLE001
        ability.telemetry = None

    try:
        raw = ability.run(params)
    except Exception as exc:  # noqa: BLE001
        # VaultLockedError gets a friendlier message.
        try:
            from services.vault_service import VaultLockedError  # noqa: PLC0415
            if isinstance(exc, VaultLockedError):
                return {
                    "status": "error",
                    "result": (
                        "This function is currently unavailable. "
                        "The vault is locked. "
                        "Notify the user that you could not complete this action "
                        "because they were logged out"
                    ),
                }
        except ImportError:
            pass
        return {"status": "error", "result": f"Error: {exc}"}

    return _normalise_run_result(raw)


def _normalise_run_result(raw: object) -> dict:
    """Coerce the raw return value of ability.run() into a {'status','result'} dict."""
    if isinstance(raw, dict):
        if "status" in raw:
            # Already in canonical form.
            result_val = raw.get("result", "")
            return {"status": raw["status"], "result": str(result_val) if result_val is not None else ""}
        # Legacy dict without status — treat as success.
        return {"status": "success", "result": raw}
    if isinstance(raw, str):
        return {"status": "success", "result": raw}
    return {"status": "success", "result": str(raw) if raw is not None else ""}


class Ability(ABC):
    """Base class for every dispatchable tool.

    ``use()`` is the single chokepoint for all tool calls from the ACT loop:
    match → resolve permission → PolicyManager.wrap(execute) → record. The
    gated work runs in ``execute()`` (emit → run → return).

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

    # Bound per-call by Ability._bind(): the invoking MessageProcessor (the
    # "parent" of this tool call). A tool reads ALL its context off this —
    # self.MessageProcessor.config.channel, .config.policy_channel, ._uid, etc.
    # This is the traceability spine: every hop holds a real reference to its
    # parent instead of reaching into a hidden global. None only on a synthetic
    # / never-bound instance.
    MessageProcessor: "object | None" = None
    # Set by _run_ability() immediately before run(): the flattened client
    # telemetry dict (location / locale / time / currency …) or None when no
    # client context is stored yet (fresh boot, no heartbeat).
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

    # ── The single tool-call entry point ──────────────────────────────────────

    @staticmethod
    def use(mp: object, tool_name: str, params: dict) -> str:
        """The one path every ACT-loop tool call takes.

        match → resolve permission (inline) → PolicyManager.wrap(execute) →
        record → return a STRING. Records EVERY outcome (allow result, block,
        unknown) so the rendered trail tells the model what happened and it does
        not retry a blocked tool forever. No cancel check — the loop guards
        cancel_event one line before calling this. Spec §5 / TKT-797.
        """
        if AbilityRegistry is None or PolicyManager is None:
            _populate_module_aliases()

        from services.message_processor import _sanitize_llm_args  # noqa: PLC0415

        params = _sanitize_llm_args(dict(params))
        act_summary = params.pop("act_summary", None)
        config = getattr(mp, "config", None)

        ability = Ability._bind(mp, tool_name)
        if ability is None:
            result_text = f"Unknown tool: {tool_name}"
        else:
            action = params.get("action")
            permission = f"{tool_name}.{action}" if action else tool_name
            result_text = PolicyManager.wrap(
                channel=getattr(config, "policy_channel", None),
                permission=permission,
                callback=lambda: ability.execute(params, act_summary),
            )

        from services.act_trail import ActTrail  # noqa: PLC0415
        ActTrail().record(
            tool_name=tool_name,
            params=params,
            result=result_text,
            transcript_id=getattr(mp, "uid", None),
            ephemeral=True,
        )
        return result_text

    @staticmethod
    def match(tool_name: str) -> "Ability | None":
        """Resolve a tool name to its handler. Native → registry; _mcp_* → a
        synthetic _MCPAbility proxy (so MCP flows through the same gate); unknown
        native name → None (use() turns this into an 'Unknown tool' string)."""
        if AbilityRegistry is None:
            _populate_module_aliases()
        if tool_name.startswith("_mcp_"):
            return _MCPAbility(tool_name)
        try:
            return AbilityRegistry.get(tool_name)
        except KeyError:
            return None

    @staticmethod
    def _bind(mp: object, tool_name: str) -> "Ability | None":
        """Resolve *tool_name* to a FRESH per-call Ability instance bound to *mp*.

        Native abilities live in the registry as singletons; binding *mp* onto a
        shared instance would race across concurrent turns. So every call gets
        its own instance (abilities are stateless, no custom ``__init__``) with
        the invoking MessageProcessor attached as ``self.MessageProcessor`` — the
        parent the tool reads its context from. ``_mcp_*`` names get a fresh
        ``_MCPAbility`` proxy (already per-call). Unknown native name → None.
        """
        if AbilityRegistry is None:
            _populate_module_aliases()
        if tool_name.startswith("_mcp_"):
            ability: "Ability" = _MCPAbility(tool_name)
        else:
            try:
                template = AbilityRegistry.get(tool_name)
            except KeyError:
                return None
            ability = type(template)()
        ability.MessageProcessor = mp
        return ability

    # ── Instance method — the actual tool logic ───────────────────────────────

    @abstractmethod
    def run(self, params: dict) -> "dict | str":
        """Execute the ability.  Called by Ability.execute() via _run_ability().

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

    # ── Self-scaffolding executor — the allow-path callback use() hands to wrap ──

    def execute(self, params: dict, act_summary: "str | None" = None) -> str:
        """Emit → (async-decision) → run → emit. Called ONLY on the allow path
        (use() passes this as wrap's callback). The per-call ``async`` flag —
        popped here, a framework key never passed to run() — decides whether the
        real work blocks this ACT iteration or runs on a background thread; run()
        is identical either way. Owns contextvar copying for async delivery,
        VaultLockedError handling, and result normalisation. Returns the result
        text STRING. Recording is use()'s job, not ours.

        Reads the bound parent off ``self.MessageProcessor`` (set by _bind()).
        act_summary (popped from params by use()) is the WS tooltip; it is NOT a
        run() argument. Spec §4.0 / §5 / D5.
        """
        run_async = bool(params.pop("async", False))
        config = getattr(self.MessageProcessor, "config", None)
        emitter = ActEventEmitter(config)
        call_id = uuid4().hex[:12]
        emitter.emit({
            "type": "act_tool_start",
            "name": self.NAME,
            "id": call_id,
            "summary": act_summary,
        })

        if run_async:
            # AsyncDelegateRunner owns the daemon-thread lifecycle + the captured
            # mp it delivers through; it returns the placeholder immediately so
            # this ACT iteration is never blocked (spec §4.0 / §4.4).
            from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415
            placeholder = async_delegate_runner.spawn(self, params, self.MessageProcessor)
            result = {"status": "success", "result": placeholder}
        else:
            result = _run_ability(self, params)

        ok = result.get("status") != "error"
        emitter.emit({"type": "act_tool_end", "name": self.NAME, "id": call_id, "ok": ok})
        return str(result.get("result", ""))

    # ── Schema hooks (unchanged) ──────────────────────────────────────────────

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
        import copy as _copy  # noqa: PLC0415
        schema = _copy.deepcopy(self.INPUT_SCHEMA)
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


# ── Module-level aliases for patchability ─────────────────────────────────────
#
# Populated AFTER the Ability class is fully defined to avoid the circular
# import that would occur if imported at the top of the file (_registry.py
# imports Ability from here).
#
# unittest.mock.patch("abilities._base.AbilityRegistry.get", ...) works because
# patch targets the name in this module's namespace and then patches the
# attribute on the object it finds there.

def _populate_module_aliases() -> None:
    global AbilityRegistry, PolicyManager
    try:
        from abilities._registry import AbilityRegistry as _AR  # noqa: PLC0415
        AbilityRegistry = _AR
    except ImportError:
        pass
    try:
        from services.policy_manager import PolicyManager as _PM  # noqa: PLC0415
        PolicyManager = _PM
    except ImportError:
        pass


_populate_module_aliases()


# Imported at the bottom (after Ability is defined) so _mcp_ability's
# ``from abilities._base import Ability`` resolves without a circular import.
# match()/_bind() reference _MCPAbility as a module global at call time.
from abilities._mcp_ability import _MCPAbility  # noqa: E402

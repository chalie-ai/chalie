"""``ToolDispatcher`` — the single chokepoint for every ACT-loop tool call.

Every tool call from the ACT loop enters through ``ToolDispatcher(mp).dispatch()``:
match the handler → bind a fresh per-call instance to the invoking ``mp`` →
gate it through ``PolicyManager.wrap`` → run it via ``_execute`` → record the
outcome on the act-trail → return a STRING. This is the ONLY path from
``MessageProcessor._loop()`` to ability execution.

The dispatcher is bound to the invoking MessageProcessor (``self._mp``) — it
imports ``AbilityRegistry`` / ``PolicyManager`` normally (neither imports it
back, so there is no circular-import dance and no alias hack). ``_MCPAbility``
is imported normally too. Only ``_sanitize_llm_args`` (defined in
``message_processor``, which imports this module) and the async runner are
deferred to break the import cycle.

Spec: ACT Loop Orchestrator Refactor §5, §7b; eliminate-_base §4.2; TKT-797.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

# AbilityRegistry resolves native tool names; PolicyManager gates the allow path.
# Neither imports this module, so both import normally — no alias/patch hack.
from abilities._event_emitter import ActEventEmitter
from abilities._mcp_ability import _MCPAbility
from abilities._registry import AbilityRegistry
# ClientContext owns the per-request client-telemetry snapshot that _run()
# flattens onto ability.telemetry before run().
from services.client_context import ClientContext
from services.policy_manager import PolicyManager

if TYPE_CHECKING:
    # Annotation-only: dispatch resolves abilities through the registry, never
    # by constructing Ability directly.
    from abilities._ability import Ability

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """Dispatches one tool call on behalf of the invoking MessageProcessor.

    ``dispatch()`` is the single chokepoint: match → bind → gate
    (PolicyManager.wrap) → execute → record. The gated work runs in
    ``_execute()`` (emit → async-decision → run → emit).

    Spec §5 / AC-4.
    """

    def __init__(self, mp: object) -> None:
        self._mp = mp

    # ── The single tool-call entry point ──────────────────────────────────────

    def dispatch(self, tool_name: str, params: dict) -> str:
        """The one path every ACT-loop tool call takes.

        match → bind → resolve permission (inline) → PolicyManager.wrap(execute)
        → record → return a STRING. Records EVERY outcome (allow result, block,
        unknown) so the rendered trail tells the model what happened and it does
        not retry a blocked tool forever. No cancel check — the loop guards
        cancel_event one line before calling this. Spec §5 / TKT-797.
        """
        from services.message_processor import _sanitize_llm_args  # noqa: PLC0415

        params = _sanitize_llm_args(dict(params))
        act_summary = params.pop("act_summary", None)
        config = getattr(self._mp, "config", None)

        ability = self._bind(tool_name)
        if ability is None:
            result_text = f"Unknown tool: {tool_name}"
        else:
            action = params.get("action")
            permission = f"{tool_name}.{action}" if action else tool_name
            result_text = PolicyManager.wrap(
                channel=getattr(config, "policy_channel", None),
                permission=permission,
                callback=lambda: self._execute(ability, params, act_summary),
            )

        from services.act_trail import ActTrail  # noqa: PLC0415
        ActTrail().record(
            tool_name=tool_name,
            params=params,
            result=result_text,
            transcript_id=getattr(self._mp, "uid", None),
            ephemeral=True,
        )
        return result_text

    # ── Resolution ─────────────────────────────────────────────────────────────

    def _match(self, tool_name: str) -> "Ability | None":
        """Resolve a tool name to its handler. Native → the registry singleton
        (a shared template); _mcp_* → a fresh synthetic _MCPAbility proxy (so MCP
        flows through the same gate); unknown native name → None (dispatch()
        turns this into an 'Unknown tool' string)."""
        if tool_name.startswith("_mcp_"):
            return _MCPAbility(tool_name)
        try:
            return AbilityRegistry.get(tool_name)
        except KeyError:
            return None

    def _bind(self, tool_name: str) -> "Ability | None":
        """Resolve *tool_name* to a FRESH per-call Ability instance bound to the
        invoking ``mp``.

        Native abilities live in the registry as singletons; binding *mp* onto a
        shared instance would race across concurrent turns. So every call gets
        its own instance (abilities are stateless, no custom ``__init__``) with
        the invoking MessageProcessor attached as ``self.MessageProcessor`` — the
        parent the tool reads its context from. ``_mcp_*`` names resolve to a
        proxy that is already per-call, so it is bound as-is. Unknown name → None.
        """
        ability = self._match(tool_name)
        if ability is None:
            return None
        # The registry returns a shared singleton template for native tools —
        # instantiate a fresh copy so the per-call mp binding never races. The
        # _mcp_* proxy is already a fresh per-call instance (custom __init__).
        if not tool_name.startswith("_mcp_"):
            ability = type(ability)()
        ability.MessageProcessor = self._mp
        return ability

    # ── Self-scaffolding executor — the allow-path callback dispatch() hands to wrap ──

    def _execute(self, ability: "Ability", params: dict, act_summary: "str | None" = None) -> str:
        """Emit → (async-decision) → run → emit. Called ONLY on the allow path
        (dispatch() passes this as wrap's callback). The per-call ``async`` flag —
        popped here, a framework key never passed to run() — decides whether the
        real work blocks this ACT iteration or runs on a background thread; run()
        is identical either way. Returns the result text STRING. Recording is
        dispatch()'s job, not ours.

        Reads the bound parent off ``ability.MessageProcessor`` (set by _bind()).
        act_summary (popped from params by dispatch()) is the WS tooltip; it is
        NOT a run() argument. Spec §4.0 / §4.2 / D5.
        """
        run_async = bool(params.pop("async", False))
        config = getattr(self._mp, "config", None)
        emitter = ActEventEmitter(config)
        call_id = uuid4().hex[:12]
        emitter.emit({
            "type": "act_tool_start",
            "name": ability.NAME,
            "id": call_id,
            "summary": act_summary,
        })

        if run_async:
            # AsyncDelegateRunner owns the daemon-thread lifecycle + the captured
            # mp it delivers through; it returns the placeholder immediately so
            # this ACT iteration is never blocked (spec §4.0 / §4.4).
            from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415
            placeholder = async_delegate_runner.spawn(ability, params, self._mp)
            result = {"status": "success", "result": placeholder}
        else:
            result = self._run(ability, params)

        ok = result.get("status") != "error"
        emitter.emit({"type": "act_tool_end", "name": ability.NAME, "id": call_id, "ok": ok})
        return str(result.get("result", ""))

    # ── Synchronous run primitive (shared by inline dispatch + AsyncDelegateRunner) ──

    @staticmethod
    def _run(ability: "Ability", params: dict) -> dict:
        """Execute ability.run() synchronously and normalise the result.

        Loads the flattened client telemetry onto ``ability.telemetry`` just
        before run(); the ability reads its parent off ``self.MessageProcessor``.

        There is no wall-clock bound — an ability runs to completion. The
        framework never abandons a tool call: the only things that stop a running
        tool are cooperative cancellation (cancel_event) and the network-level
        I/O timeouts inside individual abilities.

        Returns a result dict with 'status' and 'result' keys.

        Spec §4.2 / I13.
        """
        try:
            ctx = ClientContext.current()
            ability.telemetry = ctx.as_dict() if ctx else None
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

        return ToolDispatcher._normalise(raw)

    @staticmethod
    def _normalise(raw: object) -> dict:
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

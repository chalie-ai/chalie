"""``ToolDispatcher`` — the single chokepoint for every tool call in a turn.

Every tool call in the turn chain enters through ``ToolDispatcher(mp).dispatch()``:
match the handler → bind a fresh per-call instance to the invoking ``mp`` →
gate it through ``PolicyManager.wrap`` → run it via ``_execute`` → render the
sealed wire envelope → record the outcome on the act-trail → return a STRING.
This is the ONLY path from ``MessageProcessor._step()`` to ability execution,
and the ONLY place the ``[<tool>(status=…)]\n…\n[end:<tool>]`` envelope is
formatted (``_render``).

``run()`` MUST return an :class:`abilities._result.ToolResult`.  Anything else is
a hard error: the dispatcher renders ``status=error, code=non-canonical-result``
naming the ability and the offending type, and logs at ERROR.  There is no
legacy-dict / plain-str compatibility shim — divergence fails loudly.

The dispatcher is bound to the invoking MessageProcessor (``self._mp``) — it
imports ``AbilityRegistry`` / ``PolicyManager`` normally (neither imports it
back, so there is no circular-import dance and no alias hack). ``_MCPAbility``
is imported normally too. Only ``_sanitize_llm_args`` (defined in
``message_processor``, which imports this module) and the async runner are
deferred to break the import cycle.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

# AbilityRegistry resolves native tool names; PolicyManager gates the allow path.
# Neither imports this module, so both import normally — no alias/patch hack.
from abilities._event_emitter import ActEventEmitter
from abilities._mcp_ability import _MCPAbility
from abilities._params import KeyHealer
from abilities._registry import AbilityRegistry
from abilities._result import ToolParamError, ToolResult
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
    (PolicyManager.wrap) → execute → render → record. The gated work runs in
    ``_execute()`` (emit → async-decision → run → render → emit).

    Spec §5 / AC-4.
    """

    def __init__(self, mp: object, key_healer: "KeyHealer | None" = None) -> None:
        self._mp = mp
        # The key healer is injected (DIP); the shared default heals against the
        # production VARIANTS registry. A test can supply a probe healer without
        # touching the registry.
        self._key_healer = key_healer or KeyHealer()

    # ── The single tool-call entry point ──────────────────────────────────────

    def dispatch(self, tool_name: str, params: dict) -> str:
        """The one path every ACT-loop tool call takes.

        match → bind → ACTION_REQUIRED pre-gate → resolve permission (inline) →
        PolicyManager.wrap(execute)
        → record → return a STRING. Records EVERY outcome (allow result, block,
        unknown) so the rendered trail tells the model what happened and it does
        not retry a blocked tool forever. No cancel check — the loop guards
        cancel_event one line before calling this. Spec §5.
        """
        from services.message_processor import _sanitize_llm_args  # noqa: PLC0415

        params = _sanitize_llm_args(dict(params))
        act_summary = params.pop("act_summary", None)
        config = getattr(self._mp, "config", None)

        ability = self._bind(tool_name)
        if ability is None:
            result_text = f"Unknown tool: {tool_name}"
        else:
            # Heal model-mangled argument KEYS against the tool's declared schema
            # before any gate or run() reads them: a stray-quote/case/alias key
            # (e.g. 'source"', 'URL', or read's 'url' for 'source') is rewritten
            # to its canonical parameter, so the ACTION_REQUIRED pre-gate and the
            # ability see the real key instead of a corrupt one that would bounce
            # on a spurious required-field error (TKT-963 / TKT-964). Defensive: a
            # registry/schema fault must never break dispatch — on failure the raw
            # params flow through unchanged.
            try:
                params = self._key_healer.heal(params, ability.get_parameters())
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[ToolDispatcher] key canonicalisation failed for %s — "
                    "proceeding with raw params", tool_name,
                )
            if (pre := self._prevalidate(ability, params)) is not None:
                # ACTION_REQUIRED pre-gate fires BEFORE the permission is formed: a
                # hallucinated action would otherwise lazily seed a bogus
                # '<tool>.<action>' ask row and freeze the turn waiting for human
                # approval. Malformed input never reaches the policy gate or run().
                result_text = self._render(tool_name, pre, None)
            else:
                # The risk class the gate keys on is derived from the inputs via the
                # ability's classify_action hook (default None), NOT trusted from a
                # model-supplied 'action' — a self-declared action is prompt-injectable
                # and must never decide the permission. Fall back to the action param
                # only when the tool offers no classification.
                classified = ability.classify_action(params)
                action = classified if classified is not None else params.get("action")
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
            # Anchors to the assistant step row that emitted this call (mp.anchor,
            # set by _store_row before the step's tools dispatch); its turn is
            # derived by joining transcript on (channel, turn_id) at read time.
            # Falls back to the input row (mp.uid) for framework / turn-zero /
            # compaction calls that fire before any step row exists.
            transcript_id=getattr(self._mp, "anchor", None) or getattr(self._mp, "uid", None),
            # The live blue-box summary, persisted so the chat refresh can
            # re-render the chip's summary instead of dropping it on reload.
            summary=act_summary,
        )
        return result_text

    # ── Resolution ─────────────────────────────────────────────────────────────

    def _bind(self, tool_name: str) -> "Ability | None":
        """Resolve *tool_name* to a FRESH per-call Ability instance bound to the
        invoking ``mp`` (passed to ``Ability.__init__`` → ``self.mp``).

        Native abilities live in the registry as singletons; binding *mp* onto a
        shared instance would race across concurrent turns. So every call gets
        its own instance (abilities are stateless, no custom ``__init__``)
        constructed with the invoking MessageProcessor — the parent the tool
        reads its context from via ``self.mp``. ``_mcp_*`` names resolve to a
        fresh synthetic ``_MCPAbility`` proxy, also mp-bound. Unknown native name
        → None (dispatch() turns this into an 'Unknown tool' string).
        """
        if tool_name.startswith("_mcp_"):
            return _MCPAbility(tool_name, mp=self._mp)
        try:
            template = AbilityRegistry.get(tool_name)
        except KeyError:
            return None
        return type(template)(mp=self._mp)

    # ── Self-scaffolding executor — the allow-path callback dispatch() hands to wrap ──

    def _execute(self, ability: "Ability", params: dict, act_summary: "str | None" = None) -> str:
        """Emit → (action pre-validation) → (async-decision) → run → render →
        emit. Called ONLY on the allow path (dispatch() passes this as wrap's
        callback). The per-call ``async`` flag — popped here, a framework key
        never passed to run() — decides whether the real work blocks this ACT
        iteration or runs on a background thread; run() is identical either way.
        Returns the rendered envelope STRING. Recording is dispatch()'s job, not
        ours.

        Reads the bound parent off ``ability.mp`` (set by _bind()). act_summary
        (popped from params by dispatch()) is the WS tooltip; it is NOT a run()
        argument. Spec §4.0 / §4.2 / D5.
        """
        run_async = bool(params.pop("async", False))
        config = getattr(self._mp, "config", None)
        emitter = ActEventEmitter(config)
        call_id = uuid4().hex[:12]
        tool_name = ability.get_name()
        emitter.emit({
            "type": "act_tool_start",
            "name": tool_name,
            "id": call_id,
            "summary": act_summary,
        })

        if run_async:
            # AsyncDelegateRunner owns the daemon-thread lifecycle + the captured
            # mp it delivers through; it returns the placeholder immediately so
            # this ACT iteration is never blocked (spec §4.0 / §4.4). The
            # placeholder is prose the model reads while the real work runs.
            from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415
            placeholder = async_delegate_runner.spawn(ability, params, self._mp)
            tr = ToolResult.ok(str(placeholder))
        else:
            tr = self._run(ability, params)

        # Rich-media ordinal is assigned ONLY when the invoking mp broadcasts to
        # the user. Subagents / background channels never get a card: their
        # natural-language synthesis is consumed by the parent, so a span emitted
        # at that hop has no tool_calls row paired to it. This is the single
        # physical chokepoint that gates the entire card path.
        ordinal = None
        if tr.rich is not None and getattr(config, "broadcast_to", None) == "user":
            ordinal = self._next_ordinal(tool_name)

        rendered = self._render(tool_name, tr, ordinal)
        emitter.emit({"type": "act_tool_end", "name": tool_name, "id": call_id, "ok": tr.status == "success"})
        return rendered

    # ── Action pre-validation (ACTION_REQUIRED) ────────────────────────────────

    @staticmethod
    def _prevalidate(ability: "Ability", params: dict) -> "ToolResult | None":
        """Validate the ability's ACTION_REQUIRED map BEFORE the policy gate.

        An empty map (the default) means no pre-validation — unmigrated tools are
        untouched. Otherwise: an unknown action → ``code=unknown-action`` with
        ``valid=`` the action-map keys; a known action missing required params →
        a SINGLE ``code=missing-params`` error naming ALL missing params.

        The map key ``""`` covers action-less tools.
        """
        action_map = getattr(ability, "ACTION_REQUIRED", None) or {}
        if not action_map:
            return None

        action = params.get("action")
        key = action if action is not None else ""
        if key not in action_map:
            return ToolResult.err(
                f"Unknown action {action!r}." if action is not None
                else "This tool requires an 'action'.",
                code="unknown-action",
                valid=tuple(k for k in action_map if k),
            )

        missing = [p for p in action_map[key] if not params.get(p)]
        if missing:
            named = ", ".join(missing)
            return ToolResult.err(
                f"Missing required parameter(s) for action {key!r}: {named}."
                if key else f"Missing required parameter(s): {named}.",
                code="missing-params",
                valid=tuple(action_map[key]),
            )
        return None

    # ── Synchronous run primitive (shared by inline dispatch + AsyncDelegateRunner) ──

    @staticmethod
    def _run(ability: "Ability", params: dict) -> ToolResult:
        """Execute ability.run() synchronously and enforce the ToolResult contract.

        Loads the flattened client telemetry onto ``ability.telemetry`` just
        before run(); the ability reads its parent off ``self.mp``.

        There is no wall-clock bound — an ability runs to completion. The
        framework never abandons a tool call: the only things that stop a running
        tool are cooperative cancellation (cancel_event) and the network-level
        I/O timeouts inside individual abilities.

        Returns a :class:`ToolResult`. A ``ToolParamError`` raised from run() is
        rendered canonically with its code/hint/valid. A raised exception becomes
        ``code=unhandled-exception`` (with the VaultLockedError friendly message).
        A non-ToolResult return value HARD-FAILS as ``code=non-canonical-result``.

        Spec §4.2 / I13.
        """
        try:
            ctx = ClientContext.current()
            ability.telemetry = ctx.as_dict() if ctx else None
        except Exception:  # noqa: BLE001
            ability.telemetry = None

        try:
            raw = ability.run(params)
        except ToolParamError as exc:
            return ToolResult.err(exc.message, code=exc.code, hint=exc.hint, valid=exc.valid)
        except Exception as exc:  # noqa: BLE001
            # VaultLockedError gets a friendlier message.
            try:
                from services.vault_service import VaultLockedError  # noqa: PLC0415
                if isinstance(exc, VaultLockedError):
                    return ToolResult.err(
                        "This function is currently unavailable. "
                        "The vault is locked. "
                        "Notify the user that you could not complete this action "
                        "because they were logged out",
                        code="vault-locked",
                    )
            except ImportError:
                pass
            return ToolResult.err(f"Error: {exc}", code="unhandled-exception")

        if not isinstance(raw, ToolResult):
            offending = type(raw).__name__
            logger.error(
                "[ToolDispatcher] %s.run() returned a non-ToolResult (%s) — every "
                "ability must return abilities._result.ToolResult via ok()/err().",
                ability.get_name(), offending,
            )
            return ToolResult.err(
                f"{ability.get_name()} returned a non-canonical result of type "
                f"{offending}; abilities must return a ToolResult.",
                code="non-canonical-result",
            )
        return raw

    # ── The single wire-envelope renderer ──────────────────────────────────────

    def _next_ordinal(self, tool_name: str) -> int:
        """Per-turn, per-tool ordinal counter held on the invoking mp.

        Keyed by tool name and scoped to the mp (one ACT turn), so two weather
        calls in the same turn get ordinals 1 and 2 — exactly what the rich-media
        parser needs to pair each card to its tool_calls row.
        """
        counters = getattr(self._mp, "_rich_media_ordinals", None)
        if counters is None:
            counters = {}
            setattr(self._mp, "_rich_media_ordinals", counters)
        counters[tool_name] = counters.get(tool_name, 0) + 1
        return counters[tool_name]

    @staticmethod
    def _render(tool_name: str, tr: ToolResult, ordinal: "int | None" = None) -> str:
        """Render the sealed wire envelope for *tr* — the ONLY envelope formatter.

        success: ``[<tool>(status=success, <meta>)]\\n<body>\\n[end:<tool>]`` —
        dict/list body as compact JSON, str body verbatim. When *ordinal* is set
        (rich card on a user-broadcast turn) the rich block (``\\n\\n`` +
        instruction with the ordinal-keyed span tag) is appended to the body so
        the rich-media parser can pair the card.

        error: ``[<tool>(status=error, code=<code>, <meta>)]\\n<message>`` plus a
        ``hint:`` line and a ``valid:`` line when those are set, then
        ``[end:<tool>]``.
        """
        if tr.status == "error":
            head_parts = ["status=error", f"code={tr.code}"]
            head_parts.extend(f"{k}={_meta_val(v)}" for k, v in tr.meta.items())
            lines = [f"[{tool_name}({', '.join(head_parts)})]", str(tr.body)]
            if tr.hint:
                lines.append(f"hint: {tr.hint}")
            if tr.valid:
                lines.append(f"valid: {' | '.join(tr.valid)}")
            lines.append(f"[end:{tool_name}]")
            return "\n".join(lines)

        head_parts = ["status=success"]
        head_parts.extend(f"{k}={_meta_val(v)}" for k, v in tr.meta.items())
        if isinstance(tr.body, (dict, list)):
            body_str = json.dumps(tr.body, ensure_ascii=False, separators=(",", ":"))
        else:
            body_str = str(tr.body)

        if ordinal is not None and tr.rich is not None:
            body_str = ToolDispatcher._render_rich(tool_name, tr, ordinal, body_str)

        return f"[{tool_name}({', '.join(head_parts)})]\n{body_str}\n[end:{tool_name}]"

    @staticmethod
    def _render_rich(tool_name: str, tr: ToolResult, ordinal: int, body_str: str) -> str:
        """Append the rich-media block: ``<body>\\n\\n<card instruction>``.

        The instruction carries the ``<span id='<tool>_<ordinal>'>`` tag the
        rich-media parser anchors on (``rich_media_parser._find_payload`` splits
        the envelope body on the first blank line — head = card payload JSON,
        trailer = this instruction). The card payload JSON is *tr.rich* serialised
        compactly; the model is told to wrap its synthesis in the span.
        """
        tag = f"{tool_name}_{ordinal}"
        payload_json = json.dumps(tr.rich, ensure_ascii=False, separators=(",", ":"))
        instruction = _RICH_INSTRUCTION.format(tag=tag)
        return f"{payload_json}\n\n{instruction}"


# Body shown to the model when a card is paired: the structured tool body is the
# card payload (parsed by rich_media_parser), and the model MUST wrap its
# synthesis in the span tag or the user sees only plain text.
_RICH_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in "
    "<span id='{tag}'>your synthesis here</span>. The span renders as a card; "
    "without it the user sees only plain text."
)


def _meta_val(v: object) -> str:
    """Render a scalar meta value: booleans lower-cased (``true``/``false``),
    everything else via ``str``."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

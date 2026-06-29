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
from typing import TYPE_CHECKING, Callable, cast

# AbilityRegistry resolves native tool names; PolicyManager gates the allow path.
# Neither imports this module, so both import normally — no alias/patch hack.
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
    from services.processor_config import ProcessorConfig

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
        # The tool_calls row id _execute opened for the call in flight (None until
        # one runs). dispatch() writes the result onto it (finish) instead of
        # opening a second row; reset per dispatch().
        self._pending_call_id: int | None = None

    # ── The single tool-call entry point ──────────────────────────────────────

    def dispatch(self, tool_name: str, params: dict[str, object]) -> str:
        """The one path every ACT-loop tool call takes.

        match → bind → ACTION_REQUIRED pre-gate → resolve permission (inline) →
        PolicyManager.wrap(execute)
        → record → return a STRING. Records EVERY outcome (allow result, block,
        unknown) so the rendered trail tells the model what happened and it does
        not retry a blocked tool forever. No cancel check — the loop guards
        cancel_event one line before calling this. Spec §5.
        """
        from services.message_processor import _sanitize_llm_args  # noqa: PLC0415

        _sanitize: "Callable[[dict[str, object]], dict[str, object]]" = cast(
            "Callable[[dict[str, object]], dict[str, object]]", _sanitize_llm_args
        )
        params = _sanitize(dict(params))
        act_summary: str | None = cast("str | None", params.pop("act_summary", None))
        config = getattr(self._mp, "config", None)
        self._pending_call_id = None

        ability = self._bind(tool_name)
        if ability is None:
            result_text = f"Unknown tool: {tool_name}"
        else:
            # Heal model-mangled argument KEYS against the tool's declared schema
            # before any gate or run() reads them: a stray-quote/case/alias key
            # (e.g. 'source"', 'URL', or read's 'url' for 'source') is rewritten
            # to its canonical parameter, so the ACTION_REQUIRED pre-gate and the
            # ability see the real key instead of a corrupt one that would bounce
            # on a spurious required-field error. Defensive: a
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
                    channel=cast("ProcessorConfig.PolicyChannel", getattr(config, "policy_channel", None)),
                    permission=permission,
                    callback=lambda: self._execute(ability, params, act_summary),
                    # The turn's cancel_event lets a parked `ask` prompt unwind on
                    # cancel instead of pinning the per-channel lock.
                    # Sourced off the invoking mp (the action endpoint's ctx exposes
                    # it too); absent → None → today's blocking wait (self-no-op).
                    cancel_event=getattr(self._mp, "cancel_event", None),
                )

        # Computed BEFORE the record below, so this call is not yet on the trail
        # and cannot self-match. The steer (``""`` when not a repeat) rides back
        # to the model on top of the unchanged recorded error.
        steer = self._repeat_error_steer(tool_name, result_text)

        from services.act_trail import ActTrail  # noqa: PLC0415
        if self._pending_call_id is not None:
            # The allow path already opened the row (and started its live timer)
            # in _execute — just write the final result onto it.
            ActTrail().finish(call_id=self._pending_call_id, result=result_text)
        else:
            # A call that never entered execution (unknown tool, denied, or
            # pre-validation failure): no live timer brackets it, so record the
            # outcome whole — the rendered trail tells the model what happened so
            # it does not retry a blocked tool forever. Anchors to the assistant
            # step row that emitted the call (mp.anchor), falling back to the input
            # row (mp.uid) for framework / turn-zero / compaction calls.
            ActTrail().record(
                tool_name=tool_name,
                params=params,
                result=result_text,
                transcript_id=getattr(self._mp, "anchor", None) or getattr(self._mp, "uid", None),
                summary=act_summary,
            )
        return result_text + steer

    def _signal(self, state: str, **extra: object) -> None:
        """Relay a live tool signal (tool_called / tool_done) through the invoking
        mp's broadcast chokepoint, keyed by its turn. A context that exposes no
        ``broadcast`` (the /action endpoint's ctx) is a silent no-op."""
        broadcast = getattr(self._mp, "broadcast", None)
        if broadcast is None:
            return
        broadcast(state, getattr(self._mp, "turn_id", None), **extra)

    def _repeat_error_steer(self, tool_name: str, result_text: str) -> str:
        """Steering suffix when this exact error already fired for an identical
        call earlier this turn — else ``""`` (self-no-op, appended unconditionally).

        The ACT loop has no iteration cap (a runaway turn is a hard-restart
        condition), so a tool stuck returning the same error can spin for minutes
        until the model abandons it. When the identical error envelope is already
        on the turn's trail, append a one-shot instruction telling the model to
        stop retrying, re-check the schema, and escalate to the user. Identical
        params yield an identical deterministic error, so envelope-equality IS
        call-identity — and the success path never pays the query (the cheap
        prefix check short-circuits first)."""
        if not result_text.startswith(f"[{tool_name}(status=error"):
            return ""
        channel = getattr(getattr(self._mp, "config", None), "channel", None)
        turn_id = getattr(self._mp, "turn_id", None)
        if channel is None or turn_id is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415
        seen = any(
            row.get("tool_name") == tool_name and row.get("result") == result_text
            for row in ActTrail().fetch_by_turn(channel, turn_id)
        )
        return _REPEAT_ERROR_STEER if seen else ""

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

    def _execute(self, ability: "Ability", params: dict[str, object], act_summary: "str | None" = None) -> str:
        """Open row + tool_called → (async-decision) → run → tool_done → render.
        Called ONLY on the allow path (dispatch() passes this as wrap's callback).
        The per-call ``async`` flag — popped here, a framework key never passed to
        run() — decides whether the real work blocks this ACT iteration or runs on
        a background thread; run() is identical either way. Returns the rendered
        envelope STRING.

        The tool_calls row is opened HERE (the instant the call begins) so the
        live ``tool_called`` signal can name it and the surface can start the
        act-trail timer; ``tool_done`` stops it. The row id is stashed on
        ``self._pending_call_id`` so dispatch() writes the result onto it (finish)
        rather than opening a second row.

        Reads the bound parent off ``ability.mp`` (set by _bind()). act_summary
        (popped from params by dispatch()) is the live summary; it is NOT a run()
        argument. Spec §4.0 / §4.2 / D5.
        """
        from services.act_trail import ActTrail  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        run_async = bool(params.pop("async", False))
        config = getattr(self._mp, "config", None)
        tool_name = ability.get_name()
        transcript_id = getattr(self._mp, "anchor", None) or getattr(self._mp, "uid", None)

        # Open the row NOW (with the live summary) so tool_called names it; the
        # result lands when dispatch() finishes it after this returns.
        call_id = ActTrail().start(
            tool_name=tool_name, params=params, transcript_id=transcript_id, summary=act_summary,
        )
        self._pending_call_id = call_id
        self._signal(
            MessageProcessor._WS_TOOL_CALLED,
            id=call_id, name=tool_name, summary=act_summary, transcript_row_id=transcript_id,
        )

        if run_async:
            # AsyncDelegateRunner owns the daemon-thread lifecycle + the captured
            # mp it delivers through; it returns the placeholder immediately so
            # this ACT iteration is never blocked (spec §4.0 / §4.4). The
            # placeholder is prose the model reads while the real work runs.
            from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415
            placeholder = async_delegate_runner.spawn(ability, params, self._mp, act_summary)
            tr = ToolResult.ok(str(placeholder))
        else:
            tr = self._run(ability, params)

        # Execution returned — stop the act-trail timer the tool_called started.
        self._signal(MessageProcessor._WS_TOOL_DONE, id=call_id, summary=act_summary)

        # Rich-media ordinal is assigned ONLY when the invoking mp broadcasts to
        # the user. Subagents / background channels never get a card: their
        # natural-language synthesis is consumed by the parent, so a span emitted
        # at that hop has no tool_calls row paired to it. This is the single
        # physical chokepoint that gates the entire card path.
        ordinal = None
        if tr.rich is not None and getattr(config, "broadcast_to", None) == "user":
            ordinal = self._next_ordinal(tool_name)

        return self._render(tool_name, tr, ordinal)

    # ── Action pre-validation (ACTION_REQUIRED) ────────────────────────────────

    @staticmethod
    def _prevalidate(ability: "Ability", params: dict[str, object]) -> "ToolResult | None":
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
    def _run(ability: "Ability", params: dict[str, object]) -> ToolResult:
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
        counters: dict[str, int] = cast("dict[str, int]", getattr(self._mp, "_rich_media_ordinals", None))
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

# Appended to the SECOND (and later) identical error a tool returns in one turn,
# to break a retry loop. Written plainly so smaller models act on it.
_REPEAT_ERROR_STEER = (
    "\n\n[loop-guard] You already made this exact call this turn and got the same "
    "error. Do NOT call it again the same way. First, look up the tool's correct "
    "inputs with the `find_tools` tool and fix your input, then try once more. If "
    "it still fails, stop retrying: tell the user what error you are getting, that "
    "this tool may not be working right now, and ask them how they want to proceed."
)


def _meta_val(v: object) -> str:
    """Render a scalar meta value: booleans lower-cased (``true``/``false``),
    everything else via ``str``."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DispatchService — the single chokepoint every ACT-loop tool call takes.

Every tool call enters through ``self.mp.dispatch_service.dispatch()``: match
the handler (``_bind``) → heal argument keys → ACTION_REQUIRED pre-gate →
policy-gate the call → run it (``_execute``) → render the sealed wire envelope
(``_render``) → persist + emit the outcome through ``self.mp.tool_call_service``
→ return a STRING. This is the ONLY path from the ACT loop to ability
execution, and the ONLY place the ``[<tool>(status=…)]\\n…\\n[end:<tool>]``
envelope is formatted.

``run()`` MUST return an :class:`abilities._result.ToolResult`. Anything else is
a hard error: this service renders ``status=error, code=non-canonical-result``
naming the ability and the offending type, and logs at ERROR. There is no
legacy-dict / plain-str compatibility shim — divergence fails loudly.

Persistence + WS emission never happen here directly — every outcome (started /
finished / recorded) is written and broadcast through ``self.mp.tool_call_service``
(§3.5); this service only decides WHAT happened and hands it over.

Policy interim note: setting lookup-or-create, the interactive ask-gate (park +
WS notify + the cross-request ``_permission_gates`` registry ``POST
/api/policies/respond`` wakes), and the blocked-log write are not yet modelled
on the new active-record spine — there is no ``Policy`` model in this phase and
this service may not emit WS directly (above), so re-pointing the ask-gate's
notification through ``Websocket.broadcast`` is not reachable from here. Until a
``PolicyService`` sibling exists (E/G), this service calls the untouched
``services.policy_manager.PolicyManager`` INSTANCE method (``authorize``,
replacing the static ``wrap`` entry point) for that storage + ask-gate
mechanics, built on the static :class:`~services.database.Database` gateway
(thread-local connections, so it shares the same connection as every other
caller) rather than the forbidden shared-singleton getter. The channel +
should_stop it gates with are read off the typed ``self.mp`` — the getattr wedges the old dispatcher used for both are
gone.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable, cast

from abilities._mcp_ability import _MCPAbility
from services.key_healer import KeyHealer
from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from abilities._result import ToolResult
from exceptions import VaultLockedError
from models.tool_call import ToolCall
from services.client_context import ClientContext
from services.policy_manager import PolicyManager

if TYPE_CHECKING:
    from abilities._ability import Ability
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class DispatchService:
    """Dispatches one tool call on behalf of the owning ``MessageProcessor``.

    ``dispatch()`` is the single chokepoint: match → bind → gate (``_authorize``)
    → execute → render → record. The gated work runs in ``_execute()`` (open row
    → run → render); ``dispatch()`` writes the terminal outcome afterwards.
    """

    def __init__(self, mp: "MessageProcessor") -> None:
        self.mp = mp
        # Per-turn, per-tool rich-media ordinal counter (§ordinal pairing) — scoped
        # to this instance's lifetime, which mirrors one ACT turn exactly the way
        # the pre-rewrite counter was scoped to its owning mp.
        self._ordinals: dict[str, int] = {}

    # ── The single tool-call entry point ──────────────────────────────────────

    def dispatch(self, tool_name: str, params: dict[str, object]) -> str:
        """The one path every ACT-loop tool call takes.

        match → bind → ACTION_REQUIRED pre-gate → resolve permission → gate →
        record → return a STRING. Records EVERY outcome (allow result, block,
        unknown) so the rendered trail tells the model what happened and it does
        not retry a blocked tool forever. No cancel check — the loop guards
        should_stop() one line before calling this.

        Appends a steering suffix to the persisted result when the same tool
        has already fired with identical params this turn, breaking the loop
        before the hard runaway-kill threshold. The steer is persisted into the
        tool_calls row so the model re-reads it on its next step (see
        ``prompt_service`` rendering ``call.result``); the return value carries
        the same string so callers see what the model will see."""
        params, act_summary, ability = self._prepare(tool_name, params)
        if ability is None:
            result_text = f"Unknown tool: {tool_name}"
            steer = self._repeat_error_steer(tool_name, result_text) or self._repeat_call_steer(tool_name, params)
            self.mp.tool_call_service.record(
                tool_name=tool_name, params=params, result=result_text + steer,
                summary=act_summary, state=ToolCall.ERROR, emit=False,
            )
            return result_text + steer

        result_text, call_id, state = self._dispatch_bound(tool_name, ability, params, act_summary)
        steer = self._repeat_error_steer(tool_name, result_text) or self._repeat_call_steer(tool_name, params)
        if call_id is not None:
            self.mp.tool_call_service.finish(call_id=call_id, result=result_text + steer, state=state)
        else:
            self.mp.tool_call_service.record(
                tool_name=tool_name, params=params, result=result_text + steer,
                summary=act_summary, state=state, emit=True,
            )
        return result_text + steer

    def _prepare(
        self, tool_name: str, params: dict[str, object],
    ) -> "tuple[dict[str, object], str | None, Ability | None]":
        """Turn a raw provider tool call into its EXECUTED identity, without running
        or recording anything: strip LLM sentinel tokens from the values, lift out
        the model's ``act_summary`` narration, bind the ability, and heal the
        argument KEYS against its declared schema (a stray-quote/case/alias key
        rewritten to its canonical parameter). Returns
        ``(healed_params, act_summary, ability)``; ``ability`` is ``None`` for an
        unknown tool, in which case the params are sanitised + act_summary-stripped
        but not healed (no schema to heal against).

        The ONE canonicalisation recipe: :meth:`dispatch` runs it to execute the
        call, and :meth:`canonical_params` runs it to give the runaway guard a key
        byte-identical to what will actually execute — so the two can never drift.
        A registry/schema fault must never break dispatch, so a heal failure logs
        and falls back to the raw (sanitised) params.
        """
        params = cast("dict[str, object]", self.mp.provider_service.sanitize_args(dict(params)))
        act_summary = cast("str | None", params.pop("act_summary", None))
        ability = self._bind(tool_name)
        if ability is None:
            return params, act_summary, None
        try:
            params = KeyHealer().heal(params, ability.get_parameters())
        except Exception:  # noqa: BLE001
            logger.exception(
                "[DispatchService] key canonicalisation failed for %s — "
                "proceeding with raw params", tool_name,
            )
        return params, act_summary, ability

    def canonical_params(self, tool_name: str, params: dict[str, object]) -> dict[str, object]:
        """The params ``run()`` will actually receive for *tool_name* — sanitised,
        ``act_summary``- and ``async``-stripped, and key-healed — computed WITHOUT
        executing or recording anything. The runaway guard keys its per-tool tally
        on this, so neither cycling synonym keys (``city``/``loc``/``place``/
        ``region`` → ``location``) NOR toggling the framework ``async`` flag around
        otherwise-identical params — both of which collapse to one executed call —
        can evade the loop backstop."""
        params, _act_summary, _ability = self._prepare(tool_name, params)
        # ``async`` is a framework control flag ``_execute`` pops before run() (like
        # ``act_summary``, already lifted by ``_prepare``): it decides sync-vs-async
        # dispatch, never what the tool does, so it is not part of the executed
        # identity the guard tallies on.
        params.pop("async", None)
        return params

    def _dispatch_bound(
        self,
        tool_name: str,
        ability: "Ability",
        params: dict[str, object],
        act_summary: "str | None",
    ) -> tuple[str, int | None, str]:
        """The ability-found path for ALREADY-canonicalised params (:meth:`_prepare`
        has sanitised the values, lifted ``act_summary``, and healed the keys):
        ACTION_REQUIRED pre-gate → resolve permission → policy-gate ``_execute``.
        Returns the rendered result, the tool_calls row id opened on the allow path
        (``None`` when the call never executed), and the terminal state to write."""
        if (pre := self._prevalidate(ability, params)) is not None:
            # ACTION_REQUIRED pre-gate fires BEFORE the permission is formed: a
            # hallucinated action would otherwise lazily seed a bogus
            # '<tool>.<action>' ask row and freeze the turn waiting for human
            # approval. Malformed input never reaches the policy gate or run().
            return self._render(tool_name, pre, None), None, ToolCall.ERROR

        # The risk class the gate keys on is derived from the inputs via the
        # ability's classify_action hook (default None), NOT trusted from a
        # model-supplied 'action' — a self-declared action is prompt-injectable
        # and must never decide the permission. Fall back to the action param
        # only when the tool offers no classification.
        classified = ability.classify_action(params)
        action = classified if classified is not None else params.get("action")
        permission = f"{tool_name}.{action}" if action else tool_name

        call_id: int | None = None
        state = ToolCall.ERROR

        def _run_gated() -> str:
            nonlocal call_id, state
            text, call_id, state = self._execute(ability, params, act_summary)
            return text

        result_text = self._authorize(permission, _run_gated)
        return result_text, call_id, state

    def _repeat_error_steer(self, tool_name: str, result_text: str) -> str:
        """Steering suffix when this exact error already fired for an identical
        call earlier in this exchange — else ``""`` (self-no-op, appended
        unconditionally).

        The ACT loop has no iteration cap, so a tool stuck returning the same
        error can spin for minutes until the model abandons it. When the
        identical error envelope is already on this exchange's trail
        (``by_exchange`` — a spin is a within-recursion pathology, not a user
        re-asking in a later reply), append a one-shot instruction telling the
        model to stop retrying, re-check the schema, and escalate to the user.
        Identical params yield an identical deterministic error, so
        envelope-equality IS call-identity — and the success path never pays the
        query (the cheap prefix check short-circuits first)."""
        if not result_text.startswith(f"[{tool_name}(status=error"):
            return ""
        seen = any(
            call.tool_name == tool_name and call.result == result_text
            for call in self.mp.tool_call_service.by_exchange()
        )
        return _REPEAT_ERROR_STEER if seen else ""

    def _repeat_call_steer(self, tool_name: str, params: dict[str, object]) -> str:
        """Steering suffix when the runaway guard has already tallied this many
        identical calls this turn — else ``""``.

        Reads the tally set by :meth:`~controllers.message_processor.MessageProcessor._guard_runaway`
        BEFORE dispatch, so the steer fires on the SECOND identical call —
        before the hard kill threshold at ``_RUNAWAY_TOOL_CALL_LIMIT``.

        ``params`` is the post-``_prepare`` identity (sanitised + act_summary-
        lifted + key-healed) — only ``async`` remains to be stripped to byte-
        match the guard's key. A copy is taken and ``async`` popped on the copy
        so the live ``params`` dict is never mutated.

        Dispatches that bypass ``_step`` (the compactor, document upload, or
        the async delegate's dedicated mp) have zero tally on this mp and so
        never steer here."""
        identity = dict(params)
        identity.pop("async", None)
        if self.mp.repeat_call_count(tool_name, identity) >= _REPEAT_CALL_STEER_THRESHOLD:
            return _REPEAT_CALL_STEER
        return ""

    # ── Policy gate ─────────────────────────────────────────────────────────────

    def _authorize(self, permission: str, callback: "Callable[[], str]") -> str:
        """Gate *callback* for this turn's (channel, permission) — the spine's
        replacement for the static ``PolicyManager.wrap`` entry point: the
        instance method (``authorize``) is invoked directly, with the channel and
        the cooperative ``should_stop`` sourced off the typed ``self.mp`` instead
        of the old getattr wedges. ``should_stop`` lets a parked ask-prompt unwind
        when the turn is cancelled instead of pinning the per-channel lock."""
        return PolicyManager().authorize(
            channel=self.mp.config.policy_channel,
            permission=permission,
            callback=callback,
            should_stop=self.mp.turn_execution_service.should_stop,
        )

    # ── Resolution ─────────────────────────────────────────────────────────────

    def _bind(self, tool_name: str) -> "Ability | None":
        """Resolve *tool_name* to a FRESH per-call Ability instance bound to the
        owning ``mp`` (passed to ``Ability.__init__`` → ``self.mp``).

        Native abilities live in the registry as singletons; binding *mp* onto a
        shared instance would race across concurrent turns. So every call gets
        its own instance (abilities are stateless, no custom ``__init__``)
        constructed with the owning MessageProcessor. ``_mcp_*`` names resolve to
        a fresh synthetic ``_MCPAbility`` proxy, also mp-bound. Unknown native
        name → None (``dispatch()`` turns this into an 'Unknown tool' string)."""
        if tool_name.startswith("_mcp_"):
            return _MCPAbility(tool_name, mp=self.mp)
        try:
            template = AbilityRegistry.get(tool_name)
        except KeyError:
            return None
        return type(template)(mp=self.mp)

    # ── The allow-path executor ─────────────────────────────────────────────────

    def _execute(
        self, ability: "Ability", params: dict[str, object], act_summary: "str | None" = None,
    ) -> tuple[str, "int | None", str]:
        """Open row (via ``tool_call_service.start`` — emits ``started``) →
        (async-decision) → run → resolve terminal state → render. Called ONLY on
        the allow path (``_dispatch_bound`` passes this as the gate's callback).
        The per-call ``async`` flag — popped here, a framework key never passed
        to run() — decides whether the real work blocks this ACT iteration or
        runs on a background thread; run() is identical either way. Returns the
        rendered envelope STRING plus the opened row id and its resolved terminal
        state, so ``_dispatch_bound`` can finish that same row instead of opening
        a second one.

        Reads the bound parent off ``ability.mp`` (set by ``_bind()``).
        act_summary (popped from params by ``dispatch()``) is the live summary;
        it is NOT a run() argument."""
        run_async = bool(params.pop("async", False))
        tool_name = ability.NAME

        call_id = self.mp.tool_call_service.start(
            tool_name=tool_name, params=params, summary=act_summary,
        )

        if run_async:
            # AsyncDelegateRunner owns the daemon-thread lifecycle + the captured
            # mp it delivers through; it returns the placeholder immediately so
            # this ACT iteration is never blocked. The placeholder is prose the
            # model reads while the real work runs.
            from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415
            placeholder = async_delegate_runner.spawn(ability, params, self.mp, act_summary)
            tr = ToolResult.ok(str(placeholder))
        else:
            tr = self._run(ability, params)

        state = ToolCall.ERROR if tr.status == "error" else ToolCall.DONE

        # Rich-media ordinal is assigned ONLY when the owning mp broadcasts to a
        # live surface (``broadcast_to`` set — the user spine or a schedule
        # thread). Subagents / background channels never get a card: their
        # natural-language synthesis is consumed by the parent, so a span emitted
        # at that hop has no tool_calls row paired to it. This is the single
        # physical chokepoint that gates the entire card path.
        ordinal = None
        if tr.rich is not None and self.mp.config.broadcast_to is not None:
            ordinal = self._next_ordinal(tool_name)

        # The follow-up nudge fires only on a real synchronous SUCCESS: never on
        # an error result, never on the async placeholder (the nudge would fire
        # before the real work ran). Empty default => nothing appended, so
        # non-overriding abilities render byte-identical to before.
        follow_up = ability.get_follow_up(tr) if (not run_async and tr.status == "success") else ""
        return self._render(tool_name, tr, ordinal, follow_up=follow_up), call_id, state

    # ── Action pre-validation (ACTION_REQUIRED) ────────────────────────────────

    def _prevalidate(self, ability: "Ability", params: dict[str, object]) -> "ToolResult | None":
        """Validate the ability's ACTION_REQUIRED map BEFORE the policy gate.

        An empty map (the default) means no pre-validation — unmigrated tools are
        untouched. Otherwise: an unknown action → ``code=unknown-action`` with
        ``valid=`` the action-map keys; a known action missing required params →
        a SINGLE ``code=missing-params`` error naming ALL missing params.

        The map key ``""`` covers action-less tools."""
        action_map = ability.ACTION_REQUIRED
        if not action_map:
            return None

        action = params.get("action")
        key = action if isinstance(action, str) else ""
        if key not in action_map:
            return ToolResult.err(
                f"Unknown action {action!r}." if action is not None
                else "This tool requires an 'action'.",
                code="unknown-action",
                valid=tuple(k for k in action_map if k),
            )

        # Non-empty is the global rule; ``ALLOW_EMPTY`` opts a param out of it.
        # An opted-out param is still required to be PRESENT — only its emptiness
        # stops being a failure, so ``replace=""`` reaches the ability while a
        # genuinely absent ``replace`` is still reported missing.
        exempt = ability.ALLOW_EMPTY
        missing = [
            p for p in action_map[key]
            if (p not in params if p in exempt else not params.get(p))
        ]
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

    def _run(self, ability: "Ability", params: dict[str, object]) -> ToolResult:
        """Execute ability.run() synchronously and enforce the ToolResult contract.

        Loads the flattened client telemetry onto ``ability.telemetry`` just
        before run(); the ability reads its parent off ``self.mp``.

        There is no wall-clock bound — an ability runs to completion. The
        framework never abandons a tool call: the only things that stop a
        running tool are cooperative cancellation (should_stop) and the
        network-level I/O timeouts inside individual abilities.

        Returns a :class:`ToolResult`. A bad parameter never raises: the
        ability's PARAMS bag returns the error ``ToolResult`` it authored at
        the failure site, and that result is passed through untouched. A
        raised exception becomes ``code=unhandled-exception`` (with the
        VaultLockedError friendly message). A non-ToolResult return value
        HARD-FAILS as ``code=non-canonical-result``."""
        try:
            ctx = ClientContext.current()
            ability.telemetry = ctx.as_dict() if ctx else None
        except Exception:  # noqa: BLE001
            ability.telemetry = None

        try:
            # The typed-input door: every first-party ability declares PARAMS,
            # and its bag is built HERE via from_params — which returns either
            # the bag or the error ToolResult the bag authored at the failure
            # site. Value or error, both are return values; no exception
            # crosses this boundary. The raw-dict branch serves the synthetic
            # _MCPAbility proxies only, whose schemas live on the remote MCP
            # server and can never have a compile-time bag.
            if ability.PARAMS is None:
                raw = ability.run(params)
            else:
                built = ability.PARAMS.from_params(params)
                if isinstance(built, ToolResult):
                    return built
                raw = ability.run(built)
        except Exception as exc:  # noqa: BLE001
            # VaultLockedError gets a friendlier message.
            try:
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
                "[DispatchService] %s.run() returned a non-ToolResult (%s) — every "
                "ability must return abilities._result.ToolResult via ok()/err().",
                ability.NAME, offending,
            )
            return ToolResult.err(
                f"{ability.NAME} returned a non-canonical result of type "
                f"{offending}; abilities must return a ToolResult.",
                code="non-canonical-result",
            )
        return raw

    # ── The single wire-envelope renderer ──────────────────────────────────────

    def _next_ordinal(self, tool_name: str) -> int:
        """Per-turn, per-tool ordinal counter held on this instance.

        Keyed by tool name and scoped to this DispatchService (one ACT turn), so
        two weather calls in the same turn get ordinals 1 and 2 — exactly what
        the rich-media parser needs to pair each card to its tool_calls row."""
        self._ordinals[tool_name] = self._ordinals.get(tool_name, 0) + 1
        return self._ordinals[tool_name]

    def _render(
        self,
        tool_name: str,
        tr: ToolResult,
        ordinal: "int | None" = None,
        follow_up: str = "",
    ) -> str:
        """Render the sealed wire envelope for *tr* — the ONLY envelope formatter.

        success: ``[<tool>(status=success, <meta>)]\\n<body>\\n[end:<tool>]`` —
        dict/list body as compact JSON, str body verbatim. When *ordinal* is set
        (rich card on a user-broadcast turn) the rich block (``\\n\\n`` +
        instruction with the ordinal-keyed span tag) is appended to the body so
        the rich-media parser can pair the card. When *follow_up* is non-empty
        the follow-up instruction block is appended to the body AFTER any rich
        block and BEFORE the closing ``[end:<tool>]``.

        error: ``[<tool>(status=error, code=<code>, <meta>)]\\n<message>`` plus a
        ``hint:`` line and a ``valid:`` line when those are set, then
        ``[end:<tool>]``."""
        if tr.status == "error":
            head_parts = ["status=error", f"code={tr.code}"]
            head_parts.extend(f"{k}={self._meta_val(v)}" for k, v in tr.meta.items())
            lines = [f"[{tool_name}({', '.join(head_parts)})]", str(tr.body)]
            if tr.hint:
                lines.append(f"hint: {tr.hint}")
            if tr.valid:
                lines.append(f"valid: {' | '.join(tr.valid)}")
            lines.append(f"[end:{tool_name}]")
            return "\n".join(lines)

        head_parts = ["status=success"]
        head_parts.extend(f"{k}={self._meta_val(v)}" for k, v in tr.meta.items())
        if isinstance(tr.body, (dict, list)):
            body_str = json.dumps(tr.body, ensure_ascii=False, separators=(",", ":"))
        else:
            body_str = str(tr.body)

        if ordinal is not None and tr.rich is not None:
            body_str = self._render_rich(tool_name, tr, ordinal)

        if follow_up:
            body_str = f"{body_str}\n{_FOLLOW_UP_BLOCK.format(text=follow_up)}"

        return f"[{tool_name}({', '.join(head_parts)})]\n{body_str}\n[end:{tool_name}]"

    def _render_rich(self, tool_name: str, tr: ToolResult, ordinal: int) -> str:
        """Build the rich-media block that REPLACES the tool body:
        ``<card payload JSON>\\n\\n<card instruction>``.

        The instruction carries the ``<span id='<tool>_<ordinal>'>`` tag the
        rich-media parser anchors on. The card payload JSON is *tr.rich*
        serialised compactly; the model is told to wrap its synthesis in the
        span."""
        tag = f"{tool_name}_{ordinal}"
        payload_json = json.dumps(tr.rich, ensure_ascii=False, separators=(",", ":"))
        instruction = _RICH_INSTRUCTION.format(tag=tag)
        return f"{payload_json}\n\n{instruction}"

    def _meta_val(self, v: object) -> str:
        """Render a scalar meta value: booleans lower-cased (``true``/``false``),
        everything else via ``str``."""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)


# Body shown to the model when a card is paired: the structured tool body is the
# card payload (parsed by rich_media_parser), and the model MUST wrap its
# synthesis in the span tag or the user sees only plain text.
_RICH_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in "
    "<span id='{tag}'>your synthesis here</span>. The span renders as a card; "
    "without it the user sees only plain text."
)

# A standing next-step nudge a tool appends to its OWN successful result. Placed
# INSIDE the envelope (after any rich block, before ``[end:<tool>]``).
_FOLLOW_UP_BLOCK = "[follow_up_instruction]\n{text}\n[end:follow_up_instruction]"

# Appended to the SECOND (and later) identical error a tool returns in one turn,
# to break a retry loop. Written plainly so smaller models act on it.
_REPEAT_ERROR_STEER = (
    f"\n\n[loop-guard] You already made this exact call this turn and got the same "
    f"error. Do NOT call it again the same way. First, look up the tool's correct "
    f"inputs with the `{FindToolsAbility.NAME}` tool and fix your input, then try once more. If "
    f"it still fails, stop retrying: tell the user what error you are getting, that "
    f"this tool may not be working right now, and ask them how they want to proceed."
)

#: Repeat-call steer fires when the runaway guard has already tallied this many
#: identical calls this turn. Below this threshold every call dispatches normally.
_REPEAT_CALL_STEER_THRESHOLD = 2

# Appended to the SECOND (and later) identical call a tool makes in one turn —
# BEFORE the runaway kill threshold — to steer the model out of the loop rather
# than waiting for the hard crash. Written plainly so smaller models act on it.
_REPEAT_CALL_STEER = (
    "\n\n[loop-guard] You already invoked this exact tool with these exact "
    "parameters this turn and got the result above. Do NOT call it again with "
    "the same parameters. Change the parameters, use a different tool, or stop "
    "calling tools and answer with what you already have. Repeating the identical "
    "call again will abort this turn."
)

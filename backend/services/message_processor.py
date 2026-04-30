# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MessageProcessor — abstract base class for all LLM message processors.

North star: /Volumes/llm/chalie-plans/message-processing.md

Lifecycle: one instance per turn. Two turns never share the same object.
Do not add `.instance()` / singleton accessors.

Each input channel (WebSocket, DMN timer, goal pursuit, scheduled prompt, …)
constructs its own MessageProcessor subclass directly. The subclass hardcodes
its CHANNEL and ROLE, implements getUserDefinition() and getUserPrompt(), and
fans out post-turn services via postTurn(). The base class provides the ACT
loop (send()), atomic persistence (store()), tool dispatch (handleTool()), and
compaction primitives.
"""

import contextlib
import contextvars
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from services.llm_service import PayloadTooLargeError
from services.metrics_accumulator import MetricsAccumulator
from services.system_message_prompt import SystemMessagePrompt
from services.time_utils import utc_now
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

_LLM_SENTINEL_PATTERNS = (
    re.compile(r'<\|[^|<>]*\|>'),
    re.compile(r'<\|[^|<>]*\|'),
)


def _sanitize_llm_args(value):
    if isinstance(value, str):
        for p in _LLM_SENTINEL_PATTERNS:
            value = p.sub('', value)
        return value.strip()
    if isinstance(value, list):
        return [_sanitize_llm_args(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_llm_args(v) for k, v in value.items()}
    return value


# ── Current-processor context ─────────────────────────────────────────────────
#
# Innate skills and downstream services sometimes need to reach the
# MessageProcessor instance for the turn they are running inside (e.g. to
# append to ``_memory_query_history`` or read ``_memory_seed``). The
# MessageProcessor instance is not part of any tool signature — skills are
# dispatched by name via ``handleTool()`` and must not receive the
# processor as an argument.
#
# We expose an async-safe ``ContextVar`` + a context manager so that
# ``send()`` can bind the running processor for the duration of a turn and
# tools called from inside that turn can discover it via
# ``current_processor()``. Outside a turn this returns ``None`` and callers
# MUST degrade gracefully.
_CURRENT_PROCESSOR: contextvars.ContextVar["MessageProcessor | None"] = (
    contextvars.ContextVar("chalie_current_processor", default=None)
)


def current_processor() -> "MessageProcessor | None":
    """Return the `MessageProcessor` for the current turn, or None.

    Returns None when called outside a ``MessageProcessor.send()`` turn
    (worker threads, tests, legacy orchestrator). Callers must handle that.
    """
    return _CURRENT_PROCESSOR.get()


@contextlib.contextmanager
def bind_current_processor(processor: "MessageProcessor"):
    """Context manager that binds ``processor`` as the current-turn processor.

    Wrap the body of ``MessageProcessor.send()`` with this. Resets the
    ContextVar on exit regardless of success / exception path.
    """
    token = _CURRENT_PROCESSOR.set(processor)
    try:
        yield processor
    finally:
        _CURRENT_PROCESSOR.reset(token)


class MessageProcessor:
    """Abstract base for every LLM message processor.

    Subclasses must:
    - Set `CHANNEL` (str) — fixed transcript channel for this processor.
    - Set `ROLE` (str) — transcript role for the input row (e.g. 'user').
    - Implement `getUserPrompt() -> str`
    - Implement `getUserDefinition() -> str`

    Subclasses may:
    - Override `SYSTEM_PROMPT_CLASS` with a concrete `SystemMessagePrompt` subclass.
    - Override `ALWAYS_AVAILABLE` with a list of ability names injected as
      innate tools every iteration.
    - Override `DISCOVERABLE` with a list of ability names that ``find_tools``
      may surface for this processor at runtime.
    - Override `getDynamicTools()` to filter or augment discovered tools.
    - Override `postTurn()` to fan out per-channel post-turn services.
    """

    # ── Class constants (overridable by subclasses) ───────────────────────────

    JOB: str = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = SystemMessagePrompt  # class reference, not instance
    # Ability names pre-injected as native tools on every ACT iteration.
    ALWAYS_AVAILABLE: list[str] = []
    # Ability names ``find_tools`` may surface for this processor at runtime.
    # ``find_tools`` itself is gated to ``WHERE name IN DISCOVERABLE`` so a
    # processor can never discover anything outside this list.
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS: int = 30
    MAX_TIMEOUT: int = 900    # seconds — ACT loop only
    THINKING_TIMEOUT: int = 600  # seconds — exploration pass budget (independent of ACT)
    # Ceiling for a single getPreviousMessages() pull. Commit 7's compaction
    # budget assumes a row count this low — if a channel ever exceeds it we
    # want compaction to kick in, not an unbounded fetch.
    _TRANSCRIPT_FETCH_LIMIT: int = 2000

    # ── Subclass must set ─────────────────────────────────────────────────────

    CHANNEL: str = ''   # e.g. 'user', 'dmn', 'goal_pursuit', 'scheduled'
    ROLE: str = ''      # e.g. 'user', 'proactive_thought', 'goal_pursuit'

    # When True, send() skips write_input_row() and store() skips the
    # assistant row — self._uid stays None for the entire turn.
    # Set this on internal processors (EpisodeEncoderProcessor,
    # SuperEpisodeEncoderProcessor) that must not pollute the transcript.
    SKIP_TRANSCRIPT_WRITE: bool = False

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        self._memory_seed: str | None = None
        # Raw recall query used by pre_act(); kept separate from _memory_seed
        # (which is the formatted tag block) so recall_episodes() can embed
        # the original query for drift computation rather than the block string.
        self._memory_seed_query: str | None = None
        # Tracks the current ACT loop iteration so handleTool() can include
        # it in emitted tool events without thread-local indirection.
        self._current_iteration: int = 0
        # Per-turn log of memory recall queries (seed + llm_recall).
        # Populated by the memory skill recall path; consumed by the next
        # recall call for redundancy-narrow and drift-expand computation.
        # Entries: {'query': str, 'embedding': list[float],
        #           'caller': 'seed'|'llm_recall', 'effective_radius': float}.
        # Never persisted — cleared when the instance is discarded.
        self._memory_query_history: list[dict] = []
        self._act_trail: list[str] = []
        self._discovered_tools: list[dict] = []
        self._uid: int | None = None
        # Default is 'low' — classifier must explicitly set medium/high.
        # A 'medium' default would silently apply deliberation pressure to every
        # turn where the gate wasn't run (non-user channels) or crashed —
        # regressing benchmark behaviour on simple recall/chit-chat.
        self._thinking_level: str = 'low'
        self._deliberation_scalar: float | None = None   # raw sigmoid for this turn
        self._deliberation_ema: float | None = None      # EMA after this turn's update
        self._thinking_exploration: str | None = None
        # One-shot guard: a 413 from the provider triggers a Stage 2 ACT
        # restart, but only once per turn. A second 413 after compaction
        # means the compacted body is still over the transport-level cap;
        # restarting again would loop forever.
        self._payload_too_large_recovered: bool = False
        # Accumulator starts immediately so exploration + compaction tokens count.
        self._metrics: MetricsAccumulator = MetricsAccumulator()

    def set_turn_start(self, ts: float) -> None:
        """Override the accumulator start time (called from _handle_chat before thread spawn)."""
        self._metrics.start_time = ts

    # ── Abstract — subclass implements ───────────────────────────────────────

    def getUserPrompt(self) -> str:
        """Build the body of the user-message for the current ACT iteration.

        Called once per ACT iteration inside send(). Does NOT emit the
        ``### Checkpoint`` / ``### Current State`` headers — those are added
        by send().
        """
        raise NotImplementedError

    def getUserDefinition(self) -> str:
        """One-sentence description of who the 'user' is for this processor.

        Injected as the first line of the system prompt. Examples:
          UserMessageProcessor   → user synthesis string (real human)
          DMNMessageProcessor    → "The user is 'proactive_thought' — ..."
          GoalPursuitProcessor   → "The user is 'goal_pursuit' — ..."
        """
        raise NotImplementedError

    # ── Overridable hook ─────────────────────────────────────────────────────

    def getDynamicTools(self) -> list[dict]:
        """Return tool schemas discovered during this turn via find_tools.

        Default returns self._discovered_tools directly (identity — subclasses
        that mutate the list during a turn see the mutations reflected here).
        Subclasses may override to filter, replace, or suppress.
        """
        return self._discovered_tools

    # ── Final (concrete on base) ──────────────────────────────────────────────

    def getSystemPrompt(self) -> str:
        """Build the full system prompt for this turn.

        Prepends getUserDefinition() to the body produced by SYSTEM_PROMPT_CLASS.
        A fresh SYSTEM_PROMPT_CLASS instance is created, used, and discarded —
        no caching; this is called once per turn by send().

        Format: ``"{user_definition}\n\n{system_prompt_body}"``

        ``SYSTEM_PROMPT_CLASS`` defaults to the abstract ``SystemMessagePrompt``
        base, which cannot be instantiated — any subclass that fails to override
        it will raise ``TypeError`` at this call site rather than silently
        producing a blank body. Subclasses must set ``SYSTEM_PROMPT_CLASS`` to
        a concrete ``SystemMessagePrompt`` subclass.

        **Zero-arg construction is the contract.** Every `SystemMessagePrompt`
        subclass takes no constructor parameters — its `getPrompt()` returns
        a static body string.
        That keeps this base-class call site pure — no knowledge of
        subclass-specific args leaks up. Subclasses of `MessageProcessor` that
        need richer system-prompt inputs (personality voice, …) override
        `getSystemPrompt()` themselves and weave in the extra context around the
        template returned by `SYSTEM_PROMPT_CLASS().getPrompt()`.
        """
        # Intentionally zero-arg — see docstring. Subclasses override this
        # method (not SYSTEM_PROMPT_CLASS's signature) to pass real context.
        body = self.SYSTEM_PROMPT_CLASS().getPrompt()
        body = self._substitute_provider_placeholders(body)
        return f"{self.getUserDefinition()}\n\n{body}"

    def _substitute_provider_placeholders(self, body: str) -> str:
        """Replace per-provider placeholders in a system-prompt body.

        Currently substitutes ``{{provider_content_field_name}}`` with the
        active provider's CONTENT_FIELD_LABEL (e.g. ``message.content`` for
        Ollama, ``content[].text`` for Anthropic) so the model is told the
        exact JSON field where its user-visible prose lands.

        Best-effort: if the placeholder is absent or the provider lookup
        fails, the body passes through unchanged.
        """
        if "{{provider_content_field_name}}" not in body:
            return body
        try:
            from services.providers import Providers
            provider = Providers.instance()._resolve(self.JOB)
            label = getattr(provider, 'CONTENT_FIELD_LABEL', None)
        except Exception:
            label = None
        if not label:
            return body
        return body.replace("{{provider_content_field_name}}", label)

    def getTools(self) -> list[dict]:
        """Return the full tool list for the current ACT iteration.

        Resolution order (first-seen wins on duplicates):
        1. ALWAYS_AVAILABLE — innate tier (resolved via AbilityRegistry).
           Base is []; subclasses set the explicit list of ability names that
           are pre-injected on every iteration.
        2. getDynamicTools() — abilities discovered this turn via find_tools.
           Gated by ``DISCOVERABLE`` inside ``find_tools`` itself.

        Schema shape: {name, description, input_schema} — pulled from each
        Ability's NAME, SUMMARY, and INPUT_SCHEMA ClassVars respectively.

        Deduplication preserves first-seen order so the innate tier cannot
        be shadowed by a dynamic entry of the same name.
        """
        from abilities._registry import AbilityRegistry

        native: list[dict] = []
        if self.ALWAYS_AVAILABLE:
            for tool_name in self.ALWAYS_AVAILABLE:
                try:
                    ability = AbilityRegistry.get(tool_name)
                    native.append({
                        'name': ability.NAME,
                        'description': ability.SUMMARY,
                        'input_schema': ability.INPUT_SCHEMA,
                    })
                except KeyError:
                    logger.warning(
                        "[MessageProcessor.getTools] No ability registered for '%s'",
                        tool_name,
                    )

        dynamic = self.getDynamicTools()

        seen: set[str] = set()
        result: list[dict] = []
        for schema in native + dynamic:
            name = schema.get('name')
            if name and name not in seen:
                seen.add(name)
                result.append(schema)

        return result

    def getActLoopTrail(self) -> str:
        """Return the ACT loop trail as a single string for prompt injection.

        Returns '' on the first iteration (empty list). On subsequent
        iterations returns '\n'.join(self._act_trail).
        """
        return '\n'.join(self._act_trail)

    def getPreviousMessages(self, token_budget: int | None = None) -> str:
        """Assemble the ## Previous Messages block for this channel.

        Format (locked by the north star,
        ``/Volumes/llm/chalie-plans/message-processing.md`` § "Literal format"):

        - Input rows  : ``[YYYY-MM-DD HH:MM] <role>: <content>``
                        — role rendered **lowercase** for transcript input
                          rows. ``assistant`` is the sole exception — see next.
        - Assistant   : ``[YYYY-MM-DD HH:MM] Assistant: <content>``
                        — title case, capital A.
        - Durable     : ``[<tool_name>(<key>=<value>;…)] <result>``
          tool_calls    — **bare**, no timestamp prefix, no ``TOOL()``
                          wrapper. Inherits the owning row's timestamp
                          implicitly by positional placement.

        Algorithm:
        1. Look up the compaction row for self.CHANNEL.
        2. If a compaction exists, prepend ``compacted_text`` as the opening
           block and read transcript rows with ``id > compacted_up_to_id``.
           If no compaction exists, read all transcript rows for the channel.
        3. For each transcript row, emit the input / assistant line, then
           immediately below, any **durable** (``ephemeral=0``) tool_calls
           linked to that row in the bare format above.
        4. Ephemeral (``ephemeral=1``) tool_call rows are never emitted here.
        5. Durable tool_calls whose name is in ``_NEVER_RENDER_IN_PREVIOUS``
           are filtered out — the ``compaction`` pseudo-tool is an audit-only
           DTO whose content is already surfaced via the ``compactions`` table
           prepend (step 2) and must never double-render.

        The ``token_budget`` parameter is accepted for forward-compatibility
        with Commit 7 (compaction). In Commit 2/2a it is silently ignored — no
        truncation is performed.

        Returns '' when the channel has no transcript rows and no compaction.
        """
        del token_budget  # forward-compat placeholder; see docstring above
        from services import compaction_persistence, transcript_service
        from services.tool_call_service import ToolCallService
        from services.tool_render_and_record_service import ToolRenderAndRecordService

        compaction = compaction_persistence.get_compaction(self.CHANNEL)
        watermark = compaction['compacted_up_to_id'] if compaction else 0

        entries = transcript_service.get_recent(
            self.CHANNEL, limit=self._TRANSCRIPT_FETCH_LIMIT, since_id=watermark
        )

        if not entries and not (compaction and compaction.get('compacted_text')):
            return ''

        # Batch-load durable tool_calls for all transcript rows.
        # `include_ephemeral=False` enforces the north star rule: Previous
        # Messages must only surface ephemeral=0 rows (narration,
        # user_steer, tool_compaction, act_restart, and batched LLM tool
        # results are audit-only and never replay in future context).
        all_ids = [e['id'] for e in entries if e.get('id')]
        durable_by_id: dict[int, list] = {}
        if all_ids:
            tcs = ToolCallService()
            durable_by_id = tcs.get_by_transcript_ids(
                all_ids, include_ephemeral=False
            )

        lines: list[str] = []

        # Prepend compacted summary if it exists
        if compaction and compaction.get('compacted_text'):
            lines.append(compaction['compacted_text'])

        for entry in entries:
            ts = _format_ts(entry.get('created_at'), row_kind='transcript', row_id=entry.get('id'))
            raw_role = (entry.get('role') or 'unknown')
            role_label = 'Assistant' if raw_role == 'assistant' else raw_role
            content = (entry.get('content') or '').replace('\n', ' ').strip()
            lines.append(f"[{ts}] {role_label}: {content}")

            # Interleave durable tool_calls under this transcript row.
            # Hard filter: compaction pseudo-tool DTOs NEVER surface in
            # Previous Messages — their content is already replayed via the
            # `compactions` table prepend above. (Decision 4B — compaction
            # tool must NEVER make it to Previous Messages.)
            for tc in durable_by_id.get(entry.get('id'), []):
                tc_name = tc.get('tool_name') or tc.get('name') or 'tool'
                if tc_name in _NEVER_RENDER_IN_PREVIOUS:
                    continue
                tc_params = _parse_tc_params(tc.get('params'))
                tc_result = tc.get('result') or ''
                lines.append(
                    ToolRenderAndRecordService.render_static(
                        tc_name, tc_params, tc_result
                    )
                )

        return '\n'.join(lines)

    # ── Tool dispatch ──────────────────────────────────────────────────────────

    def handleTool(self, tc: dict) -> str:
        """Dispatch a single LLM tool call, record it, and add to context.

        Uses self._dispatcher (created once per turn in send()) so that
        tools discovered mid-turn via find_tools are registered as handlers
        and dispatched through the same path as innate skills.

        Never re-raises — errors become strings the LLM sees next iteration.
        """
        import time as _time
        from uuid import uuid4
        from services.tool_render_and_record_service import ToolRenderAndRecordService

        tool_name = (tc.get('name') if isinstance(tc, dict) else None) or 'unknown'
        tc_input = tc.get('input', {}) if isinstance(tc, dict) else {}
        if not isinstance(tc_input, dict):
            tc_input = {}
        tc_input = _sanitize_llm_args(tc_input)

        self._metrics.record_tool(tool_name)

        # Stable call_id: prefer the id field from the LLM; mint one if absent.
        call_id = (tc.get('id') if isinstance(tc, dict) else None) or uuid4().hex[:12]
        t_start = _time.monotonic()
        self._emit_tool_event({
            'type': 'act_tool_start',
            'call_id': call_id,
            'name': tool_name,
            'iter': self._current_iteration,
        })

        ok = True
        result_text = ''
        try:
            # 1. Dispatch via the per-turn dispatcher
            dispatch = self._dispatcher.dispatch_action(
                self.CHANNEL, {'type': tool_name, **tc_input}
            )
            result_text = str(dispatch.get('result', ''))

            # find_tools side effect — inject discovered schemas AND
            # register discovered tools as handlers on self._dispatcher
            # so subsequent iterations dispatch through the same path.
            if (
                tool_name == 'find_tools'
                and dispatch.get('status') == 'success'
            ):
                discovered = dispatch.get('_discovered_tools', [])
                if discovered:
                    from abilities._registry import AbilityRegistry
                    existing_names = {
                        t.get('name') for t in self._discovered_tools
                    }
                    for name in discovered:
                        if name in existing_names:
                            continue
                        try:
                            a = AbilityRegistry.get(name)
                            schema = {
                                'name': a.NAME,
                                'description': a.SUMMARY,
                                'input_schema': a.INPUT_SCHEMA,
                            }
                        except KeyError:
                            continue
                        self._discovered_tools.append(schema)
                        existing_names.add(name)

        except Exception as exc:
            ok = False
            result_text = f"ERROR: {tool_name} failed: {exc}"
            logger.error(
                "[MessageProcessor.handleTool] tool=%s raised: %s",
                tool_name, exc, exc_info=True,
            )
        finally:
            self._emit_tool_event({
                'type': 'act_tool_end',
                'call_id': call_id,
                'ms': int((_time.monotonic() - t_start) * 1000),
                'ok': ok,
            })

        # 2. Render + Record
        try:
            rendered = ToolRenderAndRecordService(
                tool_name=tool_name,
                params=tc_input,
                result=result_text,
                ephemeral=True,
                transcript_id=self._uid,
            ).renderAndRecord()
        except Exception as exc:
            rendered = f"[{tool_name}()] {result_text}"
            logger.error(
                "[MessageProcessor.handleTool] renderAndRecord failed tool=%s: %s",
                tool_name, exc, exc_info=True,
            )

        # 3. Add to context
        self._act_trail.append(rendered)
        return result_text

    def send(self, request_id: str | None = None) -> str:
        """Run the full turn: memory seed → ACT loop → store → postTurn.

        Two-stage mid-ACT compaction (north star § Context Compaction):
        - At the start of every iteration, the rebuilt user-message body
          (wrapped by ``_wrap_with_checkpoint``) is measured against 80%
          of the provider's context window for ``self.JOB``.
        - Over threshold → Stage 1 (``_run_stage1_tool_compaction``) trims
          the accumulated tool-use trail in place, preserving ephemeral=0
          DTOs and ``user_steer`` entries.
        - Re-measure after Stage 1. Still over → Stage 2
          (``_run_stage2_act_restart``) writes a full checkpoint via
          ``_run_full_compaction``, collapses ephemeral=1 DTOs (preserving
          ``user_steer`` and ephemeral=0) into a single ``act_restart``
          DTO, clears the trail + discovered tools, and resets
          ``iteration`` to 0 for a clean loop restart.
        - ``loop_start`` is anchored AFTER ``_run_thinking_gate()`` completes.
          The exploration pass runs under its own ``THINKING_TIMEOUT`` envelope
          (independent budget). MAX_TIMEOUT covers only ACT loop iterations.
        - ``loop_start`` is intentionally NOT reset on Stage 2 restart —
          the MAX_TIMEOUT wall-clock guard keeps ticking so runaway turns
          eventually hit the cap.
        - If Stage 2 fails (LLM error) the loop breaks to the cap-exit
          path and returns ``final_text=''``. Retrying the provider call
          against an oversize user_body would almost certainly fail and
          waste budget.

        Exception semantics:
        - Provider errors (inside the main ACT loop) propagate immediately.
          store() is NOT called; no partial rows land. The atomic-turn
          guarantee from Commit 4 holds.
        - Compaction LLM errors are trapped inside the helpers and logged;
          Stage 1 degrades silently (loop continues with uncompacted trail);
          Stage 2 returns False so the loop breaks to cap exit.
        - handleTool() errors are already trapped inside handleTool().
        - _emit_narration() errors are logged and swallowed — never kill
          the ACT loop (north star § ACT Loop step 4).
        - store() errors propagate. If store() raises, postTurn() is NOT
          called and the exception surfaces to the caller.
        - postTurn() errors are caught and logged. The turn is already
          stored; the caller still receives the final response.

        Final-text semantics:
        - Clean exit (LLM returned text with no tool_calls) → final_text is
          the terminating text.
        - Cap exit (MAX_ITERATIONS / MAX_TIMEOUT / Stage 2 failure) →
          final_text is ''. The last iteration's mid-loop narration is
          NEVER stored as the assistant row — that text represents partial
          thinking, not a final answer. Storing it would violate north star
          § Storage Model ("final ACT-loop response only").
        """
        from services.act_dispatcher_service import ActDispatcherService
        from services.providers import Providers
        from services.tool_render_and_record_service import ToolRenderAndRecordService
        from services.transcript_service import write_input_row

        with bind_current_processor(self):
            # Write input row BEFORE the loop so transcript_id is available
            # for ToolRenderAndRecordService during tool dispatch.
            # Skipped for internal processors (SKIP_TRANSCRIPT_WRITE=True) so
            # they leave no trace in the transcript table.
            if not self.SKIP_TRANSCRIPT_WRITE:
                self._uid = write_input_row(self.CHANNEL, self.ROLE, self._raw_input)

            # Single dispatcher for the entire turn. Tools discovered
            # mid-turn via find_tools are registered as handlers on this
            # instance so all tools dispatch through the same path.
            self._dispatcher = ActDispatcherService()

            with self._metrics.stage('pre_act'):
                self.pre_act()
            with self._metrics.stage('thinking_gate'):
                self._run_thinking_gate()   # CHANNEL='user' only, guarded internally

            # Anchor MAX_TIMEOUT AFTER the thinking gate. The exploration pass
            # runs under its own THINKING_TIMEOUT envelope (independent budget),
            # so MAX_TIMEOUT covers only ACT loop iterations.
            loop_start = time.time()

            # Log exploration injection once per turn — at this single point,
            # not inside the ACT loop.
            if self._thinking_exploration:
                logger.info(
                    "[THINKING] Chain of Thought injected into user body (chars=%d)",
                    len(self._thinking_exploration),
                )

            raw_limit = Providers.instance().get_context_limit(job=self.JOB)
            context_limit: int = (
                int(raw_limit)
                if isinstance(raw_limit, (int, float)) and raw_limit > 0
                else 32_000
            )

            self._current_iteration = 0
            llm_response = None
            loop_exited_cleanly = False

            while (
                self._current_iteration < self.MAX_ITERATIONS
                and time.time() - loop_start < self.MAX_TIMEOUT
            ):
                with self._metrics.iteration(self._current_iteration):
                    with self._metrics.stage('prompt_assembly'):
                        user_body = self.getUserPrompt()
                        user_body = _wrap_with_exploration(self.CHANNEL, user_body)
                        user_body = _wrap_with_checkpoint(self.CHANNEL, user_body)

                    # Two-stage mid-ACT compaction: triggered when the rendered
                    # user-message body (including checkpoint envelope) exceeds
                    # 80% of the provider's context window.
                    #
                    # Stage 1: compress the accumulated tool-use trail in place.
                    # Stage 2 (fallback): full checkpoint compaction + loop restart.
                    # Stage 2 resets iteration to 0 but NOT loop_start — the
                    # wall-clock MAX_TIMEOUT guard is a safety net against runaway
                    # turns; compaction LLM calls count against it deliberately.
                    if self._check_threshold(user_body, context_limit):
                        logger.warning(
                            "[COMPACTION] %s: user body over 80%% threshold "
                            "(ctx_limit=%d) — running Stage 1",
                            self.CHANNEL, context_limit,
                        )
                        self._run_stage1_tool_compaction()
                        # Re-render after Stage 1 trim and re-check threshold.
                        with self._metrics.stage('prompt_assembly'):
                            user_body = self.getUserPrompt()
                            user_body = _wrap_with_exploration(self.CHANNEL, user_body)
                            user_body = _wrap_with_checkpoint(self.CHANNEL, user_body)
                        if self._check_threshold(user_body, context_limit):
                            logger.warning(
                                "[COMPACTION] %s: still over threshold after Stage 1 "
                                "— running Stage 2 (ACT restart)",
                                self.CHANNEL,
                            )
                            if self._run_stage2_act_restart():
                                self._current_iteration = 0
                                continue
                            # Stage 2 failed (compaction LLM error). Retrying the
                            # main provider call against an over-threshold body
                            # would almost certainly fail too — break to cap exit
                            # and return final_text=''.
                            logger.error(
                                "[COMPACTION] %s: Stage 2 failed — breaking to "
                                "cap exit (final_text='')",
                                self.CHANNEL,
                            )
                            break

                    with self._metrics.stage('prompt_assembly'):
                        system_prompt = self.getSystemPrompt()
                        tools = self.getTools()

                    # Single-element messages[] so the provider sees one user turn
                    # containing the full literal-text body (Previous Messages,
                    # memory seed, raw input, ACT trail). The provider's multi-turn
                    # interface is intentionally NOT used — history lives in the
                    # getUserPrompt() text block, not in the messages array.
                    messages = [{'role': 'user', 'content': user_body}]

                    # Provider errors propagate here. store() is not called if
                    # this raises — the turn leaves no trace in the DB.
                    # PayloadTooLargeError (HTTP 413) is the one exception: the
                    # provider's transport-level body cap was hit (e.g. Ollama
                    # Cloud edge proxy), so we run a Stage 2 ACT restart once
                    # to compact the trail and retry. A second 413 after that
                    # means even the compacted body is too big — break to cap
                    # exit rather than loop forever.
                    try:
                        llm_response = Providers.instance().send_messages(
                            system_prompt, messages, job=self.JOB, tools=tools,
                            thinking_mode=self._get_thinking_mode_for_send(),
                        )
                    except PayloadTooLargeError as exc:
                        if self._payload_too_large_recovered:
                            logger.error(
                                "[COMPACTION] %s: PayloadTooLargeError after Stage 2 "
                                "restart — breaking to cap exit (final_text=''): %s",
                                self.CHANNEL, exc,
                            )
                            break
                        logger.warning(
                            "[COMPACTION] %s: PayloadTooLarge from provider — "
                            "running Stage 2 ACT restart (%s)",
                            self.CHANNEL, exc,
                        )
                        self._payload_too_large_recovered = True
                        if self._run_stage2_act_restart():
                            self._current_iteration = 0
                            continue
                        logger.error(
                            "[COMPACTION] %s: Stage 2 failed after PayloadTooLarge "
                            "— breaking to cap exit (final_text='')",
                            self.CHANNEL,
                        )
                        break
                    self._metrics.accumulate(llm_response)

                    if not llm_response.tool_calls:
                        loop_exited_cleanly = True
                        break

                    # Narration text BEFORE tool dispatch — the LLM emitted the
                    # narration in its response ahead of the tool_use block, so
                    # the stored timeline must reflect that semantic order. The
                    # transcript-timeline example in the north star § Storage
                    # Model shows the narration DTO preceding the tool_call DTOs
                    # for the same iteration.
                    with self._metrics.stage('post_tool_records'):
                        if llm_response.text:
                            rendered = ToolRenderAndRecordService(
                                tool_name='narration',
                                params={},
                                result=llm_response.text,
                                ephemeral=True,
                                transcript_id=self._uid,
                            ).renderAndRecord()
                            self._act_trail.append(rendered)
                            try:
                                self._emit_narration(llm_response.text, self._current_iteration)
                            except Exception as exc:
                                logger.error(
                                    "[MessageProcessor.send] _emit_narration raised: %s",
                                    exc, exc_info=True,
                                )

                        for tc in llm_response.tool_calls:
                            self.handleTool(tc)  # never raises; appends DTO + trail

                        for steer in self._drain_steering(request_id):
                            rendered = ToolRenderAndRecordService(
                                tool_name='user_steer',
                                params={},
                                result=steer,
                                ephemeral=True,
                                transcript_id=self._uid,
                            ).renderAndRecord()
                            self._act_trail.append(rendered)

                    self._current_iteration += 1

            if loop_exited_cleanly:
                final_text = (llm_response.text or '') if llm_response else ''
            else:
                # Cap exit — no clean terminating text. The last iteration's
                # narration is already captured as a narration DTO; we
                # must NOT re-use it as the assistant row.
                logger.warning(
                    "[MessageProcessor.send] ACT loop hit safety cap "
                    "(iteration=%d, elapsed=%.1fs, max_iter=%d, max_timeout=%d) — "
                    "final_text set to '' to avoid persisting mid-loop narration "
                    "as assistant response",
                    self._current_iteration,
                    time.time() - loop_start,
                    self.MAX_ITERATIONS,
                    self.MAX_TIMEOUT,
                )
                final_text = ''

            with self._metrics.stage('store'):
                self.store(final_text)
            try:
                with self._metrics.stage('post_turn'):
                    self.postTurn()
            except Exception as e:
                logger.error(
                    "[POSTTURN] Failed (turn already stored): %s", e, exc_info=True
                )
            return final_text

    def _emit_narration(self, text: str, iteration: int) -> None:
        """Base no-op. UserMessageProcessor overrides in Commit 8 to push
        mid-loop text to the websocket via the on_narration callback."""
        pass

    def _emit_tool_event(self, event: dict) -> None:
        """Subclasses MUST override BOTH this method AND the
        ``_on_tool_event`` attribute. The base no-op never reads any
        attribute."""
        pass

    def _drain_steering(self, request_id: str | None) -> list[str]:
        """Return any mid-loop steering messages queued for this request.

        Commit 6 stub — base always returns [] so DMN, goal-pursuit, and
        scheduled subclasses silently produce no user_steer DTOs. Will be
        wired to the user_input_queue MemoryStore key in Commit 8 (or a
        follow-up) inside UserMessageProcessor, which overrides this method
        to drain ``steer:{request_id}`` from MemoryStore.
        """
        return []

    # ── Commit 7: two-stage mid-ACT compaction ────────────────────────────────

    def _measure_user_message(self, user_msg: str) -> int:
        """Return a fast token estimate for the rendered user-message body."""
        from services.llm_service import estimate_tokens
        return estimate_tokens(user_msg)

    def _check_threshold(self, user_msg: str, context_limit: int) -> bool:
        """Return True if the user-message body exceeds 80% of context_limit.

        Strict greater-than: a message exactly at 80% is NOT compaction-eligible.
        """
        return self._measure_user_message(user_msg) > int(context_limit * 0.80)

    def _run_full_compaction(self) -> 'str | None':
        """Run a full checkpoint compaction for this channel.

        Orchestrator: reads prior compaction + entries since watermark,
        formats the LLM input, dispatches to ``FullCompactionProcessor``,
        then writes the ``compactions`` row + ``tool_calls`` audit row from
        the returned text.

        Returns the compacted text on success, None on failure.
        Records via ToolRenderAndRecordService (ephemeral=False).
        """
        from services import compaction_persistence
        from services.compaction_message_processor import FullCompactionProcessor
        from services.database_service import get_shared_db_service
        from services.llm_service import estimate_tokens
        from services.providers import Providers

        prior = compaction_persistence.get_compaction(self.CHANNEL)
        watermark = prior['compacted_up_to_id'] if prior else 0
        prev_text = (prior.get('compacted_text') or '').strip() if prior else ''

        entries = list(compaction_persistence.get_entries_since(self.CHANNEL, watermark))

        # Nothing to compact — bail before hitting the LLM. Without this guard
        # we would send a bare "## New Conversation Turns" header to the
        # compaction LLM and overwrite the existing checkpoint with whatever
        # it hallucinates.
        if not entries and not prev_text:
            logger.warning(
                "[COMPACTION] %s: _run_full_compaction called with no entries "
                "and no prior checkpoint — skipping LLM call",
                self.CHANNEL,
            )
            return None

        # Refresh the context limit — Stage 2 is a rare path and the provider
        # may have been reconfigured since send() cached the value. Use the
        # compaction processor's JOB (not self.JOB) so the cap matches the LLM
        # that will actually service the call.
        raw_limit = Providers.instance().get_context_limit(job=FullCompactionProcessor.JOB)
        context_limit = (
            int(raw_limit)
            if isinstance(raw_limit, (int, float)) and raw_limit > 0
            else 32_000
        )
        input_cap = int(context_limit * 0.90)

        def _format_entry(entry: dict) -> str:
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            raw_ts = entry.get('created_at') or ''
            ts_label = TimeFormatterService.local(raw_ts) or _MISSING_TS_PLACEHOLDER
            return f"[{ts_label}] {role}: {content}"

        def _build_input(rendered_entries: list[str]) -> str:
            chunks: list[str] = []
            if prev_text:
                chunks.append(f"## Previous Summary\n{prev_text}")
            chunks.append("## New Conversation Turns")
            chunks.extend(rendered_entries)
            return '\n\n'.join(chunks)

        rendered = [_format_entry(e) for e in entries]
        compaction_input = _build_input(rendered)

        # Enforce the 90% cap — drop oldest rows first. The previous
        # checkpoint (prev_text) is never dropped.
        dropped = 0
        while (
            rendered
            and estimate_tokens(compaction_input) > input_cap
        ):
            rendered.pop(0)
            dropped += 1
            compaction_input = _build_input(rendered)
        if dropped:
            logger.warning(
                "[COMPACTION] %s: dropped %d oldest entries to fit 90%% cap "
                "(cap=%d tokens)",
                self.CHANNEL, dropped, input_cap,
            )

        # After trimming, if nothing remains there is nothing to compact —
        # even if a prior checkpoint exists it is already up-to-date. Bail
        # without hitting the LLM; the next turn can retry as new entries
        # accumulate below the 90% cap.
        if not rendered:
            logger.warning(
                "[COMPACTION] %s: 90%% cap dropped every new entry — "
                "skipping LLM call (prior checkpoint preserved)",
                self.CHANNEL,
            )
            return None

        proc = FullCompactionProcessor(raw_input=compaction_input)
        try:
            compacted_text = (proc.send() or '').strip()
        except PayloadTooLargeError as exc:
            # Compaction LLM itself rejected the body — payload was already
            # over the transport cap before we could compact it. Distinct
            # log so on-call doesn't conflate this with a generic LLM error.
            logger.error(
                "[COMPACTION] %s: compaction LLM hit HTTP 413 — cannot "
                "compact further; turn will hit cap exit: %s",
                self.CHANNEL, exc,
            )
            return None
        except Exception as exc:
            logger.error(
                "[COMPACTION] %s: LLM call failed during _run_full_compaction: %s",
                self.CHANNEL, exc,
                exc_info=True,
            )
            return None
        finally:
            self._metrics.merge(proc._metrics)

        if not compacted_text:
            logger.warning(
                "[COMPACTION] %s: LLM returned empty compaction text", self.CHANNEL
            )
            return None

        # Watermark advances only to the highest row we actually fed the LLM.
        # Dropped rows stay unseen so the next compaction re-reads them.
        if rendered:
            # ``rendered`` and ``entries`` were sliced in lockstep — after
            # ``dropped`` pops, entries[dropped:] is the consumed slice.
            consumed = entries[dropped:]
            new_watermark = max(e.get('id', 0) for e in consumed)
        else:
            new_watermark = watermark

        token_count = estimate_tokens(compacted_text)

        # Upsert — omit overflow_content from SET clause (legacy field, do not touch).
        try:
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO compactions
                        (channel, compacted_text, compacted_up_to_id, token_count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(channel) DO UPDATE SET
                        compacted_text      = excluded.compacted_text,
                        compacted_up_to_id  = excluded.compacted_up_to_id,
                        token_count         = excluded.token_count,
                        updated_at          = excluded.updated_at
                    """,
                    (
                        self.CHANNEL,
                        compacted_text,
                        new_watermark,
                        token_count,
                        utc_now().isoformat(),
                    ),
                )
                cursor.close()
        except Exception as exc:
            logger.error(
                "[COMPACTION] %s: failed to upsert compactions row: %s",
                self.CHANNEL, exc,
                exc_info=True,
            )
            return None

        from services.tool_render_and_record_service import ToolRenderAndRecordService
        ToolRenderAndRecordService(
            tool_name='compaction',
            params={},
            result=compacted_text,
            ephemeral=False,
            transcript_id=self._uid,
        ).renderAndRecord()

        logger.info(
            "[COMPACTION] %s: full compaction written, watermark=%d, tokens=%d",
            self.CHANNEL, new_watermark, token_count,
        )
        return compacted_text

    def _run_stage1_tool_compaction(self) -> None:
        """Stage 1 mid-ACT compaction: compress the tool-use trail via LLM.

        Orchestrator: dispatches to ``TrailCompactionProcessor``, then
        replaces ``self._act_trail`` with the returned summary. On failure,
        returns without mutating state.
        """
        from services.compaction_message_processor import TrailCompactionProcessor

        trail_text = '\n'.join(self._act_trail)
        if not trail_text.strip():
            return

        proc = TrailCompactionProcessor(raw_input=trail_text)
        try:
            summary_text = (proc.send() or '').strip()
        except Exception as exc:
            logger.warning(
                "[COMPACTION] %s: Stage 1 tool compaction LLM call failed: %s",
                self.CHANNEL, exc,
            )
            return
        finally:
            self._metrics.merge(proc._metrics)

        if not summary_text:
            logger.warning(
                "[COMPACTION] %s: Stage 1 tool compaction returned empty summary",
                self.CHANNEL,
            )
            return

        from services.tool_render_and_record_service import ToolRenderAndRecordService
        rendered = ToolRenderAndRecordService(
            tool_name='tool_compaction',
            params={},
            result=summary_text,
            ephemeral=True,
            transcript_id=self._uid,
        ).renderAndRecord()

        self._act_trail = [rendered]

    def _run_stage2_act_restart(self) -> bool:
        """Stage 2: full compaction + ACT loop restart.

        Returns True → reset iteration to 0. False → compaction failed.
        """
        compacted_text = self._run_full_compaction()
        if compacted_text is None:
            logger.warning(
                "[COMPACTION] %s: Stage 2 _run_full_compaction returned None — "
                "continuing loop without restart; turn will likely hit cap",
                self.CHANNEL,
            )
            return False

        from services.tool_render_and_record_service import ToolRenderAndRecordService
        ToolRenderAndRecordService(
            tool_name='act_restart',
            params={},
            result='ACT loop restarted after context compaction',
            ephemeral=True,
            transcript_id=self._uid,
        ).renderAndRecord()

        self._act_trail = []
        self._discovered_tools = []
        # Exploration text was generated against the now-collapsed pre-restart
        # context and has already been summarised into the checkpoint. Keeping
        # it would re-inflate every subsequent user_body via
        # _wrap_with_exploration() — directly defeating the recovery for the
        # 413 path (large exploration block alone can exceed the cloud cap).
        self._thinking_exploration = None

        return True

    def store(self, llm_response: str) -> None:
        """Write the assistant transcript row. Input row was already written
        at the top of send(). Tool calls were recorded inline via
        ToolRenderAndRecordService during the ACT loop.

        When SKIP_TRANSCRIPT_WRITE is True (internal processors), this is a
        no-op — no rows are written and self._uid remains None.
        """
        if self.SKIP_TRANSCRIPT_WRITE:
            self._uid = None
            return
        from services.transcript_service import write_assistant_row
        write_assistant_row(self.CHANNEL, llm_response)

    # ── Overridable hooks ────────────────────────────────────────────────────

    def pre_act(self) -> None:
        """Pre-ACT-loop hook. Default is a no-op.

        Called from send() after the input transcript row is written
        (self._uid is populated) but before the ACT loop starts.
        UserMessageProcessor overrides to run the memory seed via the
        canonical tool dispatch path.
        """
        pass

    # ── Thinking-gate (CHANNEL='user' only) ──────────────────────────────────

    def _run_thinking_gate(self) -> None:
        """Regression-head deliberation scoring. Writes self._thinking_level.

        No-op for non-user channels (classifier is OOD for autonomous flows).
        Never raises. On failure → self._thinking_level = 'low', EMA untouched.
        """
        if self.CHANNEL != 'user':
            return

        try:
            from services.deliberation_score_service import DeliberationScoreService
            from services.deliberation_ema_service import DeliberationEmaService

            scalar = DeliberationScoreService().classify(self._raw_input)
            ema_svc = DeliberationEmaService()

            if scalar is None:
                self._thinking_level = 'low'
                self._deliberation_scalar = None
                self._deliberation_ema = ema_svc.peek()
                logger.info(
                    "[DELIBERATION] turn=%s scalar=None ema=%s bucket=low fallback=true",
                    self._uid, self._deliberation_ema,
                )
                self._thinking_exploration = None
                return

            ema, bucket = ema_svc.update_and_bucket(scalar)
            self._thinking_level = bucket
            self._deliberation_scalar = scalar
            self._deliberation_ema = ema
            logger.info(
                "[DELIBERATION] turn=%s scalar=%.4f ema=%.4f bucket=%s fallback=false",
                self._uid, scalar, ema, bucket,
            )

            if self._thinking_level == 'high':
                try:
                    with ThreadPoolExecutor(max_workers=1) as _pool:
                        _future = _pool.submit(self._run_thinking_exploration)
                        try:
                            self._thinking_exploration = _future.result(
                                timeout=self.THINKING_TIMEOUT
                            )
                        except FuturesTimeoutError:
                            logger.warning(
                                "[THINKING] exploration exceeded THINKING_TIMEOUT=%ds"
                                " — proceeding without exploration",
                                self.THINKING_TIMEOUT,
                            )
                            self._thinking_exploration = None
                except Exception as exc:
                    logger.info(
                        "[THINKING] exploration failed (%s) — high turn proceeds "
                        "without exploration", exc,
                    )
                    self._thinking_exploration = None
                if self._thinking_exploration is not None:
                    self._persist_exploration_to_tool_calls(self._uid)
            else:
                self._thinking_exploration = None

            if self._uid is not None:
                from services.database_service import get_shared_db_service
                try:
                    db = get_shared_db_service()
                    with db.connection() as conn:
                        conn.execute(
                            "UPDATE transcript SET deliberation_score = ? WHERE id = ?",
                            (scalar, self._uid),
                        )
                except Exception as exc:
                    logger.warning(
                        "[DELIBERATION] persist failed for uid=%s: %s",
                        self._uid, exc,
                    )

        except Exception:
            logger.exception("[DELIBERATION] gate failed; defaulting to 'low'")
            self._thinking_level = 'low'
            self._deliberation_scalar = None
            self._thinking_exploration = None

    def _run_thinking_exploration(self) -> 'str | None':
        """One same-job exploration pass for high-mode turns.

        Asks the model to think out loud about the user's request: assess
        gaps in its knowledge, evaluate which tools would help, and flag
        non-obvious aspects. Output is Chain-of-Thought that gets
        re-injected into the ACT loop via _wrap_with_exploration so the
        model can act on its own reasoning.

        Tools schema is sent so the model can reason about available
        capabilities, but the prompt instructs it not to invoke them.
        Any tool_calls in the response are discarded (single-pass only).

        The model may output 'NOTHING' if the request is straightforward,
        in which case None is returned and no exploration is injected.

        Returns None on any failure (network, provider rejection, etc).
        Logged at INFO. NEVER raises.
        """
        from services.providers import Providers

        _EXPLORATION_PREFIX = (
            "Think out loud about the user's request before responding.\n\n"
            "Consider:\n"
            "- What does the ideal response look like? What would make it genuinely useful?\n"
            "- Do you already know enough to answer well, or are there gaps?\n"
            "- Would any of your available tools fill those gaps? Which ones, in what order?\n"
            "- Is there anything non-obvious about this request you might miss on a first read?\n\n"
            "Whatever you output here will be shown to you as Chain of Thought on the next "
            "pass — write to your future self. Be specific: name the tools you plan to use, "
            "flag uncertainties, note key facts you want to remember to include.\n\n"
            "If the request is straightforward and you have nothing useful to say to yourself, "
            "output exactly: NOTHING\n\n"
            "DO NOT INVOKE TOOLS — they are disabled in this phase. Think only."
            "\n\n---\n\n"
        )

        try:
            user_body = self.getUserPrompt()
            user_body = _wrap_with_checkpoint(self.CHANNEL, user_body)
            system_prompt = self.getSystemPrompt()
            tools = self.getTools()

            response = Providers.instance().send_messages(
                system_prompt,
                [{'role': 'user', 'content': _EXPLORATION_PREFIX + user_body}],
                job=self.JOB,
                tools=tools,
                thinking_mode='high',
            )
            self._metrics.accumulate(response)

            if response.tool_calls:
                logger.debug(
                    "[THINKING] exploration model attempted %d tool call(s) — discarded",
                    len(response.tool_calls),
                )

            text = (response.text or '').strip()
            if text.upper() == 'NOTHING':
                return None
            return text if text else None

        except Exception as exc:
            logger.info("[THINKING] exploration failed (%s)", exc)
            return None

    def _get_thinking_mode_for_send(self) -> 'str | None':
        """Map self._thinking_level → provider thinking_mode kwarg.

        Only fires on the user channel — non-user channels (DMN, scheduler,
        cron, goal-pursuit, internal flows) get None so background work
        stays cheap.
        """
        if self.CHANNEL != 'user':
            return None
        if self._thinking_level == 'high':
            return 'high'
        if self._thinking_level == 'medium':
            return 'medium'
        return None

    def _persist_exploration_to_tool_calls(self, transcript_id: 'int | None') -> None:
        """Insert the exploration text as a durable tool_calls row.

        Stored with tool_name='thinking', ephemeral=0 so it survives
        compaction and surfaces as part of the durable audit trail.
        Persistence failure logs INFO and does NOT abort the turn.
        """
        if transcript_id is None or self._thinking_exploration is None:
            return
        from services.tool_render_and_record_service import ToolRenderAndRecordService
        try:
            ToolRenderAndRecordService(
                tool_name='thinking',
                params={},
                result=self._thinking_exploration,
                ephemeral=False,
                transcript_id=transcript_id,
            ).renderAndRecord()
        except Exception as exc:
            logger.info(
                "[THINKING] failed to persist exploration to tool_calls (%s)", exc
            )

    def postTurn(self) -> None:
        """Per-channel post-turn service fan-out.

        Base is a no-op. UserMessageProcessor overrides in Commit 8 with the
        eight-service fan-out (LUT canonicalization via memory skill, phase
        updates, etc.). Each subclass is the sole orchestrator of its own tail.
        """
        pass


# ── Module-private helpers ────────────────────────────────────────────────────


#: Placeholder rendered when a row has a missing / empty / unparseable
#: ``created_at`` value. Must be exactly 16 characters so the
#: ``[YYYY-MM-DD HH:MM]`` column width in Previous Messages stays stable.
_MISSING_TS_PLACEHOLDER = '????-??-?? ??:??'


#: Durable tool_call names that **must never** surface in Previous Messages.
#:
#: ``compaction`` — stored ``ephemeral=0`` for audit purposes, but its content
#: is already replayed to the LLM through the ``### Checkpoint`` envelope built
#: by ``_wrap_with_checkpoint``. Letting it also render in Previous Messages
#: would duplicate the summary on every subsequent turn. (Decision 4B —
#: resolved 2026-04-10.)
#:
#: ``thinking`` — the pre-turn exploration block is stored ``ephemeral=0`` as
#: an audit row via ``_persist_exploration_to_tool_calls``, but is prepended
#: live into the ACT-loop user body via ``_wrap_with_exploration`` on every
#: iteration (cheap string-prefix, single LLM call before the loop). Rendering
#: it again in Previous Messages would double-inject the exploration text and
#: pollute the transcript for the LLM.
#:
#: ``tool_compaction`` and ``act_restart`` do NOT need to be listed here —
#: both are stored ``ephemeral=1`` and are already filtered out of Previous
#: Messages by the durable-only query in ``getPreviousMessages``.
_NEVER_RENDER_IN_PREVIOUS: frozenset[str] = frozenset({'compaction', 'thinking'})


def _wrap_with_checkpoint(channel: str, user_body: str) -> str:
    """Wrap the user-message body with a ### Checkpoint envelope.

    When a compaction row exists for ``channel``, prepends the compacted
    summary under a ``### Checkpoint`` header and places the bare body
    under a ``### Current State`` header. Returns the bare body unchanged
    when there is no checkpoint or the stored compacted_text is empty.

    Called by ``send()`` on every ACT iteration, immediately after
    ``getUserPrompt()`` returns.
    """
    from services import compaction_persistence

    row = compaction_persistence.get_compaction(channel)
    if not row:
        return user_body
    compacted = (row.get('compacted_text') or '').strip()
    if not compacted:
        return user_body
    return (
        "### Checkpoint - What you were previously discussing / doing\n"
        f"{compacted}\n"
        "\n"
        "---\n"
        "### Current State - What's happening in the current turn\n"
        f"{user_body}"
    )


def _wrap_with_exploration(channel: str, user_body: str) -> str:
    """Prepend the thinking exploration block to the user-message body.

    Channel-gated: returns ``user_body`` unchanged for any non-user channel so
    background flows (DMN, goal-pursuit, scheduled) are never affected.

    Reads ``_thinking_exploration`` from the active processor via
    ``current_processor()``. Returns ``user_body`` unchanged when:
    - ``channel != 'user'``
    - no active processor is bound (called outside a turn)
    - ``_thinking_exploration`` is falsy (None, empty string)

    The exploration LLM call runs ONCE per turn (inside _run_thinking_gate).
    This helper is called on each ACT iteration to keep the already-computed
    exploration text in the user body. The INFO log for the injection is
    emitted once pre-loop inside send() — not here.

    Apply this wrapper BEFORE ``_wrap_with_checkpoint`` so the exploration block
    sits at the top of ``### Current State`` when a compaction exists.
    """
    if channel != 'user':
        return user_body
    proc = current_processor()
    if proc is None:
        return user_body
    exploration_text = getattr(proc, '_thinking_exploration', None)
    if not exploration_text:
        return user_body
    return (
        "## Chain of Thought\n"
        "Below is your initial reaction to this prompt, played back. "
        "Use it as grounding but pivot as needed based on the conversation.\n\n"
        "---\n\n"
        f"{exploration_text}\n\n"
        "---\n\n"
        + user_body
    )


def _parse_tc_params(raw: object) -> dict:
    """Parse the ``tool_calls.params`` column into a dict for rendering.

    The DB stores params as a JSON-encoded string. Callers may also pass a
    pre-parsed dict (tests mocking the service). This helper normalises both
    paths and returns ``{}`` on any parse failure — the rendered line becomes
    ``[tool_name()] result`` which is still valid per the north star format.
    """
    if raw is None or raw == '':
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _format_ts(
    raw: str | None,
    *,
    row_kind: str = 'row',
    row_id: int | None = None,
) -> str:
    """Format a raw SQLite/ISO timestamp into ``YYYY-MM-DD HH:MM`` in the user's
    local timezone.

    Storage is UTC (``utc_now().isoformat()``); the LLM only ever sees local
    wall-clock time. Conversion runs through
    :meth:`TimeFormatterService.local`, which handles tz lookup.

    If ``raw`` is ``None``, empty, or unparseable, return
    ``_MISSING_TS_PLACEHOLDER`` and emit a single warning log so the problem
    is visible in production without spamming. Rationale: ``parse_utc`` falls
    back to ``datetime.min`` (``0001-01-01 00:00``) on bad input, which would
    otherwise silently corrupt Previous Messages with a bogus "year 1" prefix
    the LLM would treat as real context.

    ``row_kind`` + ``row_id`` are logged but not shown to the LLM — the
    placeholder is the only thing the prompt sees.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        logger.warning(
            "[MessageProcessor._format_ts] missing created_at on %s id=%s — "
            "rendering placeholder", row_kind, row_id,
        )
        return _MISSING_TS_PLACEHOLDER

    formatted = TimeFormatterService.local(raw)
    if formatted is None:
        logger.warning(
            "[MessageProcessor._format_ts] unparseable created_at=%r on %s "
            "id=%s — rendering placeholder", raw, row_kind, row_id,
        )
        return _MISSING_TS_PLACEHOLDER

    return formatted

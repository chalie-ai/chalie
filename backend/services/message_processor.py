# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MessageProcessor — abstract base class for all LLM message processors.

Lifecycle: one instance per turn. Two turns never share the same object.
Do not add `.instance()` / singleton accessors.

Each input channel (WebSocket, DMN timer, goal pursuit, scheduled prompt, …)
constructs its own MessageProcessor subclass directly. The subclass hardcodes
its CHANNEL and ROLE, implements get_user_definition() and get_user_prompt(), and
fans out post-turn services via post_turn(). The base class provides the ACT
loop (send()), atomic persistence (store()), tool dispatch (handle_tool()), and
compaction primitives.
"""

import contextlib
import contextvars
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from services.llm_service import PayloadTooLargeError
from services.metrics_accumulator import MetricsAccumulator
from services.system_message_prompt import SystemMessagePrompt
from services.time_utils import utc_now
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

# ── Compaction summary parser ─────────────────────────────────────────────────
#
# Parses the <summary>…</summary> block produced by ContinuityCompactionProcessor.
# Used by _run_full_compaction to extract the stored result from raw LLM output.

_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>", re.IGNORECASE)
_COMPACTION_FAILURE_FMT = "[COMPACTION] %s: continuity failure — reason=%s"

# Maximum bytes fed to _SUMMARY_RE; bounds backtracking on malformed LLM output.
_SUMMARY_RE_CAP = 65_536


def _extract_compaction_summary(raw: 'str | None') -> 'str | None':
    """Extract the body of a <summary>…</summary> block from raw LLM output.

    Returns the stripped inner text on success.
    Returns None when:
    - raw is empty or None.
    - no <summary> tags are present in the output.
    """
    if not raw:
        return None
    m = _SUMMARY_RE.search(raw[:_SUMMARY_RE_CAP])
    return m.group(1).strip() if m else None


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
# dispatched by name via ``handle_tool()`` and must not receive the
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
    - Implement `get_user_prompt() -> str`
    - Implement `get_user_definition() -> str`

    Subclasses may:
    - Override `SYSTEM_PROMPT_CLASS` with a concrete `SystemMessagePrompt` subclass.
    - Override `ALWAYS_AVAILABLE` with a list of ability names injected as
      innate tools every iteration.
    - Override `DISCOVERABLE` with a list of ability names that ``find_tools``
      may surface for this processor at runtime.
    - Override `get_dynamic_tools()` to filter or augment discovered tools.
    - Override `post_turn()` to fan out per-channel post-turn services.
    """

    # ── Class constants (overridable by subclasses) ───────────────────────────

    LOG_LABEL: str = 'chat'
    # Usage class written to llm_call_log.usage_class for every LLM call made
    # by this processor. Override in subclasses to distinguish chat / subagent /
    # subconscious traffic in the cognition usage dashboard.
    USAGE_CLASS: str = 'chat'
    SYSTEM_PROMPT_CLASS = SystemMessagePrompt  # class reference, not instance
    # Ability names pre-injected as native tools on every ACT iteration.
    ALWAYS_AVAILABLE: list[str] = []
    # Ability names ``find_tools`` may surface for this processor at runtime.
    # ``find_tools`` itself is gated to ``WHERE name IN DISCOVERABLE`` so a
    # processor can never discover anything outside this list.
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS: int = 30
    ITERATION_TIMEOUT: int = 1800  # seconds — per-iteration safety wall (independent of ACT loop budget)
    THINKING_TIMEOUT: int = 600  # seconds — exploration pass budget (independent of ACT)
    # Ceiling for a single get_previous_messages() pull. Commit 7's compaction
    # budget assumes a row count this low — if a channel ever exceeds it we
    # want compaction to kick in, not an unbounded fetch.
    _TRANSCRIPT_FETCH_LIMIT: int = 2000

    # ── Subclass must set ─────────────────────────────────────────────────────

    CHANNEL: str = ''   # e.g. 'user', 'dmn', 'subagent', 'scheduled'
    ROLE: str = ''      # e.g. 'user', 'proactive_thought', 'subagent'

    # When True, send() skips write_input_row() and store() skips the
    # assistant row — self._uid stays None for the entire turn.
    # Set this on internal processors (EpisodeEncoderProcessor,
    # SuperEpisodeEncoderProcessor) that must not pollute the transcript.
    SKIP_TRANSCRIPT_WRITE: bool = False

    # When True, send() skips write_input_row() but store() still writes
    # the assistant row.  self._uid stays None (tool calls get transcript_id
    # NULL — safe, all downstream code guards on _uid is not None).
    SKIP_INPUT_ROW: bool = False

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        self._memory_seed: str | None = None
        # Raw recall query used by pre_act(); kept separate from _memory_seed
        # (which is the formatted tag block) so recall_episodes() can embed
        # the original query for drift computation rather than the block string.
        self._memory_seed_query: str | None = None
        # Tracks the current ACT loop iteration so handle_tool() can include
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
        # One-shot guard: any overflow recovery (proactive threshold trip
        # OR 413 from the provider) triggers a Stage 2 ACT restart, but only
        # once per turn. The proactive threshold path can mis-fire when the
        # static system_prompt + tools schema alone exceed compact_at — in
        # that case compaction shrinks user_body but the threshold still
        # trips on restart, and without this guard the loop spins forever.
        # After one recovery: send anyway and let the transport 413 path
        # decide whether the compacted body is genuinely too large.
        self._overflow_recovered_this_turn: bool = False
        # Accumulator starts immediately so exploration + compaction tokens count.
        self._metrics: MetricsAccumulator = MetricsAccumulator()
        # Per-instance deadline (seconds from epoch) for processors that want a
        # hard wall-clock cap (e.g. SubagentProcessor). None means no deadline.
        # Set by subclasses in __init__ after calling super().__init__().
        self._deadline: float | None = None
        # Cooperative cancellation flag. Set by stop endpoints to signal the
        # ACT loop to exit at the next iteration boundary. Never raises —
        # the loop checks is_set() at the top of each iteration.
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Signal the ACT loop to exit at the next iteration boundary.

        Public interface for stop endpoints — avoids reaching into the private
        ``_cancel_event`` attribute from outside the class hierarchy.
        """
        self._cancel_event.set()

    def set_turn_start(self, ts: float) -> None:
        """Override the accumulator start time (called from _handle_chat before thread spawn)."""
        self._metrics.start_time = ts

    # ── Abstract — subclass implements ───────────────────────────────────────

    def get_user_prompt(self) -> str:
        """Build the body of the user-message for the current ACT iteration.

        Called once per ACT iteration inside send(). Does NOT emit the
        ``### Checkpoint`` / ``### Current State`` headers — those are added
        by send().

        Override this method or its camelCase alias getUserPrompt().
        """
        if type(self).getUserPrompt is not MessageProcessor.getUserPrompt:
            return self.getUserPrompt()
        raise NotImplementedError

    def getUserPrompt(self) -> str:  # noqa: N802
        """CamelCase alias — override either this or get_user_prompt()."""
        if type(self).get_user_prompt is not MessageProcessor.get_user_prompt:
            return self.get_user_prompt()
        raise NotImplementedError

    def get_user_definition(self) -> str:
        """One-sentence description of who the 'user' is for this processor.

        Injected as the first line of the system prompt. Examples:
          UserMessageProcessor   → user synthesis string (real human)
          DMNMessageProcessor    → "The user is 'proactive_thought' — ..."
          SubagentProcessor      → "The user is 'subagent' — ..."

        Override this method or its camelCase alias getUserDefinition().
        """
        if type(self).getUserDefinition is not MessageProcessor.getUserDefinition:
            return self.getUserDefinition()
        raise NotImplementedError

    def getUserDefinition(self) -> str:  # noqa: N802
        """CamelCase alias — override either this or get_user_definition()."""
        if type(self).get_user_definition is not MessageProcessor.get_user_definition:
            return self.get_user_definition()
        raise NotImplementedError

    # ── Overridable hook ─────────────────────────────────────────────────────

    def get_dynamic_tools(self) -> list[dict]:
        """Return tool schemas discovered during this turn via find_tools.

        Default returns self._discovered_tools directly (identity — subclasses
        that mutate the list during a turn see the mutations reflected here).
        Subclasses may override to filter, replace, or suppress.
        """
        return self._discovered_tools

    # ── Final (concrete on base) ──────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Build the full system prompt for this turn.

        Prepends get_user_definition() to the body produced by SYSTEM_PROMPT_CLASS.
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
        `get_system_prompt()` themselves and weave in the extra context around the
        template returned by `SYSTEM_PROMPT_CLASS().get_prompt()`.
        """
        # Intentionally zero-arg — see docstring. Subclasses override this
        # method (not SYSTEM_PROMPT_CLASS's signature) to pass real context.
        body = self.SYSTEM_PROMPT_CLASS().get_prompt()
        body = self._substitute_provider_placeholders(body)
        return f"{self.get_user_definition()}\n\n{body}"

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
            provider = Providers.instance()._resolve(self.LOG_LABEL)
            label = getattr(provider, 'CONTENT_FIELD_LABEL', None)
        except Exception:
            label = None
        if not label:
            return body
        return body.replace("{{provider_content_field_name}}", label)

    def get_tools(self) -> list[dict]:
        """Return the full tool list for the current ACT iteration.

        Resolution order (first-seen wins on duplicates):
        1. ALWAYS_AVAILABLE — innate tier (resolved via AbilityRegistry).
           Base is []; subclasses set the explicit list of ability names that
           are pre-injected on every iteration.
        2. get_dynamic_tools() — abilities discovered this turn via find_tools.
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
                        "[MessageProcessor.get_tools] No ability registered for '%s'",
                        tool_name,
                    )

        dynamic = self.get_dynamic_tools()

        seen: set[str] = set()
        result: list[dict] = []
        for schema in native + dynamic:
            name = schema.get('name')
            if name and name not in seen:
                seen.add(name)
                result.append(schema)

        return result

    def get_act_loop_trail(self) -> str:
        """Return the ACT loop trail as a single string for prompt injection.

        Returns '' on the first iteration (empty list). On subsequent
        iterations returns '\n'.join(self._act_trail).
        """
        return '\n'.join(self._act_trail)

    def get_previous_messages(self, token_budget: int | None = None) -> str:
        """Assemble the ## Previous Messages block for this channel.

        Format (locked by the north star § "Literal format"):

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
           DTO whose its content is already surfaced via the checkpoint prepend
           (step 2) and must never double-render.

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
        # steer, tool_compaction, act_restart, and batched LLM tool
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
            # checkpoint prepend above. (Decision 4B — compaction tool must
            # NEVER make it to Previous Messages.)
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

    # ── CamelCase backward-compat shims ──────────────────────────────────────
    # Test suite calls these names directly. Each shim delegates to the
    # snake_case override so subclass method resolution works correctly.

    def getSystemPrompt(self) -> str:  # noqa: N802
        return self.get_system_prompt()

    def getTools(self) -> list[dict]:  # noqa: N802
        return self.get_tools()

    def getDynamicTools(self) -> list[dict]:  # noqa: N802
        return self.get_dynamic_tools()

    def getActLoopTrail(self) -> str:  # noqa: N802
        return self.get_act_loop_trail()

    def getPreviousMessages(self, token_budget: int | None = None) -> str:  # noqa: N802
        return self.get_previous_messages(token_budget)

    def handleTool(self, tc: dict) -> str:  # noqa: N802
        return self.handle_tool(tc)

    def postTurn(self) -> None:  # noqa: N802
        self.post_turn()

    # ── Tool dispatch ──────────────────────────────────────────────────────────

    def handle_tool(self, tc: dict) -> str:
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
            # 1. Dispatch via the per-turn dispatcher.
            # Spread tc_input first so the action-envelope 'type' (the tool
            # name) cannot be overwritten by a schema property of the same
            # name. The dispatcher reserves 'type' for the action category;
            # schemas must use a non-colliding key (e.g. 'agent_type').
            dispatch = self._dispatcher.dispatch_action(
                self.CHANNEL, {**tc_input, 'type': tool_name}
            )
            result_text = str(dispatch.get('result', ''))

            # Merge any params updates from the ability back into tc_input
            # so they appear in the recorded tool_calls row.
            _params_upd = dispatch.get('_params_update')
            if _params_upd and isinstance(_params_upd, dict):
                tc_input.update(_params_upd)

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
            logger.exception(
                "[MessageProcessor.handle_tool] tool=%s raised: %s",
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
            ).render_and_record()
        except Exception as exc:
            rendered = f"[{tool_name}()] {result_text}"
            logger.error(
                "[MessageProcessor.handle_tool] renderAndRecord failed tool=%s: %s",
                tool_name, exc, exc_info=True,
            )

        # 3. Add to context
        self._act_trail.append(rendered)
        return result_text

    # ── send() helpers (extracted for S3776 cognitive complexity) ────────────

    def _record_iteration_narration(self, llm_response):
        """Record mid-loop narration for the current iteration."""
        if not llm_response.text:
            return
        from services.tool_render_and_record_service import ToolRenderAndRecordService
        rendered = ToolRenderAndRecordService(
            tool_name='narration',
            params={},
            result=llm_response.text,
            ephemeral=True,
            transcript_id=self._uid,
        ).render_and_record()
        self._act_trail.append(rendered)
        try:
            self._emit_narration(llm_response.text, self._current_iteration)
        except Exception as exc:
            logger.error(
                "[MessageProcessor.send] _emit_narration raised: %s",
                exc, exc_info=True,
            )

    def _apply_overflow_guard(self, system_prompt, tools, user_body):
        """Check payload size vs provider context and run compaction if needed.

        Returns:
            'continue' — compaction succeeded, restart iteration from 0.
            None       — proceed with the LLM call (no overflow or already recovered).
        """
        if not self._check_threshold(system_prompt, tools, user_body):
            return None

        if self._overflow_recovered_this_turn:
            logger.warning(
                "[COMPACTION] %s: payload still exceeds compact_at "
                "after one overflow recovery — sending anyway "
                "(system_prompt + tools likely dominate)",
                self.CHANNEL,
            )
            return None

        logger.warning(
            "[COMPACTION] %s: full payload exceeds compact_at — running overflow handler",
            self.CHANNEL,
        )
        if self._handle_overflow():
            self._overflow_recovered_this_turn = True
            return 'continue'

        logger.warning(
            "[COMPACTION] %s: overflow handler returned False — "
            "sending anyway (one-shot recovery exhausted)",
            self.CHANNEL,
        )
        self._overflow_recovered_this_turn = True
        return None

    def _handle_413(self, exc):
        """Handle PayloadTooLargeError from the provider.

        Returns:
            'continue' — compaction succeeded, restart iteration.
            'break'    — unrecoverable, exit the ACT loop.
        """
        if self._overflow_recovered_this_turn:
            logger.error(
                "[COMPACTION] %s: PayloadTooLargeError after overflow "
                "recovery — breaking to cap exit (final_text=''): %s",
                self.CHANNEL, exc,
            )
            return 'break'

        logger.warning(
            "[COMPACTION] %s: PayloadTooLarge from provider — "
            "running overflow handler (%s)",
            self.CHANNEL, exc,
        )
        self._overflow_recovered_this_turn = True
        if self._handle_overflow():
            return 'continue'

        logger.error(
            "[COMPACTION] %s: overflow handler failed after "
            "PayloadTooLarge — breaking to cap exit (final_text='')",
            self.CHANNEL,
        )
        return 'break'

    def send(self, request_id: str | None = None) -> str:
        """Run the full turn: memory seed → ACT loop → store → post_turn.

        Single-path overflow handling (north star § Context Compaction):
        - At the start of every iteration, the rebuilt user-message body
          (wrapped by ``_wrap_with_checkpoint``) is measured against 80%
          of the provider's context window for ``self.LOG_LABEL``.
        - Over threshold → ``_handle_overflow()`` runs a full continuity
          compaction via ``ContinuityCompactionProcessor``, writes an
          append-only ``tool_calls`` audit row, clears ACT state, and
          resets iteration to 0 for a clean loop restart.
        - ``PayloadTooLargeError`` (HTTP 413) from the provider triggers the
          same ``_handle_overflow()`` path, but only once per turn
          (``_overflow_recovered_this_turn`` guard, shared with the proactive
          threshold path so any combination of trips fires at most one
          compaction per turn).
        - The exploration pass runs under its own ``THINKING_TIMEOUT`` envelope
          (independent budget). The ACT loop has no whole-loop wall-clock budget.
        - Overflow handling does NOT reset the per-instance deadline — the
          wall-clock guard (if set) keeps ticking so subagent turns eventually
          hit the cap.
        - If overflow handling fails (LLM error, parse failure) the loop breaks
          to the cap-exit path and returns ``final_text=''``.

        ACT loop termination:
        - Clean exit: LLM returned text with no tool_calls.
        - User-initiated cancel: ``_cancel_event`` set by stop endpoints;
          checked at the top of each iteration before any LLM call.
          UMP and SubagentProcessor run ``while True:`` — this is their
          only hard iteration cap besides ITERATION_TIMEOUT and deadline.
        - MAX_ITERATIONS safety cap — background processors only (DMN=100,
          EAMP=200, PatternMatch=100, GeoPattern=100). UMP and SubagentProcessor
          override ``_iteration_cap_reached()`` to return False.
        - Per-instance deadline (``self._deadline``) — opt-in, set by
          SubagentProcessor based on agent_type. Base class: no deadline.
        - ITERATION_TIMEOUT (1800s) per-iteration safety wall — fires only
          if a single iteration hangs past 30 min (runaway tool call or
          blocking provider). Should never fire in practice.

        Exception semantics:
        - Provider errors (inside the main ACT loop) propagate immediately.
          store() is NOT called; no partial rows land. The atomic-turn
          guarantee from Commit 4 holds.
        - Compaction LLM errors are trapped inside the helpers and logged;
          Stage 1 degrades silently (loop continues with uncompacted trail);
          Stage 2 returns False so the loop breaks to cap exit.
        - handle_tool() errors are already trapped inside handle_tool().
        - _emit_narration() errors are logged and swallowed — never kill
          the ACT loop (north star § ACT Loop step 4).
        - store() errors propagate. If store() raises, post_turn() is NOT
          called and the exception surfaces to the caller.
        - post_turn() errors are caught and logged. The turn is already
          stored; the caller still receives the final response.

        Final-text semantics:
        - Clean exit (LLM returned text with no tool_calls) → final_text is
          the terminating text.
        - Cap exit (MAX_ITERATIONS / deadline / ITERATION_TIMEOUT / Stage 2
          failure) → final_text is ''. The last iteration's mid-loop narration
          is NEVER stored as the assistant row — that text represents partial
          thinking, not a final answer. Storing it would violate north star
          § Storage Model ("final ACT-loop response only").
        """
        from services.act_dispatcher_service import ActDispatcherService
        from services.providers import Providers
        from services.transcript_service import write_input_row

        with bind_current_processor(self):
            # Write input row BEFORE the loop so transcript_id is available
            # for ToolRenderAndRecordService during tool dispatch.
            # Skipped for internal processors (SKIP_TRANSCRIPT_WRITE=True) so
            # they leave no trace in the transcript table.
            # SKIP_INPUT_ROW suppresses only the input row — store() still
            # writes the assistant row (hidden_input isolation).
            if not self.SKIP_TRANSCRIPT_WRITE and not self.SKIP_INPUT_ROW:
                self._uid = write_input_row(self.CHANNEL, self.ROLE, self._raw_input)

            # Single dispatcher for the entire turn. Tools discovered
            # mid-turn via find_tools are registered as handlers on this
            # instance so all tools dispatch through the same path.
            self._dispatcher = ActDispatcherService()
            # Explicit reset makes the per-turn ordinal contract enforceable
            # even if dispatcher lifetime ever changes (Fix 1).
            self._dispatcher.reset_turn_ordinals()

            with self._metrics.stage('pre_act'):
                self.pre_act()
            with self._metrics.stage('thinking_gate'):
                self._run_thinking_gate()   # CHANNEL='user' only, guarded internally

            # Log exploration injection once per turn — at this single point,
            # not inside the ACT loop.
            if self._thinking_exploration:
                logger.info(
                    "[THINKING] Chain of Thought injected into user body (chars=%d)",
                    len(self._thinking_exploration),
                )

            with self._metrics.stage('file_attachments'):
                self._process_file_attachments()

            self._current_iteration = 0
            llm_response = None
            loop_exited_cleanly = False

            while True:
                # Cooperative cancellation — set by stop endpoints between iterations.
                if self._cancel_event.is_set():
                    logger.info(
                        "[MessageProcessor] %s: ACT loop cancelled by user "
                        "(iteration=%d)",
                        self.CHANNEL, self._current_iteration,
                    )
                    break

                # Per-instance deadline check (opt-in — SubagentProcessor only).
                # Base class never sets self._deadline so this is a no-op for UMP.
                if self._deadline is not None and time.time() > self._deadline:
                    logger.warning(
                        "[MessageProcessor.send] %s: per-instance deadline exceeded "
                        "(iteration=%d) — breaking to cap exit",
                        self.CHANNEL, self._current_iteration,
                    )
                    break

                # Iteration cap check (background processors only — UMP and
                # SubagentProcessor override _iteration_cap_reached() to return
                # False, making their loops unbounded except for user stop /
                # deadline / ITERATION_TIMEOUT / clean exit).
                if self._iteration_cap_reached():
                    logger.warning(
                        "[MessageProcessor.send] %s: ACT loop hit safety cap "
                        "(iteration=%d, max_iter=%d) — "
                        "final_text set to '' to avoid persisting mid-loop narration "
                        "as assistant response",
                        self.CHANNEL,
                        self._current_iteration,
                        self.MAX_ITERATIONS,
                    )
                    break

                iter_start = time.time()
                with self._metrics.iteration(self._current_iteration):
                    with self._metrics.stage('prompt_assembly'):
                        user_body = self.get_user_prompt()
                        user_body = _wrap_with_exploration(self.CHANNEL, user_body)
                        user_body = _wrap_with_checkpoint(self.CHANNEL, user_body)
                        system_prompt = self.get_system_prompt()
                        tools = self.get_tools()

                    # Single-path overflow handling: triggered when the full
                    # assembled payload (system_prompt + tools schema + user_body)
                    # exceeds 80% of the provider's context window.
                    #
                    # _handle_overflow() runs a full continuity compaction,
                    # resets ACT state, and returns True on success so the loop
                    # restarts at iter=0. On failure, returns False and the loop
                    # breaks to cap exit (final_text=''). The per-instance
                    # deadline (self._deadline) keeps ticking — compaction LLM
                    # calls count against the subagent's wall-clock budget.
                    overflow_action = self._apply_overflow_guard(
                        system_prompt, tools, user_body,
                    )
                    if overflow_action == 'continue':
                        continue

                    # Single-element messages[] so the provider sees one user turn
                    # containing the full literal-text body (Previous Messages,
                    # memory seed, raw input, ACT trail). The provider's multi-turn
                    # interface is intentionally NOT used — history lives in the
                    # get_user_prompt() text block, not in the messages array.
                    messages = [{'role': 'user', 'content': user_body}]

                    # Provider errors propagate here. store() is not called if
                    # this raises — the turn leaves no trace in the DB.
                    # PayloadTooLargeError (HTTP 413): the provider's transport-
                    # level body cap was hit (e.g. Ollama Cloud edge proxy). We
                    # run _handle_overflow() once to compact and retry. A second
                    # 413 after recovery means even the compacted body is too big
                    # — break to cap exit rather than loop forever.
                    try:
                        llm_response = Providers.instance().send_messages(
                            system_prompt, messages, job=self.LOG_LABEL, tools=tools,
                            thinking_mode=self._get_thinking_mode_for_send(),
                        )
                    except PayloadTooLargeError as exc:
                        action_413 = self._handle_413(exc)
                        if action_413 == 'continue':
                            continue
                        break  # 'break' — unrecoverable
                    self._metrics.accumulate(llm_response)

                    trail_before = len(self._act_trail)

                    with self._metrics.stage('post_tool_records'):
                        for tc in list(llm_response.tool_calls or []):
                            self.handle_tool(tc)

                    if len(self._act_trail) == trail_before:
                        loop_exited_cleanly = True
                        break

                    self._record_iteration_narration(llm_response)

                    self._current_iteration += 1

                # Per-iteration safety wall: a single iteration should NEVER
                # take longer than ITERATION_TIMEOUT seconds. In practice this
                # only fires if a tool blocks the thread for > 30 min (runaway
                # subprocess, hung network call with no timeout).
                iter_elapsed = time.time() - iter_start
                if iter_elapsed > self.ITERATION_TIMEOUT:
                    logger.error(
                        "[MessageProcessor] %s: iteration %d exceeded %ds "
                        "(elapsed=%.1fs) — interrupting ACT loop",
                        self.CHANNEL,
                        self._current_iteration,
                        self.ITERATION_TIMEOUT,
                        iter_elapsed,
                    )
                    break

            if loop_exited_cleanly:
                final_text = (llm_response.text or '') if llm_response else ''
            else:
                # Non-clean exit (cap / deadline / cancel / ITERATION_TIMEOUT).
                # The last iteration's narration is already captured as a narration
                # DTO; we must NOT re-use it as the assistant row.
                final_text = ''

            with self._metrics.stage('store'):
                self.store(final_text)
            try:
                with self._metrics.stage('post_turn'):
                    self.post_turn()
            except Exception as e:
                logger.error(
                    "[POSTTURN] Failed (turn already stored): %s", e, exc_info=True
                )
            return final_text

    def _iteration_cap_reached(self) -> bool:
        """Return True when the ACT loop should break due to the iteration cap.

        Background processors (DMN, EAMP, PatternMatch, GeoPattern) use the
        default MAX_ITERATIONS cap. UMP and SubagentProcessor override this to
        return False, making their loops unbounded (terminated only by user stop,
        deadline, ITERATION_TIMEOUT, or clean exit).
        """
        return self._current_iteration >= self.MAX_ITERATIONS

    def _emit_narration(self, text: str, iteration: int) -> None:
        """Base no-op. UserMessageProcessor overrides in Commit 8 to push
        mid-loop text to the websocket via the on_narration callback."""
        pass

    def _emit_tool_event(self, event: dict) -> None:
        """Subclasses MUST override BOTH this method AND the
        ``_on_tool_event`` attribute. The base no-op never reads any
        attribute."""
        pass


    # ── Single-path overflow + continuity compaction ─────────────────────────

    def _check_threshold(
        self, system_prompt: str, tools: list, user_body: str
    ) -> bool:
        """Return True if the actual provider payload exceeds the persisted
        compact_at threshold for this processor's provider.

        Delegates to ``Providers.estimate_payload_tokens`` which calls the
        resolved provider's ``build_request_body`` — the identical serialisation
        path used by ``send_messages`` — so the measured byte count matches what
        will be sent over the wire rather than a synthesised approximation.
        compact_at is 80% of the provider's max_tokens (the full context window).
        Strict greater-than: a payload exactly at compact_at is NOT eligible.
        """
        from services.providers import Providers
        compact_at = Providers.instance().get_compact_at()
        if compact_at is None:
            # Boot backfill should always populate compact_at. Missing value
            # indicates a misconfigured provider row — log loudly and skip
            # compaction this turn. Do NOT compute a fallback on the fly.
            logger.error(
                "[COMPACTION] %s: compact_at missing — skipping overflow check",
                self.CHANNEL,
            )
            return False
        messages = [{'role': 'user', 'content': user_body}]
        estimated = Providers.instance().estimate_payload_tokens(
            system_prompt, messages, tools, job=self.LOG_LABEL
        )
        return estimated > compact_at

    def _handle_overflow(self) -> bool:
        """Single overflow-handling path: full compaction + ACT loop state reset.

        Called when either:
        - The rendered user_body exceeds 80% of the context window before an
          LLM call (threshold overflow).
        - The provider returns PayloadTooLargeError (HTTP 413) and the turn
          has not yet been through overflow recovery.

        Contract:
        - Runs ``_run_full_compaction(exclude_id=self._uid)`` to summarise
          the channel history up to (but not including) the current turn.
        - On success:
            - Resets ``_current_iteration`` to 0.
            - Clears ``_act_trail``, ``_discovered_tools``.
            - Clears ``_thinking_exploration`` so a large exploration block
              cannot re-inflate ``user_body`` on the next iteration.
            - Deletes ephemeral=1 tool_calls rows for this turn from the DB
              so a clean prompt is assembled on restart.
            - Returns True (caller re-enters the loop at iter 0).
        - On failure (LLM error, empty output, parse failure):
            - Returns False (caller breaks to cap exit, returns final_text='').
        """
        summary = self._run_full_compaction(exclude_id=self._uid)
        if summary is None:
            logger.warning(
                "[COMPACTION] %s: _handle_overflow: _run_full_compaction returned None "
                "— breaking to cap exit",
                self.CHANNEL,
            )
            return False

        # Clear all per-turn ACT state so the restarted loop starts clean.
        self._act_trail = []
        self._discovered_tools = []
        self._thinking_exploration = None
        self._current_iteration = 0

        # Purge ephemeral tool_calls for this turn. Keeps the durable audit
        # rows (tool_name='thinking', tool_name='compaction', ephemeral=0)
        # while dropping all intermediate work (ephemeral=1) so the prompt
        # assembled on restart is not contaminated by prior-iteration state.
        if self._uid is not None:
            try:
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()
                with db.connection() as conn:
                    conn.execute(
                        "DELETE FROM tool_calls WHERE transcript_id = ? AND ephemeral = 1",
                        (self._uid,),
                    )
            except Exception as exc:
                logger.warning(
                    "[COMPACTION] %s: failed to purge ephemeral tool_calls for uid=%s: %s",
                    self.CHANNEL, self._uid, exc,
                )

        return True

    def _run_full_compaction(self, exclude_id: 'int | None' = None) -> 'str | None':
        """Run a full continuity compaction for this channel.

        Orchestrator: reads prior compaction + entries since watermark,
        formats the LLM input (continuity-first envelope), dispatches to
        ``ContinuityCompactionProcessor``, parses ``<summary>`` from the
        output, then writes an append-only ``tool_calls`` audit row.

        Args:
            exclude_id: When set, filters this transcript ID from the rendered
                entries list. Used by ``_handle_overflow`` to exclude the
                current turn's input row from the compaction so the LLM does
                not see a partial / unanswered user message.

        Returns:
            The extracted ``<summary>`` body on success, None on failure.
            On failure, writes a ``status=failure`` audit row so the failure
            is traceable but invisible to the canonical lookup (which filters
            for ``status=success``).
        """
        from services import compaction_persistence

        prior = compaction_persistence.get_compaction(self.CHANNEL)
        watermark = prior['compacted_up_to_id'] if prior else 0
        prev_text = (prior.get('compacted_text') or '').strip() if prior else ''

        all_entries = list(compaction_persistence.get_entries_since(self.CHANNEL, watermark))

        # Filter out the current turn's input row so the LLM does not see an
        # incomplete user message (question not yet answered).
        if exclude_id is not None:
            entries = [e for e in all_entries if e.get('id') != exclude_id]
        else:
            entries = all_entries

        # Nothing to compact — bail before hitting the LLM.
        if not entries and not prev_text:
            logger.warning(
                "[COMPACTION] %s: _run_full_compaction called with no entries "
                "and no prior checkpoint — skipping LLM call",
                self.CHANNEL,
            )
            return None

        rendered = [_format_compaction_entry(e) for e in entries]
        compaction_input = _build_compaction_input(prev_text, rendered)
        in_chars = len(compaction_input)

        raw_output, _ = self._dispatch_compaction_llm(compaction_input, watermark)
        if raw_output is None:
            return None

        if not raw_output:
            reason = "LLM returned empty output"
            logger.warning(_COMPACTION_FAILURE_FMT, self.CHANNEL, reason)
            self._write_compaction_audit_row(
                watermark=watermark, status='failure', summary='', reason=reason
            )
            return None

        summary = _extract_compaction_summary(raw_output)
        if not summary:
            reason = "no <summary> tags in LLM output"
            logger.warning(_COMPACTION_FAILURE_FMT, self.CHANNEL, reason)
            self._write_compaction_audit_row(
                watermark=watermark, status='failure', summary='', reason=reason
            )
            return None

        # Watermark advances to the highest ID fed to the LLM.
        new_watermark = max((e.get('id', 0) for e in entries), default=watermark)

        out_chars = len(summary)
        self._write_compaction_audit_row(
            watermark=new_watermark, status='success', summary=summary
        )

        logger.info(
            "[COMPACTION] %s: continuity success — in=%d chars, out=%d chars, "
            "watermark %d→%d",
            self.CHANNEL, in_chars, out_chars, watermark, new_watermark,
        )
        return summary

    def _dispatch_compaction_llm(
        self, compaction_input: str, watermark: int
    ) -> 'tuple[str | None, object]':
        """Invoke ContinuityCompactionProcessor and return (raw_output, proc).

        Dispatches a single LLM call for continuity compaction. Handles
        PayloadTooLargeError and generic exceptions by writing a failure audit
        row and returning (None, proc) so the caller can bail cleanly.

        Never raises — callers rely on the (None, proc) sentinel to skip
        downstream processing.
        """
        from services.compaction_message_processor import ContinuityCompactionProcessor
        proc = ContinuityCompactionProcessor(raw_input=compaction_input)
        try:
            raw_output = (proc.send() or '').strip()
            return raw_output, proc
        except PayloadTooLargeError as exc:
            reason = f"compaction LLM hit HTTP 413: {exc}"
            logger.error(_COMPACTION_FAILURE_FMT, self.CHANNEL, reason)
            self._write_compaction_audit_row(
                watermark=watermark, status='failure', summary='', reason=reason
            )
            return None, proc
        except Exception as exc:
            reason = f"LLM error: {exc}"
            logger.error(_COMPACTION_FAILURE_FMT, self.CHANNEL, reason)
            self._write_compaction_audit_row(
                watermark=watermark, status='failure', summary='', reason=reason
            )
            return None, proc
        finally:
            self._metrics.merge(proc._metrics)

    def _write_compaction_audit_row(
        self,
        *,
        watermark: int,
        status: str,
        summary: str,
        reason: str = '',
    ) -> None:
        """Write a tool_calls audit row for a compaction attempt.

        Success rows (status='success') are picked up by the canonical
        ``get_compaction()`` lookup.  Failure rows (status='failure') are
        stored for traceability but invisible to the lookup (filtered by
        ``json_extract(params, '$.status') = 'success'``).

        Never raises — persistence failure is logged and swallowed so it
        never kills the calling compaction path.
        """
        import json as _json
        if self._uid is None:
            return
        try:
            from services.database_service import get_shared_db_service
            params: dict = {'compacted_up_to_id': watermark, 'status': status}
            if reason:
                params['reason'] = reason
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_calls
                        (transcript_id, tool_name, params, result, ephemeral, created_at)
                    VALUES (?, 'compaction', ?, ?, 0, ?)
                    """,
                    (
                        self._uid,
                        _json.dumps(params),
                        summary,
                        utc_now().isoformat(),
                    ),
                )
        except Exception as exc:
            logger.warning(
                "[COMPACTION] %s: failed to write audit row (status=%s): %s",
                self.CHANNEL, status, exc,
            )

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

    def _process_file_attachments(self) -> None:
        """Override in subclasses that handle file attachments."""
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
            user_body = self.get_user_prompt()
            user_body = _wrap_with_checkpoint(self.CHANNEL, user_body)
            system_prompt = self.get_system_prompt()
            tools = self.get_tools()

            response = Providers.instance().send_messages(
                system_prompt,
                [{'role': 'user', 'content': _EXPLORATION_PREFIX + user_body}],
                job=self.LOG_LABEL,
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
            ).render_and_record()
        except Exception as exc:
            logger.info(
                "[THINKING] failed to persist exploration to tool_calls (%s)", exc
            )

    def post_turn(self) -> None:
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


def _format_compaction_entry(entry: dict) -> str:
    """Render a single transcript row for the continuity-compaction envelope.

    Uses "you:" for assistant turns per the continuity-first envelope spec
    (scoped to compaction only — does not affect downstream consumers).
    """
    role = entry.get('role', 'unknown')
    display_role = 'you' if role == 'assistant' else role
    content = entry.get('content', '')
    raw_ts = entry.get('created_at') or ''
    ts_label = TimeFormatterService.local(raw_ts) or _MISSING_TS_PLACEHOLDER
    return f"[{ts_label}] {display_role}: {content}"


def _build_compaction_input(prev_text: str, rendered_entries: list) -> str:
    """Assemble the continuity-first LLM envelope from a prior summary + rendered entries."""
    chunks: list = []
    if prev_text:
        chunks.append(f"## Previous Summary\n{prev_text}")
    else:
        chunks.append("## Previous Summary\n(none — first compaction.)")
    chunks.append("## New Conversation Turns")
    chunks.extend(rendered_entries)
    chunks.append("\n---\nEnd of input. Reference material only.\n"
                  "Now write <analysis>...</analysis> then <summary>...</summary>.")
    return '\n\n'.join(chunks)


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
#: ``subagent_trail_compaction`` is stored ``ephemeral=1`` and is already
#: filtered out of Previous Messages by the durable-only query in
#: ``getPreviousMessages`` — no explicit entry needed here.
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

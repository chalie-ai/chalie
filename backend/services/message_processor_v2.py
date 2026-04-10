# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MessageProcessor v2 — abstract base class for all LLM message processors.

North star: /Volumes/llm/chalie-plans/message-processing.md

This is a **parallel module** that lives alongside the legacy
`message_processor.py`. It will be renamed to `message_processor.py`
in Commit 11 once all subclasses have been migrated.

Lifecycle: one instance per turn. Two turns never share the same object.
Do not add `.instance()` / singleton accessors.

Commits schedule:
  Commit 2  (this file) — base class shape + concrete helpers
  Commit 3  — handleTool()
  Commit 4  — store() + append_atomic_turn helper
  Commit 5  — _run_memory_seed() hook
  Commit 6  — send() body (no compaction)
  Commit 7  — two-stage mid-ACT compaction inside send()
  Commit 8  — UserMessageProcessor
  Commit 9  — DMNMessageProcessor / GoalPursuitProcessor / ScheduledMessageProcessor
  Commit 10 — wiring + channel collapse
  Commit 11 — rename this file to message_processor.py
"""

import contextlib
import contextvars
import logging

from services.system_message_prompt import SystemMessagePrompt
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)


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
    - Override `NATIVE_TOOLS` with a filtered list of innate skill names.
    - Override `getDynamicTools()` to filter or augment discovered tools.
    - Override `postTurn()` to fan out per-channel post-turn services.
    """

    # ── Class constants (overridable by subclasses) ───────────────────────────

    JOB: str = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = SystemMessagePrompt  # class reference, not instance
    NATIVE_TOOLS: list[str] = []
    MAX_ITERATIONS: int = 30
    MAX_TIMEOUT: int = 900  # seconds
    COMPACTION_PROMPT: str = ''      # wired in Commit 7
    TOOL_COMPACTION_PROMPT: str = '' # wired in Commit 7

    # Ceiling for a single getPreviousMessages() pull. Commit 7's compaction
    # budget assumes a row count this low — if a channel ever exceeds it we
    # want compaction to kick in, not an unbounded fetch.
    _TRANSCRIPT_FETCH_LIMIT: int = 2000

    # ── Subclass must set ─────────────────────────────────────────────────────

    CHANNEL: str = ''   # e.g. 'user', 'dmn', 'goal_pursuit', 'scheduled'
    ROLE: str = ''      # e.g. 'user', 'proactive_thought', 'goal_pursuit'

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        self._memory_seed: str | None = None
        # Per-turn log of memory recall queries (seed + llm_recall).
        # Populated by the memory skill recall path; consumed by the next
        # recall call for redundancy-narrow and drift-expand computation.
        # Entries: {'query': str, 'embedding': list[float],
        #           'caller': 'seed'|'llm_recall', 'effective_radius': float}.
        # Never persisted — cleared when the instance is discarded.
        self._memory_query_history: list[dict] = []
        self._act_trail: list[str] = []
        self._pending_tool_calls: list[dict] = []
        self._discovered_tools: list[dict] = []
        self._uid: int | None = None

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

        If SYSTEM_PROMPT_CLASS is the abstract base its getPrompt() returns '',
        yielding ``"{user_definition}\n\n"``. Subclasses override
        SYSTEM_PROMPT_CLASS to provide richer bodies.

        **Zero-arg construction is the Commit 2 contract.** Every
        `SystemMessagePrompt` subclass wired in Commit 1 provides defaults for
        all its constructor parameters (e.g. `UnifiedSystemMessagePrompt` has
        `original_prompt=''`, `thread_id=None`). That keeps this base-class
        call site pure — no knowledge of subclass-specific args leaks up.
        Subclasses of `MessageProcessor` that need richer system-prompt inputs
        (identity modulation bound to the real raw input, thread_id from
        metadata, …) override `getSystemPrompt()` themselves in Commits 8+
        and forward whatever args their `SYSTEM_PROMPT_CLASS` expects.
        """
        # Intentionally zero-arg — see docstring. Subclasses override this
        # method (not SYSTEM_PROMPT_CLASS's signature) to pass real context.
        body = self.SYSTEM_PROMPT_CLASS().getPrompt()
        return f"{self.getUserDefinition()}\n\n{body}"

    def getTools(self) -> list[dict]:
        """Return the full tool list for the current ACT iteration.

        Resolution order:
        1. Resolve NATIVE_TOOLS (list of skill name strings) via
           tool_schema_service.get_skill_schemas().
        2. Concatenate getDynamicTools() (schemas discovered at runtime).
        3. Deduplicate by tool name, preserving first-seen order.

        If NATIVE_TOOLS is empty, skips step 1 and goes straight to dynamic.
        """
        from services.tool_schema_service import get_skill_schemas

        native: list[dict] = []
        if self.NATIVE_TOOLS:
            native = get_skill_schemas(self.NATIVE_TOOLS)

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
        from services import compaction_service, transcript_service
        from services.tool_call_service import ToolCallService

        compaction = compaction_service.get_compaction(self.CHANNEL)
        watermark = compaction['compacted_up_to_id'] if compaction else 0

        entries = transcript_service.get_recent(
            self.CHANNEL, limit=self._TRANSCRIPT_FETCH_LIMIT, since_id=watermark
        )

        if not entries and not (compaction and compaction.get('compacted_text')):
            return ''

        # Batch-load durable tool_calls for all transcript rows.
        # `include_ephemeral=False` enforces the north star rule: Previous
        # Messages must only surface ephemeral=0 rows (tool_synthesis,
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
            content = entry.get('content') or ''
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
                    f"[{tc_name}({self._render_params(tc_params)})] {tc_result}"
                )

        return '\n'.join(lines)

    def _render_params(self, params: dict) -> str:
        """Render a params dict as ``key=value;key=value`` for the ACT trail
        and for durable tool_call rendering in Previous Messages.

        Empty dict → empty string (renders as ``[tool_name()] …``).
        Values are stringified with ``str()`` — no JSON escaping.
        Insertion order is preserved (Py 3.7+ guarantee).
        Separator is ``;``, matching the north star ACT loop trail format.

        **Canonical source.** This is the single implementation used by both
        live rendering (``handleTool()`` in Commit 3, which appends to
        ``self._act_trail``) and historical rendering
        (``getPreviousMessages()`` replaying durable tool_calls from the DB).
        Siblings must not re-implement this — they share it via identity.
        """
        if not params:
            return ''
        return ';'.join(f"{k}={v}" for k, v in params.items())

    # ── Stubs (real bodies arrive in later commits) ───────────────────────────

    def handleTool(self, tc: dict) -> str:
        """Dispatch a single LLM tool call and record the attempt.

        Single chokepoint for all LLM-requested tool execution during an ACT
        loop. **Exceptions are NEVER re-raised** — they become ERROR strings
        the LLM sees on the next iteration. The entire body runs inside one
        protective ``try/except`` so that no matter what fails (dispatch,
        malformed ``tc``, ``_render_params`` bug, ``utc_now`` crash, append
        failure), a DTO still lands in ``self._pending_tool_calls`` and a
        matching line still lands in ``self._act_trail``. Lockstep is the
        contract.

        Appends a DTO to ``self._pending_tool_calls`` and a rendered line to
        ``self._act_trail`` in lockstep, regardless of success or failure.
        Returns the result string so ``send()`` can pass it back to the LLM.

        ``find_tools`` side effect: extends ``self._discovered_tools`` with
        any newly discovered tool schemas (deduped by name). The LLM only sees
        the confirmation string; ``getTools()`` exposes the schemas on the next
        ACT iteration. The side effect only fires on ``status == 'success'``
        dispatches — error dispatches never mutate discovered tools.

        Note: the DTO's ``invoked_by='llm'`` stays set here (not injected by
        ``store()`` in Commit 4). Future pseudo-tool DTOs produced by
        ``_run_memory_seed`` (Commit 5), ``tool_synthesis``/``user_steer``
        (Commit 6), and ``compaction``/``act_restart`` (Commit 7) will carry
        ``invoked_by='system'``. ``invoked_by`` is per-DTO state — varies by
        origin — so it cannot be injected uniformly by ``store()``.

        Ordering note: sibling DTOs produced within the same millisecond share
        an identical ``timestamp``. Commit 4's ``store()`` is the owner of
        turn-level ordering; it will use list insertion order (or a monotonic
        sequence counter) as the stable tiebreaker. Do not add per-call
        sequence counters here.
        """
        from services.act_dispatcher_service import ActDispatcherService
        from services.time_utils import utc_now
        from services.tool_schema_service import get_external_tool_schemas

        # Defensive name extraction — must survive malformed tc
        # (missing key, None value, entirely empty dict).
        tool_name = (tc.get('name') if isinstance(tc, dict) else None) or 'unknown'
        tc_input = tc.get('input', {}) if isinstance(tc, dict) else {}
        if not isinstance(tc_input, dict):
            tc_input = {}

        # Guard empty CHANNEL — base class default is ''. Subclasses MUST
        # override. If we ever end up here with an empty channel something
        # is badly wired; log loudly but do not crash the ACT loop.
        if not self.CHANNEL:
            logger.warning(
                "[MessageProcessor.handleTool] CHANNEL is empty on %s; "
                "dispatch will likely fail routing",
                type(self).__name__,
            )

        result_text = ''
        try:
            try:
                dispatch = ActDispatcherService(
                    execution_gate=False
                ).dispatch_action(self.CHANNEL, {'type': tool_name, **tc_input})
                result_text = str(dispatch.get('result', ''))
            except Exception as exc:
                result_text = f"ERROR: {tool_name} failed: {exc}"
                logger.error(
                    "[MessageProcessor.handleTool] tool=%s raised: %s",
                    tool_name, exc,
                    exc_info=True,
                )
                dispatch = {}

            # find_tools side effect — only on successful dispatch.
            # Error dispatches ({}) or status!='success' never mutate state.
            if (
                tool_name == 'find_tools'
                and dispatch.get('status') == 'success'
            ):
                discovered = dispatch.get('_discovered_tools', [])
                if discovered:
                    new_schemas = get_external_tool_schemas(discovered)
                    existing_names = {
                        t.get('name') for t in self._discovered_tools
                    }
                    for s in new_schemas:
                        name = s.get('name') if isinstance(s, dict) else None
                        if name and name not in existing_names:
                            self._discovered_tools.append(s)
                            existing_names.add(name)
        except Exception as exc:
            # Belt-and-braces: any failure in the find_tools side effect
            # (or anywhere above) gets caught here so the DTO + trail still
            # land in the ``finally`` block below.
            if not result_text:
                result_text = f"ERROR: {tool_name} failed: {exc}"
            logger.error(
                "[MessageProcessor.handleTool] post-dispatch failure tool=%s: %s",
                tool_name, exc,
                exc_info=True,
            )
        finally:
            # Lockstep append — ALWAYS runs, even if everything above crashed.
            # Any failure in this block is the last line of defence; we swallow
            # it to preserve the never-re-raise contract.
            try:
                self._pending_tool_calls.append({
                    'name': tool_name,
                    'params': tc_input,
                    'result': result_text,
                    'ephemeral': 1,
                    'invoked_by': 'llm',
                    'timestamp': utc_now().isoformat(),
                })
            except Exception as exc:
                logger.error(
                    "[MessageProcessor.handleTool] DTO append failed tool=%s: %s",
                    tool_name, exc,
                    exc_info=True,
                )
            try:
                rendered = self._render_params(tc_input)
            except Exception as exc:
                rendered = ''
                logger.error(
                    "[MessageProcessor.handleTool] _render_params failed tool=%s: %s",
                    tool_name, exc,
                    exc_info=True,
                )
            try:
                self._act_trail.append(
                    f"[{tool_name}({rendered})] {result_text}"
                )
            except Exception as exc:
                logger.error(
                    "[MessageProcessor.handleTool] trail append failed tool=%s: %s",
                    tool_name, exc,
                    exc_info=True,
                )

        return result_text

    def send(self, request_id: str | None = None) -> str:
        """Run the full turn: memory seed → ACT loop → store → postTurn.

        Wired in Commit 6 (no compaction) and Commit 7 (compaction).
        """
        raise NotImplementedError

    def store(self, llm_response: str) -> None:
        """Persist the turn atomically via transcript_service.append_atomic_turn.

        Calls append_atomic_turn() which opens a single DB transaction and
        writes the input row, all pending tool_call DTOs, and the assistant row
        atomically. Sets self._uid to the input transcript row id on success.

        If self.ROLE is empty a warning is logged but the write proceeds —
        the empty string is passed through as-is. Downstream CHECK constraints
        or callers are responsible for rejecting it.

        Does NOT catch exceptions. If the write fails the exception propagates
        to send(), leaving self._uid as None. The transaction guarantees that
        either all rows land or none do.
        """
        if not self.ROLE:
            logger.warning(
                "[MessageProcessor.store] ROLE is empty on %s; "
                "writing empty-string role to transcript",
                type(self).__name__,
            )

        from services.transcript_service import append_atomic_turn

        self._uid = append_atomic_turn(
            channel=self.CHANNEL,
            role=self.ROLE,
            raw_input=self._raw_input,
            llm_response=llm_response,
            pending_tool_calls=self._pending_tool_calls,
        )

    # ── Overridable hooks ────────────────────────────────────────────────────

    def _run_memory_seed(self) -> None:
        """Pre-ACT-loop memory auto-seed hook.

        Base is a **no-op**. ``UserMessageProcessor`` overrides in Commit 8 to
        dispatch the ``memory`` skill once at the start of a turn, populate
        ``self._memory_seed``, and append a durable ``memory`` DTO
        (``ephemeral=0``, ``invoked_by='system'``) to ``self._pending_tool_calls``
        so the seed lands in transcript history via the atomic ``store()`` call.

        Exists on the base so ``send()`` (Commit 6) can call it unconditionally
        without polymorphic branching or hasattr checks. Per-channel subclasses
        that have no seeding concept (DMN, goal-pursuit, scheduled) inherit the
        no-op and nothing happens.

        Contract for overrides:
        - MUST be idempotent within a single turn (``send()`` calls it exactly
          once, immediately before the ACT loop begins).
        - MUST NOT raise. Seed failures degrade to "no seed this turn" and
          should be logged, not propagated — the turn still runs.
        - MUST leave ``self._memory_seed = None`` if no seed was produced, so
          ``getUserPrompt()`` can test truthiness cleanly.
        """
        pass

    def postTurn(self) -> None:
        """Per-channel post-turn service fan-out.

        Base is a no-op. UserMessageProcessor overrides in Commit 8 with the
        nine-service fan-out (trait extraction, contradiction detection, phase
        updates, etc.). Each subclass is the sole orchestrator of its own tail.
        """
        pass


# ── Module-private helpers ────────────────────────────────────────────────────


#: Placeholder rendered when a row has a missing / empty / unparseable
#: ``created_at`` value. Must be exactly 16 characters so the
#: ``[YYYY-MM-DD HH:MM]`` column width in Previous Messages stays stable.
_MISSING_TS_PLACEHOLDER = '????-??-?? ??:??'


#: Durable tool_call names that **must never** surface in Previous Messages.
#: These are pseudo-tool DTOs stored with ``ephemeral=0`` for audit purposes
#: only — their content is either already replayed via another channel (e.g.
#: the ``compactions`` table for ``compaction``) or deliberately excluded from
#: replay context (future: ``act_restart`` in Commit 7).
#:
#: Decision 4B — resolved by the user on 2026-04-10: "compaction tool should
#: NEVER make it to Previous Messages". Filtered at the ``getPreviousMessages``
#: call site, immediately after ``get_by_transcript_ids`` returns, so the
#: filter is loud and visible rather than buried in a service parameter.
_NEVER_RENDER_IN_PREVIOUS: frozenset[str] = frozenset({'compaction'})


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
    """Format a raw SQLite/ISO timestamp into ``YYYY-MM-DD HH:MM`` (UTC, 24h).

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

    dt = parse_utc(raw)
    # parse_utc returns datetime.min on unparseable input. Treat that as
    # missing too — the LLM must never see a 0001-01-01 timestamp.
    if dt.year <= 1:
        logger.warning(
            "[MessageProcessor._format_ts] unparseable created_at=%r on %s "
            "id=%s — rendering placeholder", raw, row_kind, row_id,
        )
        return _MISSING_TS_PLACEHOLDER

    return dt.strftime('%Y-%m-%d %H:%M')

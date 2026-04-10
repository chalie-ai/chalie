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

import logging

from services.system_message_prompt import SystemMessagePrompt
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)


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

        Algorithm:
        1. Look up the compaction row for self.CHANNEL.
        2. If a compaction exists, prepend compacted_text as the opening block
           and read transcript rows with id > compacted_up_to_id.
           If no compaction exists, read all transcript rows for the channel.
        3. For each transcript row, emit:
               [YYYY-MM-DD HH:MM] <ROLE>: <content>
           then, immediately below, any durable (ephemeral=0) tool_calls
           linked to that row:
               [YYYY-MM-DD HH:MM] TOOL(<tool_name>): <result>
        4. Ephemeral (ephemeral=1) tool_call rows are never emitted here.

        The token_budget parameter is accepted for forward-compatibility with
        Commit 7 (compaction). In Commit 2 it is silently ignored — no
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
            role = (entry.get('role') or 'unknown').upper()
            content = entry.get('content') or ''
            lines.append(f"[{ts}] {role}: {content}")

            # Interleave durable tool_calls under this transcript row
            for tc in durable_by_id.get(entry.get('id'), []):
                tc_ts = _format_ts(
                    tc.get('created_at'),
                    row_kind='tool_call',
                    row_id=tc.get('id'),
                )
                tc_name = tc.get('tool_name') or tc.get('name', 'tool')
                tc_result = tc.get('result') or ''
                lines.append(f"[{tc_ts}] TOOL({tc_name}): {tc_result}")

        return '\n'.join(lines)

    # ── Stubs (real bodies arrive in later commits) ───────────────────────────

    def handleTool(self, tc: dict) -> str:
        """Dispatch a single LLM tool call.

        Wired in Commit 3.
        """
        raise NotImplementedError

    def send(self, request_id: str | None = None) -> str:
        """Run the full turn: memory seed → ACT loop → store → postTurn.

        Wired in Commit 6 (no compaction) and Commit 7 (compaction).
        """
        raise NotImplementedError

    def store(self, llm_response: str) -> None:
        """Persist the turn atomically via transcript_service.append_atomic_turn.

        Wired in Commit 4.
        """
        raise NotImplementedError

    # ── Overridable hook ─────────────────────────────────────────────────────

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

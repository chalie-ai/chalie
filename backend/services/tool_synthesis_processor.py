# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ToolSynthesisProcessor — Background synthesis of recent tool activity into memory.

Replaces ExperienceAssimilationService. Instead of a MemoryStore queue with
novelty gates and episode storage, this processor reads tool_calls directly
from the DB and runs a focused ACT loop that stores knowledge via the memory
skill → DataGraphService. The LLM decides what is worth keeping.

Interval: 60 minutes. Boot delay: 90 seconds.
"""

import logging
import time
from datetime import timedelta

from services.message_processor import MessageProcessor
from services.system_message_prompt import ToolSynthesisSystemMessagePrompt
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

LOOKBACK_MINUTES: int = 70          # just over 1 hour to avoid edge gaps
MAX_TOOL_CALLS: int = 100           # cap DB rows per cycle
RESULT_TRUNCATE_CHARS: int = 2000   # truncate per-entry result to avoid context bloat
INTERVAL_S: int = 3600              # 60-minute cycle
BOOT_DELAY_S: int = 90              # let other services settle first

# Tool names whose results are ephemeral operational data — not worth synthesising.
# These supplement any Python-class-level EXCLUDE_FROM_SYNTHESIS = True flags.
_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    # Memory/plumbing pseudo-tools (internal to the ACT loop)
    'memory',
    'compaction',
    'tool_synthesis',
    'user_steer',
    'tool_compaction',
    'act_restart',
    'thinking',  # exploration-pass output stored for ACT-loop re-injection only

    # Raw ephemeral data — snapshot at time of call, no durable knowledge
    'weather',
    'get_current_time',
    'get_time',
    'clock',
    'time',
    # Raw search dump — URLs/snippets are not durable knowledge
    'search',
    'web_search',
    'brave_search',
    'ddg_search',
})


class ToolSynthesisProcessor(MessageProcessor):
    """Background MessageProcessor that synthesises recent tool activity into memory.

    Constructed fresh per cycle. One instance = one ACT turn. Ephemeral after send().

    NATIVE_TOOLS = ['memory'] — the only tool exposed. No find_tools discovery.
    getDynamicTools() returns [] to explicitly block runtime tool injection.
    postTurn() is minimal — metrics only.
    """

    CHANNEL = 'tool_synthesis'
    ROLE = 'tool_synthesis'
    MAX_ITERATIONS = 10
    MAX_TIMEOUT = 300   # 5 minutes
    SYSTEM_PROMPT_CLASS = ToolSynthesisSystemMessagePrompt
    NATIVE_TOOLS: list[str] = ['memory']

    def getUserDefinition(self) -> str:
        return (
            "The user is 'tool_synthesis' — a background process that reviews "
            "recent tool activity and stores important findings in long-term memory."
        )

    def getUserPrompt(self) -> str:
        """Pass the pre-built digest through with role prefix + ACT trail."""
        trail = self.getActLoopTrail()
        parts = [f"tool_synthesis: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def getDynamicTools(self) -> list[dict]:
        """Explicitly suppress find_tools discovery — memory skill only."""
        return []

    def postTurn(self) -> None:
        """Minimal post-turn: metrics counters only."""
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('tool_synthesis_turns_total')
        except Exception as exc:
            logger.debug("[ToolSynthesis.postTurn] Metrics failed: %s", exc, exc_info=True)


# ── Worker helpers ────────────────────────────────────────────────────────────


def _should_exclude(tool_name: str) -> bool:
    """Return True if this tool's results should be excluded from synthesis."""
    if tool_name in _EXCLUDED_TOOLS:
        return True
    # Prefix-based exclusions for tool families
    if tool_name.startswith('search_'):
        return True
    return False


def _gather_recent_tool_calls() -> str | None:
    """Query tool_calls for ephemeral=1 rows in the lookback window.

    ephemeral=1 rows are the actual tool results from ACT loops.
    ephemeral=0 rows are memory seed / compaction — internal plumbing, skip.

    Returns a formatted digest string, or None if no qualifying rows found.
    """
    from services.database_service import get_shared_db_service

    cutoff = (utc_now() - timedelta(minutes=LOOKBACK_MINUTES)).isoformat()
    db = get_shared_db_service()

    # Build SQL NOT IN from _EXCLUDED_TOOLS — single source of truth for
    # tool-name exclusion.  Prefix-based rules (search_*) are applied in
    # Python below since SQL LIKE per-row would be slower than a post-filter.
    placeholders = ', '.join('?' for _ in _EXCLUDED_TOOLS)
    params: list = list(_EXCLUDED_TOOLS) + [cutoff, MAX_TOOL_CALLS]

    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT tc.tool_name, tc.params, tc.result, tc.created_at
            FROM tool_calls tc
            WHERE tc.tool_name NOT IN ({placeholders})
              AND tc.created_at >= ?
              AND tc.ephemeral = 1
            ORDER BY tc.created_at DESC
            LIMIT ?
            """,
            params,
        )
        rows = cursor.fetchall()

    if not rows:
        return None

    # Filter prefix-based exclusions and build digest
    db_count = len(rows)
    entries = []
    for tool_name, tool_params, result, created_at in rows:
        if _should_exclude(tool_name):
            continue

        ts = (created_at or '')[:16].replace('T', ' ')
        result_text = (result or '').strip()
        if len(result_text) > RESULT_TRUNCATE_CHARS:
            result_text = result_text[:RESULT_TRUNCATE_CHARS] + '...[truncated]'

        if not result_text:
            continue

        # Include truncated params for context (what was asked → what came back)
        params_text = (tool_params or '').strip()
        if len(params_text) > 200:
            params_text = params_text[:200] + '...'

        entry = f"[{ts}] {tool_name}"
        if params_text and params_text != '{}':
            entry += f" ({params_text})"
        entry += f":\n{result_text}"
        entries.append(entry)

    logger.debug("[ToolSynthesis] DB rows=%d, after filter=%d", db_count, len(entries))

    if not entries:
        return None

    header = f"Recent tool activity ({len(entries)} calls from the last {LOOKBACK_MINUTES} minutes):"
    return header + '\n\n' + '\n\n---\n\n'.join(entries)


def _run_cycle() -> None:
    """Gather recent tool calls and run one synthesis ACT turn."""
    digest = _gather_recent_tool_calls()
    if not digest:
        logger.info("[ToolSynthesis] No qualifying tool calls in lookback window — skipping")
        return

    logger.info("[ToolSynthesis] Running synthesis cycle")
    result = ToolSynthesisProcessor(
        raw_input=digest,
        metadata={'source': 'tool_synthesis'},
    ).send()

    result_text = (result or '').strip()
    if not result_text:
        logger.info("[ToolSynthesis] Cycle complete — cap exit (no final text)")
    elif 'SYNTHESIS_NO_ACTION' in result_text:
        logger.info("[ToolSynthesis] Cycle complete — no action taken")
    else:
        logger.info("[ToolSynthesis] Cycle complete — result_len=%d", len(result_text))


def tool_synthesis_worker(shared_state: dict = None):
    """Background worker — registered as a daemon thread in run.py.

    Runs one synthesis cycle every INTERVAL_S seconds. Boot delay of
    BOOT_DELAY_S allows other services to settle first.
    """
    logger.info("[ToolSynthesis] Worker starting")
    time.sleep(BOOT_DELAY_S)

    while True:
        try:
            _run_cycle()
        except KeyboardInterrupt:
            logger.info("[ToolSynthesis] Worker stopped")
            break
        except Exception as exc:
            logger.error("[ToolSynthesis] Cycle failed: %s", exc, exc_info=True)
        time.sleep(INTERVAL_S)

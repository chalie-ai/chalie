# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SystemMessagePrompt — abstract base and concrete subclasses for building the
stable system-message body sent to the LLM on every turn.

The `getUserDefinition()` line is prepended by `MessageProcessor.getSystemPrompt()`;
these classes only build the *body* that follows it.

Lifecycle: per-turn instance — `MessageProcessor.getSystemPrompt()` constructs a
fresh subclass instance, calls `getPrompt()`, and lets it go out of scope.
No singletons.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical location of all prompt files, relative to this file.
_PROMPTS_DIR = Path(__file__).parent.parent / 'prompts'

# ── Unified system prompt ─────────────────────────────────────────────────────
# Inlined as a Python constant (not a markdown file) so the stable prefix is
# cache-friendly. {{adaptive_directives}} sits at the bottom — it changes
# every turn and acts as the cache-busting suffix. Weaved by
# UserMessageProcessor.getSystemPrompt() before the prompt is sent.
_UNIFIED_PROMPT = """\
Respond directly OR invoke tools — whichever serves the request.

## Principles

1. **You reason, tools provide data.** All judgment happens here.
2. **Respond directly when possible.** If the conversation already contains everything needed, respond now.
3. **Auto-store personal facts.** If the user discloses a personal fact (e.g., "my dog's name is Biscuit", "I am allergic to peanuts"), use the `memory` skill to store it immediately. Do not ask for permission.
4. **Proactive recall.** If a user's request could be answered by information they have previously shared, use the `memory` skill to check before responding.
5. **Never fabricate tool results.** If you did not call a tool — or a call returned an error — do not pretend the action succeeded. Only reference information from actual tool results in this conversation.
6. **Use code for math.** If a request requires calculation (e.g., mortgage, interest, percentages), use the `code_eval` tool. Do not perform complex arithmetic inline.
7. **Avoid tool use for simple acknowledgments.** Do not invoke tools for messages like "thanks", "got it", or "ok".
8. **Handle ambiguity with search.** If a user's request is ambiguous (e.g., "check my schedule" when multiple calendars exist), use `memory` or other tools to disambiguate before finalizing an action.

────────────────────────────────

## Output

**Direct response**: When you have sufficient context, respond with text.

**Tool use**: When you need to take action, call the appropriate tool. Each time you call tools, you must also include a cycle summary — a brief text synthesising what the tools returned and what you plan to do next. This is shown to the user in real time.

Good: "Checked your TV and movie services — nothing matches your preferences. Checking the weather for a walk instead."
Bad: "Running weather check."

When all tool calls are complete, your final response must be a comprehensive factual synthesis of everything found. Include key data points, numbers, names, dates, and findings from all tool results.

Example: "Web searches showed Midea founded 1968 by He Xiangjian in Shunde, born 1942, revenue $50B in 2023, ~190K employees."

────────────────────────────────

{{adaptive_directives}}\
"""


class SystemMessagePrompt:
    """Abstract base class for system-message body builders.

    Default implementation returns an empty string, which is the safe no-op
    fallback. Concrete subclasses override `getPrompt()` to return their
    channel-specific body.
    """

    def getPrompt(self) -> str:
        return ''


class UnifiedSystemMessagePrompt(SystemMessagePrompt):
    """System-message body for user-facing (unified) turns.

    **Zero-arg by design (Decision Y1 — 2026-04-10).** No constructor
    parameters, no turn-specific state. Returns ``_UNIFIED_PROMPT`` verbatim,
    leaving ``{{adaptive_directives}}`` as a literal placeholder for
    ``UserMessageProcessor.getSystemPrompt()`` to weave in.

    The prompt is a Python constant (not a file read) so the stable prefix
    maximises provider-side prompt caching. ``{{adaptive_directives}}`` sits
    at the bottom to act as the per-turn cache-busting suffix.
    """

    def getPrompt(self) -> str:
        return _UNIFIED_PROMPT


class _FileBackedSystemMessagePrompt(SystemMessagePrompt):
    """Internal abstract base: reads a prompt file from `backend/prompts/` and
    returns its stripped contents. On read failure, logs a WARNING and delegates
    to ``_fallback_prompt()`` which subclasses override to return the
    legacy-compatible hardcoded fallback string.

    Subclasses set `_FILE_NAME` (filename only, e.g. ``'dmn.md'``) and
    optionally `_LOG_TAG` for the warning label, and override
    ``_fallback_prompt()`` to provide a non-empty fallback body matching the
    legacy `_load_system_prompt()` behaviour they are replacing.
    """

    _FILE_NAME: str = ''
    _LOG_TAG: str = 'SYSTEM PROMPT'

    def getPrompt(self) -> str:
        path = _PROMPTS_DIR / self._FILE_NAME
        try:
            return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            logger.warning(f"[{self._LOG_TAG}] Failed to load system prompt from {path}: {e}")
            return self._fallback_prompt()

    def _fallback_prompt(self) -> str:
        """Hardcoded fallback returned when the on-disk prompt file cannot be
        read. Default is the empty string; subclasses override to preserve the
        exact legacy fallback strings from their pre-refactor
        ``_load_system_prompt()`` functions.
        """
        return ''


class DMNSystemMessagePrompt(_FileBackedSystemMessagePrompt):
    """System-message body for DMN (background proactive) turns.

    Reads `backend/prompts/dmn.md`. On read failure, logs a WARNING and
    returns the legacy hardcoded fallback string — byte-identical to
    `dmn_message_processor._load_system_prompt()`.

    Will be wired to: `DMNMessageProcessor` (Commit 9).
    """

    _FILE_NAME = 'dmn.md'
    _LOG_TAG = 'DMN'

    # Exact string from backend/services/dmn_message_processor.py:_load_system_prompt().
    # Locked by tests — do not alter without updating both the legacy callsite
    # and the regression test.
    _LEGACY_FALLBACK = (
        "You are Chalie, running a background review of recent activity. "
        "Based on the episodes provided, do you detect any patterns, unresolved threads, "
        "or opportunities to be proactive? If so, act on it using the tools available to you. "
        "If nothing actionable stands out, respond with exactly: DMN_NO_ACTION"
    )

    def _fallback_prompt(self) -> str:
        return self._LEGACY_FALLBACK


class GoalPursuitSystemMessagePrompt(_FileBackedSystemMessagePrompt):
    """System-message body for goal-pursuit background turns.

    Reads `backend/prompts/goal-pursuit.md`. On read failure, logs a WARNING
    and returns the legacy hardcoded fallback string — byte-identical to
    `goal_pursuit_processor._load_system_prompt()`.

    Will be wired to: `GoalPursuitProcessor` (Commit 9).
    """

    _FILE_NAME = 'goal-pursuit.md'
    _LOG_TAG = 'GOAL PURSUIT'

    # Exact string from backend/services/goal_pursuit_processor.py:_load_system_prompt().
    # Locked by tests — do not alter without updating both the legacy callsite
    # and the regression test.
    _LEGACY_FALLBACK = (
        "You are Chalie, a determined assistant. Pursue the goal provided to the best "
        "of your ability. If tool calls fail or errors occur, try alternatives. "
        "When complete, provide a clear summary of what you accomplished."
    )

    def _fallback_prompt(self) -> str:
        return self._LEGACY_FALLBACK


class ScheduledSystemMessagePrompt(_FileBackedSystemMessagePrompt):
    """System-message body for scheduled-prompt turns.

    Reads `backend/prompts/scheduled-prompt.md`. On read failure, logs a
    WARNING and returns the legacy hardcoded fallback string — byte-identical
    to `scheduled_message_processor._load_system_prompt()`.

    Will be wired to: `ScheduledMessageProcessor` (Commit 9).
    """

    _FILE_NAME = 'scheduled-prompt.md'
    _LOG_TAG = 'SCHEDULED PROMPT'

    # Exact string from backend/services/scheduled_message_processor.py:_load_system_prompt().
    # Locked by tests — do not alter without updating both the legacy callsite
    # and the regression test.
    _LEGACY_FALLBACK = (
        "You are Chalie, executing a scheduled task. The user set this up earlier "
        "and it is now due. Execute the task to the best of your ability using the "
        "tools available to you. Be concise and action-oriented in your response."
    )

    def _fallback_prompt(self) -> str:
        return self._LEGACY_FALLBACK

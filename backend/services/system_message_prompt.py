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


class SystemMessagePrompt:
    """Abstract base class for system-message body builders.

    Default implementation returns an empty string, which is the safe no-op
    fallback. Concrete subclasses override `getPrompt()` to return their
    channel-specific body.
    """

    def getPrompt(self) -> str:
        return ''


class UnifiedSystemMessagePrompt(SystemMessagePrompt):
    """Pure static template renderer for the user-facing (unified) system
    prompt body.

    **Zero-arg by design (Decision Y1 — 2026-04-10).** This class has no
    constructor parameters and does not read any turn-specific state. It
    returns the UNIFIED prompt template verbatim, leaving the
    ``{{identity_modulation}}`` and ``{{adaptive_directives}}`` placeholders
    **literal** for a caller further up the stack to weave in.

    **Why the retrofit from Commit 1's parameterised version:** identity
    modulation and adaptive directives are *per-turn runtime context*, not
    static template content. They belong one level up, inside the owning
    ``MessageProcessor`` subclass's ``getSystemPrompt()`` override, which
    already holds the turn state (``self._raw_input``, ``self._metadata``).
    Keeping them *inside* ``SystemMessagePrompt`` forced parameters on the
    class, which in turn forced ``MessageProcessor.getSystemPrompt()`` to
    either know about subclass-specific args (leaky) or silently degrade
    identity/adaptive when callers forgot to override. The zero-arg contract
    makes that failure mode impossible — the weaving cannot be "forgotten"
    because the class itself cannot hold turn state.

    The weaving moves to ``UserMessageProcessor.getSystemPrompt()`` in
    Commit 8. Until then, the placeholders ride through as literal markers;
    no production code path instantiates this class yet.

    Will be wired to: ``UserMessageProcessor`` (Commit 8).
    """

    def getPrompt(self) -> str:
        from workers.digest_singletons import load_configs
        configs = load_configs()
        return configs['cortex']['prompt_map']['UNIFIED']


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

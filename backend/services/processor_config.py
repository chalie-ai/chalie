# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ProcessorConfig — frozen dataclass for per-channel MessageProcessor behaviour.

Spec: ACT Loop Orchestrator Refactor §2.

Every channel's behavioural surface is expressed through a single
ProcessorConfig instance.  The dataclass is frozen so that a config created
for one turn can never be mutated mid-loop (AC-5 / §2).

Usage
-----
Constant channels (DMN, EpisodeEncoder, …) expose a module-level instance::

    from configs.channels import DMN_CONFIG
    mp = MessageProcessor.process(raw_input, DMN_CONFIG)

Per-instance channels (UMP, ExternalAgent, …) expose factory functions::

    from configs.channels import make_user_config
    config = make_user_config(metadata=request_metadata)
    mp = MessageProcessor.process(raw_input, config)

The ``job`` property — ``f"{channel}:{role}"`` — is the telemetry label passed
to ``Providers.calculate`` / ``Providers.send_messages``.  There is no separate
``LOG_LABEL`` field; ``config.job`` IS the label (§2, AC-13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProcessorConfig:
    """Everything that varies between channels.  Immutable per-turn.

    See spec §2 for field-by-field rationale.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    channel: str
    """Transcript / telemetry channel.  E.g. 'user', 'dmn',
    'external-agent:mybot', 'delegate:web_search'."""

    role: str
    """Transcript role for the input row.  E.g. 'user', 'proactive_thought',
    'external_agent', 'pattern_match'."""

    usage_class: str
    """LLM usage class written to llm_call_log.  One of 'chat', 'subagent',
    'subconscious', 'external_agent'."""

    # ── Prompt builders ───────────────────────────────────────────────────────
    # Each callable receives the MessageProcessor instance and returns a str.
    # Using Any at runtime to avoid the circular-import problem; the TYPE_CHECKING
    # guard above provides static-analysis fidelity.

    build_user_prompt: Callable[..., str]
    """Given (mp: MessageProcessor) → str.  The raw user-turn text."""

    build_user_definition: Callable[..., str]
    """Given (mp: MessageProcessor) → str.  The user/persona preamble."""

    build_system_prompt: Callable[..., str]
    """Given (mp: MessageProcessor) → str.  The system instruction block."""

    # ── Tool visibility ───────────────────────────────────────────────────────

    always_available: list[str]
    """Tool names pinned in every LLM call (innate tier)."""

    discoverable: list[str]
    """Tool names discoverable via find_tools for this channel."""

    blocked: frozenset[str]
    """Tool names never offered to the model (e.g. subagent blocks 'subagent')."""

    # ── Loop control ──────────────────────────────────────────────────────────

    max_iterations: int | None
    """ACT loop iteration cap.  None = unbounded (UMP, ExternalAgent)."""

    skip_transcript: bool
    """True → no transcript row written at all (background processors)."""

    skip_input_row: bool
    """True → input row skipped but assistant row still written (hidden_input)."""

    suppress_history: bool
    """True → get_previous_messages() returns '' (housekeeping loops).
    Set on all channels except UMP and ExternalAgent (§2 / AC-26)."""

    # ── Live output (declarative, not a hook) ─────────────────────────────────

    broadcast_to: str | None
    """None = silent.  Non-None = stream narration + tool events to this channel.
    Only make_user_config sets this ('user'); all others leave it None (AC-28)."""

    # ── Turn-0 auto-seed (declarative, not a hook) ────────────────────────────

    memory_seed: bool
    """True → fire the memory recall tool (action='recall') once on turn 0.
    Attachments are NOT a flag: presence of metadata['attachments'] auto-fires
    document.upload per file on turn 0 (AC-30 / AC-31)."""

    # ── The only optional hook — None = no-op ────────────────────────────────

    post_turn: Callable[..., None] | None
    """Called once after the assistant row is persisted, signature:
    (mp: MessageProcessor, response_text: str) -> None.
    None = no-op.  This is the ONLY optional hook on ProcessorConfig (AC-32)."""

    # ── Derived property ──────────────────────────────────────────────────────

    @property
    def job(self) -> str:
        """Telemetry label for Provider calls: ``channel:role``.

        Passed as the ``job`` argument to ``Providers.calculate()`` and
        ``Providers.send_messages()``.  Replaces the per-subclass
        ``LOG_LABEL`` class attribute (§2 / AC-13).
        """
        return f"{self.channel}:{self.role}"

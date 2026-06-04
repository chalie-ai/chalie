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
Every channel is a named ``ProcessorConfig`` subclass with a typed ``__init__``.
The caller instantiates the subclass directly::

    from configs.channels import DmnConfig
    mp = MessageProcessor.process(raw_input, DmnConfig())

    from configs.channels import UserConfig
    mp = MessageProcessor.process(raw_input, UserConfig(metadata=request_metadata))

The ``job`` property — ``f"{channel}:{role}"`` — is the telemetry label passed
to ``Providers.calculate`` / ``Providers.send_messages``.  There is no separate
``LOG_LABEL`` field; ``config.job`` IS the label (§2, AC-13).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor


@dataclass(frozen=True)
class ProcessorConfig(ABC):
    """Everything that varies between channels.  Immutable per-turn.

    See spec §2 for field-by-field rationale.

    The three prompt builders are **abstract methods** — every channel is a
    concrete subclass that implements ``get_system_prompt`` /
    ``get_user_prompt`` / ``get_user_definition``.  The base class is abstract
    (ABC) so a subclass that forgets any of the three cannot be instantiated.
    Each method receives the turn's ``MessageProcessor`` as an explicit ``mp``
    argument; the config holds no reference to the processor and stays a pure,
    fully-immutable frozen dataclass.
    """

    # ── Policy channel (nested enum keeps processor_config.py dependency-free) ──
    class POLICY_CHANNEL(str, Enum):
        CHAT           = "chat"
        SUBCONSCIOUS   = "subconscious"
        EXTERNAL_AGENT = "external_agent"

    # ── Identity ──────────────────────────────────────────────────────────────

    channel: str
    """Transcript / telemetry channel.  E.g. 'user', 'dmn',
    'external-agent:mybot', 'delegate:web_search'."""

    role: str
    """Transcript role for the input row.  E.g. 'user', 'proactive_thought',
    'external_agent', 'pattern_match'."""

    policy_channel: "ProcessorConfig.POLICY_CHANNEL"
    """Which policy channel this processor's tool calls are gated under.
    usage_class (the llm_call_log string) is derived from it."""

    # ── Tool visibility ───────────────────────────────────────────────────────

    always_available: list[str]
    """Tool names pinned in every LLM call (innate tier)."""

    discoverable: list[str]
    """Tool names discoverable via find_tools for this channel."""

    blocked: frozenset[str]
    """Tool names never offered to the model (e.g. DMN blocks the delegate tools)."""

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
    Only UserConfig sets this ('user'); all others leave it None (AC-28)."""

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

    # ── Prompt builders (abstract — one implementation per channel) ───────────

    @abstractmethod
    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        """The system instruction block for this channel's turn.

        Receives the turn's MessageProcessor.  Every concrete config MUST
        implement this; the returned string is byte-identical to the
        pre-refactor builder output."""

    @abstractmethod
    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """The user-turn body (world state, history, input, ACT trail).

        Receives the turn's MessageProcessor.  Every concrete config MUST
        implement this."""

    @abstractmethod
    def get_user_definition(self, mp: "MessageProcessor") -> str:
        """The user/persona preamble.  Receives the turn's MessageProcessor.

        Every concrete config MUST implement this (return ``""`` when the
        channel has no user definition)."""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def job(self) -> str:
        """Telemetry label for Provider calls: ``channel:role``.

        Passed as the ``job`` argument to ``Providers.calculate()`` and
        ``Providers.send_messages()``.  Replaces the per-subclass
        ``LOG_LABEL`` class attribute (§2 / AC-13).
        """
        return f"{self.channel}:{self.role}"

    @property
    def usage_class(self) -> str:
        """LLM usage class string for llm_call_log / telemetry — the value of
        policy_channel ('chat' | 'subconscious' | 'external_agent')."""
        return self.policy_channel.value

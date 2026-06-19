# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ProcessorConfig — frozen dataclass for per-channel MessageProcessor behaviour.

Every channel's behavioural surface is expressed through a single
ProcessorConfig instance.  The dataclass is frozen so that a config created
for one turn can never be mutated mid-loop.

The ``job`` property — ``f"{channel}:{role}"`` — is the telemetry label passed
through ``Providers.send`` to the resolved provider and ``_log_after_call``.
There is no separate ``LOG_LABEL`` field; ``config.job`` IS the label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor
    from services.post_turn_hook import PostTurnHook


@dataclass(frozen=True)
class ProcessorConfig(ABC):
    """Everything that varies between channels.  Immutable per-turn.

    Every concrete subclass must implement ``get_system_prompt``,
    ``get_user_prompt``, and ``get_user_definition``; the ABC constraint ensures
    a subclass that omits any of the three cannot be instantiated.  The config
    holds no reference to the processor — it stays a pure, frozen dataclass.
    """

    # ── Async capability (ClassVar — not a dataclass field) ────────────────────
    SUPPORTS_ASYNC: ClassVar[bool] = False
    """True → the framework ``async`` boolean is exposed on every tool's schema
    for this channel, letting the model run a call in the background and receive
    the result as a later turn.  Only a push channel with a durable session can
    honour a deferred result, so this is False everywhere except UserConfig
    (§4.0 / §4.8d).  It gates schema *exposure* only — never routing."""

    uses_vision_provider: ClassVar[bool] = False
    """True -> Providers._resolve reads the brain's Vision Provider from the DB
    instead of the global selected provider. Only VisionConfig sets it."""

    uses_delegate_provider: ClassVar[bool] = False
    """True -> Providers._resolve reads the brain's Delegate Provider from the DB
    (falling back to the selected provider when none is pinned) instead of the
    global selected provider. Set by the subagent channels (web_search,
    web_browse). Vision takes precedence: VisionConfig keeps uses_vision_provider
    and never sets this (vision > delegate > chat)."""

    # ── Policy channel (nested enum keeps processor_config.py dependency-free) ──
    class PolicyChannel(str, Enum):
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

    policy_channel: "ProcessorConfig.PolicyChannel"
    """Which policy channel this processor's tool calls are gated under.
    usage_class (the llm_call_log string) is derived from it."""

    # ── Tool visibility ───────────────────────────────────────────────────────

    always_available: list[str]
    """Tool names pinned in every LLM call (innate tier).

    This is the ONLY per-config tool list. Discovery scope is not configured
    here: a tool is reachable via find_tools iff its ``Ability.DISCOVERABLE`` is
    True (a global trait) and this config pins ``find_tools`` in
    ``always_available``. There is no per-channel discoverable/blocked list."""

    # ── Loop control ──────────────────────────────────────────────────────────
    #
    # There is no iteration cap. The loop runs until the model stops
    # emitting tool calls or cooperative cancellation fires; a runaway turn is a
    # hard-restart condition, not a silently-capped one.

    skip_transcript: bool
    """True → no transcript row written at all (background processors)."""

    skip_input_row: bool
    """True → input row skipped but assistant row still written (hidden_input)."""

    suppress_history: bool
    """True → get_previous_messages() returns '' (housekeeping loops).
    Set on all channels except UMP and ExternalAgent (§2 / AC-26)."""

    # ── Live output (declarative, not a hook) ─────────────────────────────────

    broadcast_to: str | None
    """None = silent.  Non-None = stream live tool events to this channel and
    deliver the turn's end message there.  Only UserConfig sets this ('user');
    all others leave it None (AC-28)."""

    # ── Turn-0 auto-seed (declarative, not a hook) ────────────────────────────

    memory_seed: bool
    """True → fire the memory recall tool (action='recall') once on turn 0.
    Attachments are NOT a flag: presence of metadata['attachments'] auto-fires
    document.upload per file on turn 0 (AC-30 / AC-31)."""

    # ── After-turn hooks — empty tuple = no-op ───────────────────────────────

    post_turn_hooks: tuple[PostTurnHook, ...] = ()
    """Independent units of after-turn work, each ``hook.run(mp, response_text)``,
    run once after the assistant row is persisted.  Empty tuple = no-op.  Hooks
    are mutually independent and failure-isolated — see services/post_turn_hook.py
    and MessageProcessor._end_turn.  This is the ONLY hook surface on
    ProcessorConfig (AC-32 / §4.8)."""

    # ── Prompt builders (abstract — one implementation per channel) ───────────

    @abstractmethod
    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        """System instruction block for this channel's turn."""

    @abstractmethod
    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """User-turn body (world state, history, input, ACT trail)."""

    @abstractmethod
    def get_user_definition(self, mp: "MessageProcessor") -> str:
        """User/persona preamble. Return ``""`` when the channel has none."""

    # ── Per-turn image attachment (concrete hook — default: no image) ──────────

    def get_image(self, mp: "MessageProcessor") -> "dict | None":
        """VisionConfig overrides this to return
        ``{"data": <base64>, "mime_type": <str>}`` — the shape every converter
        consumes."""
        return None

    # ── Turn-scoped instance state (concrete hook — default: none) ─────────────

    def turn_scoped_state(self) -> tuple[str, ...]:
        """Names of per-instance ``mp`` attributes that belong to the WHOLE turn,
        not a single chain link.

        A turn can span several MessageProcessors: each tool-bearing step spawns a
        hidden-input continuation MP that rebuilds the request from the DB (see
        ``MessageProcessor._continue``). Accumulators a tool mutates across steps
        (counters, dedup/touched sets) and snapshots the prompt reads each step
        live on the MP instance — a fresh continuation would otherwise reset them,
        so a pattern saved in step 1 would be wrongly decayed by the post-turn hook
        that runs on the final step. ``_continue`` carries every name listed here
        forward by reference, so the whole chain shares one turn-scoped value.

        Default ``()`` — channels with no cross-step state need no override."""
        return ()

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def job(self) -> str:
        """Telemetry label passed through ``Providers.send()`` to the resolved
        provider's ``send_messages()`` and ``_log_after_call``.  Replaces the
        per-subclass ``LOG_LABEL`` class attribute."""
        return f"{self.channel}:{self.role}"

    @property
    def usage_class(self) -> str:
        """LLM usage class written to llm_call_log."""
        return self.policy_channel.value

    # ── Cloning ───────────────────────────────────────────────────────────────

    def with_hidden_input(self) -> "ProcessorConfig":
        """Return a shallow copy with ``skip_input_row=True``.

        ``copy.copy`` preserves the concrete subclass and every field without
        re-running the typed ``__init__``; ``object.__setattr__`` is required
        because frozen dataclasses forbid normal field mutation.
        """
        import copy  # noqa: PLC0415
        clone = copy.copy(self)
        object.__setattr__(clone, "skip_input_row", True)
        return clone

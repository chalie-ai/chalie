# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ProcessorConfig — frozen dataclass for per-channel MessageProcessor behaviour."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from configs.enums.config_type import ConfigTypeEnum
    from configs.enums.policy_channel import PolicyChannel


@dataclass(frozen=True)
class ProcessorConfig(ABC):
    """Everything that varies between channels.  Immutable per-turn."""

    # ── Async capability (ClassVar — not a dataclass field) ────────────────────
    SUPPORTS_ASYNC: ClassVar[bool] = False
    """True → the framework ``async`` boolean is exposed on every tool's schema
    for this channel, letting the model run a call in the background and receive
    the result as a later turn.  Only a push channel with a durable session can
    honour a deferred result, so this is False everywhere except UserConfig.
    It gates schema *exposure* only — never routing."""

    BROADCASTS_STATE: ClassVar[bool] = False
    """True → this channel streams its live progress to its surface: the lean
    ``updated`` turn-state signal via ``mp.broadcast``, the turn-execution
    lifecycle frame via TurnExecutionService, and the single tool-call frame via
    ToolCallService. Only UserConfig sets it; every other channel stays silent (each
    chokepoint no-ops). The single state-gate — replaces the scattered
    ``broadcast_to == 'user'`` checks. Distinct from ``broadcast_to`` (message-
    delivery target)."""

    thinking_mode: ClassVar[str | None] = None
    """Fixed reasoning level for a background channel whose deliberation must not
    be data-driven (thread_gist / web_search pin ``"low"``). ``None`` on every
    interactive channel — those resolve the level per-turn (the user channel's
    regression-head gate, or a caller override). ``ProviderService`` reads it
    first in ``resolve_thinking_mode`` (config pin > override > gate result)."""

    USAGE_TYPE: ClassVar[str] = "background"
    """Which spend bucket this channel's provider calls bill to in
    ``llm_call_log.type`` — ``"foreground"`` (the user's own conversation) or
    ``"background"`` (everything Chalie runs on its own behalf: delegates,
    background loops, housekeeping).

    Defaults to ``"background"`` so spend is only ever attributed to the user by an
    explicit declaration — an unclassified channel under-bills the user rather
    than silently inflating their conversation's cost. ``LlmLogService.record``
    reads this class constant directly; there is nothing to thread through a
    constructor or a delegate. Distinct from ``policy_channel``, which gates
    tools: a channel can be policy-gated as CHAT while billing as background (see
    DiscoveryConfig)."""

    uses_vision_provider: ClassVar[bool] = False
    """True -> ProviderService._resolve reads the brain's Vision Provider from the DB
    instead of the global selected provider. Only VisionConfig sets it."""

    uses_delegate_provider: ClassVar[bool] = False
    """True -> ProviderService._resolve reads the brain's Delegate Provider from the DB
    (falling back to the selected provider when none is pinned) instead of the
    global selected provider. Set by the subagent channels (web_search,
    web_browse). Vision takes precedence: VisionConfig keeps uses_vision_provider
    and never sets this (vision > delegate > chat)."""

    # ── Identity ──────────────────────────────────────────────────────────────

    channel: str
    """Transcript / telemetry channel.  E.g. 'user', 'dmn',
    'external-agent:mybot', 'delegate:web_search'."""

    role: str
    """Transcript role for the input row.  E.g. 'user', 'proactive_thought',
    'external_agent', 'pattern_match'."""

    policy_channel: "PolicyChannel"
    """Which policy channel this processor's tool calls are gated under."""

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
    Set on all channels except UMP and ExternalAgent."""

    # ── Live output (declarative, not a hook) ─────────────────────────────────

    broadcast_to: str | None
    """None = silent.  Non-None = stream live tool events to this channel and
    deliver the turn's end message there.  Only UserConfig sets this ('user');
    all others leave it None."""

    # ── Turn-0 auto-seed (declarative, not a hook) ────────────────────────────

    memory_seed: bool
    """True → fire the memory recall tool (action='recall') once on turn 0.
    Attachments are NOT a flag: presence of metadata['attachments'] auto-ingests
    each file on turn 0 (saved via FileParserService, linked in
    ``transcript_files``, dispatched ``vision`` for images, ``read`` otherwise)."""

    # ── Split-channel routing (cross-turn history read channel) ───────────────

    read_channel: str | None = None
    """Cross-turn history read channel; ``None`` ⇒ read on ``channel`` (every
    existing config). Only split-channel configs set this — writes and
    current-turn identity stay on ``channel`` while the model's cross-turn
    history is read from ``read_channel``. Today only DiscoveryConfig sets it
    (``= "user"``), and it runs MAIN-only, so the split's FORK edge cases never
    apply. Do not set ``read_channel != channel`` on a FORK/reply config."""

    # ── PromptService dispatch override ───────────────────────────────────────

    prompt_channel: str | None = None
    """Overrides the PromptService dispatch key; ``None`` ⇒ dispatch on
    ``channel``. Only configs whose ``channel`` is dynamic or channel-specific
    yet share another channel's prompt assembly set this — DiscoveryConfig
    (``= "user"``) reuses the user prompts; EAMPConfig (``= "external_agent"``)
    has a per-agent dynamic ``channel``."""

    # ── External turn-id ownership ────────────────────────────────────────────

    external_turn_id: bool = False
    """When True, the caller assigns ``turn_id`` from an external key space it
    owns (e.g. the schedule id on the ``'schedule'`` channel). A supplied
    turn_id that does not yet exist opens a new MAIN turn rather than being
    rejected as an invalid fork; forked-ness is derived from whether the turn
    already exists, not the -1 sentinel."""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """The channel's static system-prompt body — the literal prompt text.

        Default ``""`` (channels with no ACT loop). Every
        channel whose prompt is a frozen literal overrides this to return it;
        the two dynamic channels (``UserConfig`` / ``EAMPConfig``) return their
        base literal and ``PromptService`` wraps it with runtime data (voice
        line, provider content-field, resolved names). Pure declarative data —
        no service reads, no DB access (§2.4)."""
        return ""

    # ── API routing identity ──────────────────────────────────────────────────

    def type(self) -> "ConfigTypeEnum | None":
        """The top-level API routing identifier for this config, or ``None`` for the
        internal channels that the thread API never addresses. Only the two configs
        reachable via ``config_for`` override this."""
        return None

    def type_value(self) -> "str | None":
        """This config's routing type as the wire/DB string (``None`` for the
        internal channels). The single projection of :meth:`type` every spine
        service emits — the WS ``type`` field and ``turn_executions.type`` — so
        the enum→string conversion lives in exactly one place."""
        config_type = self.type()
        return config_type.value if config_type is not None else None

    # ── Cloning ───────────────────────────────────────────────────────────────

    def with_hidden_input(self) -> "ProcessorConfig":
        """Return a shallow copy with ``skip_input_row=True``."""
        import copy  # noqa: PLC0415
        clone = copy.copy(self)
        object.__setattr__(clone, "skip_input_row", True)
        return clone

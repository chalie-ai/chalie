# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Per-channel ProcessorConfig constants and factory functions.

Spec: ACT Loop Orchestrator Refactor §3.

Static channels (§3a) — constant ProcessorConfig instances:
  DMN_CONFIG, EPISODE_ENCODER_CONFIG, SKILL_SUGGESTION_CONFIG,
  COMPACTION_CONFIG, SUBAGENT_COMPACTION_CONFIG

Per-instance channels (§3b) — factory functions:
  make_user_config(metadata) -> ProcessorConfig
  make_eamp_config(agent_name, project, loop_in_human, wrapper_id) -> ProcessorConfig
  make_pattern_config(window_start, window_end) -> ProcessorConfig
  make_geo_config(window_start, window_end) -> ProcessorConfig
  make_user_summary_config() -> ProcessorConfig
  make_super_episode_config(channel, sources, spans) -> ProcessorConfig

Prompt builder implementations are stubs at T1 (no callers wired yet).
They are replaced with real implementations in T7 (UMP/EAMP) and T8
(background channels).  The structural contract — frozen dataclass, correct
channel/role/limits — is established here.
"""

from __future__ import annotations

from typing import Any

from services.processor_config import ProcessorConfig

# ── Default tool visibility (mirrors MessageProcessor class defaults) ──────────

DEFAULT_ALWAYS_AVAILABLE: list[str] = [
    "find_skills",
    "find_tools",
    "memory",
]

DEFAULT_DISCOVERABLE: list[str] = [
    "bash",
    "browser",
    "calendar",
    "chalie_docs",
    "code_eval",
    "contacts",
    "document",
    "email",
    "file_permissions",
    "file_write",
    "home",
    "list",
    "mcp_manager",
    "news",
    "place",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "review_transcript",
    "schedule",
    "search",
    "search_files",
    "skill_builder",
    "subagent",
    "timer",
    "ubiquiti",
    "weather",
    "web_download",
]


# ── §3a — Static configs (no per-instance args) ───────────────────────────────

DMN_CONFIG = ProcessorConfig(
    channel="dmn",
    role="proactive_thought",
    usage_class="subconscious",
    build_user_prompt=lambda _mp: "",
    build_user_definition=lambda _mp: "",
    build_system_prompt=lambda _mp: "",
    always_available=DEFAULT_ALWAYS_AVAILABLE,
    discoverable=DEFAULT_DISCOVERABLE,
    blocked=frozenset({"subagent"}),
    max_iterations=100,
    skip_transcript=False,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""DMN background channel.  §3a / §8b."""

EPISODE_ENCODER_CONFIG = ProcessorConfig(
    channel="episode_encoder",
    role="episode_encoder",
    usage_class="subconscious",
    build_user_prompt=lambda _mp: "",
    build_user_definition=lambda _mp: "",
    build_system_prompt=lambda _mp: "",
    always_available=[],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=1,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Episode encoder — one-shot, no tools, no transcript writes.  §3a."""

SKILL_SUGGESTION_CONFIG = ProcessorConfig(
    channel="skills_building",
    role="skills_building",
    usage_class="subconscious",
    build_user_prompt=lambda _mp: "",
    build_user_definition=lambda _mp: "",
    build_system_prompt=lambda _mp: "",
    always_available=["skill_manager"],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=5,
    skip_transcript=False,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Skill suggestion — housekeeping, suppress_history=True.  §3a."""

def _compaction_system_prompt(_mp: object) -> str:
    """System prompt for continuity (history) compaction.  §3a / §4a."""
    from services.system_message_prompt import ContinuityCompactionSystemPrompt
    return ContinuityCompactionSystemPrompt().get_prompt()


def _subagent_compaction_system_prompt(_mp: object) -> str:
    """System prompt for subagent trail compaction.  §3a / §4a."""
    from services.system_message_prompt import SubagentTrailCompactionSystemPrompt
    return SubagentTrailCompactionSystemPrompt().get_prompt()


COMPACTION_CONFIG = ProcessorConfig(
    channel="compaction",
    role="compaction",
    usage_class="subconscious",
    build_user_prompt=lambda mp: mp._raw_input,
    build_user_definition=lambda _mp: "",
    build_system_prompt=_compaction_system_prompt,
    always_available=[],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=30,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Continuity compaction — bounded loop, no tools, no transcript writes.  §3a."""

SUBAGENT_COMPACTION_CONFIG = ProcessorConfig(
    channel="subagent_compaction",
    role="subagent_compaction",
    usage_class="subconscious",
    build_user_prompt=lambda mp: mp._raw_input,
    build_user_definition=lambda _mp: "",
    build_system_prompt=_subagent_compaction_system_prompt,
    always_available=[],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=30,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Subagent-trail compaction — bounded loop, no tools, no transcript writes.  §3a."""


# ── §3b — Factory configs (per-instance args) ─────────────────────────────────


def make_user_config(metadata: dict[str, Any] | None = None) -> ProcessorConfig:
    """UMP config — conversational user channel.

    broadcast_to='user' (live output), memory_seed=True, suppress_history=False.
    Attachments auto-fire document.upload on turn 0 (no flag needed — presence
    of metadata['attachments'] drives this).  post_turn = skill suggestion only
    (no metrics, no phase — §3b / §4e / §6).

    Prompt builders are stubbed for T1; real implementations land in T7.
    """
    _metadata = metadata or {}
    return ProcessorConfig(
        channel="user",
        role="user",
        usage_class="chat",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=DEFAULT_ALWAYS_AVAILABLE,
        discoverable=DEFAULT_DISCOVERABLE,
        blocked=frozenset(),
        max_iterations=None,
        skip_transcript=False,
        skip_input_row=bool(_metadata.get("hidden_input")),
        suppress_history=False,
        broadcast_to="user",
        memory_seed=True,
        post_turn=None,
    )


def make_eamp_config(
    agent_name: str,
    project: str,
    loop_in_human: bool,
    wrapper_id: str,
) -> ProcessorConfig:
    """External-Agent Message Processor config.

    channel='external-agent:{agent_name}', role='external_agent'.
    suppress_history=False (conversational), memory_seed=True.
    post_turn dispatches disclosure when loop_in_human (§3b).

    Prompt builders are stubbed for T1; real implementations land in T7.
    """
    return ProcessorConfig(
        channel=f"external-agent:{agent_name}",
        role="external_agent",
        usage_class="external_agent",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=DEFAULT_ALWAYS_AVAILABLE,
        discoverable=DEFAULT_DISCOVERABLE,
        blocked=frozenset(),
        max_iterations=200,
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        broadcast_to=None,
        memory_seed=True,
        post_turn=None,
    )


def make_pattern_config(window_start: int, window_end: int) -> ProcessorConfig:
    """Pattern-match config — per-window background pattern recognition.

    channel/role='pattern_match', suppress_history=True, max_iterations=100.
    post_turn = confidence decay sweep (§3b).

    Prompt builders are stubbed for T1; real implementations land in T8.
    """
    return ProcessorConfig(
        channel="pattern_match",
        role="pattern_match",
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=["save_pattern", "save_graph"],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=100,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )


def make_geo_config(window_start: int, window_end: int) -> ProcessorConfig:
    """Geo-pattern config — per-window background geo recognition.

    channel/role='geo_pattern', suppress_history=True, max_iterations=30.
    post_turn = log counters only (§3b).

    Prompt builders are stubbed for T1; real implementations land in T8.
    """
    return ProcessorConfig(
        channel="geo_pattern",
        role="geo_pattern",
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=["save_pattern", "save_graph"],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=30,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )


def make_user_summary_config() -> ProcessorConfig:
    """User-summary config — one-shot user synthesis.

    channel/role='user_summary', suppress_history=True, max_iterations=1.
    post_turn parses {short, long} → data_graph (§3b).

    Prompt builders are stubbed for T1; real implementations land in T8.
    """
    return ProcessorConfig(
        channel="user_summary",
        role="user_summary",
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )


def make_super_episode_config(
    channel: str,
    sources: list[Any],
    spans: list[Any],
) -> ProcessorConfig:
    """Super-episode encoder config — per-cluster episode synthesis.

    channel/role='super_episode_encoder', suppress_history=True, max_iterations=1.
    post_turn = no-op (caller owns episode write) (§3b).

    Prompt builders are stubbed for T1; real implementations land in T8.
    """
    return ProcessorConfig(
        channel="super_episode_encoder",
        role="super_episode_encoder",
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )

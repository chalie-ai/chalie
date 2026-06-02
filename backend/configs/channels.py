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

# ── UMP prompt builders ───────────────────────────────────────────────────────

def _ump_build_user_definition(mp: object) -> str:
    """One-sentence synthesis of the real human user.

    Reads user_summary / user_summary_long from data_graph, preferring the
    long form when the converse mode is strongly active.  Falls back to a
    static peer-to-peer framing on any failure or missing row.

    Per-turn cached on mp._user_definition_cached so each ACT iteration is
    cheap.  §3b / spec body-structure §1.
    """
    _FALLBACK = "The user is a real human. Treat this conversation as peer-to-peer dialogue."
    cached = getattr(mp, "_user_definition_cached", None)
    if cached is not None:
        return cached

    try:
        from services.mode_gate_service import STEER_THRESHOLD  # noqa: PLC0415
        mode_state = getattr(mp, "_mode_state_cached", None)
        if mode_state is None:
            mode_state = {}
        prefer_long = mode_state.get("converse", 0.0) >= STEER_THRESHOLD
    except Exception:
        prefer_long = False

    try:
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        dgs = get_data_graph_service()
        rows = dgs.fetch(kinds=["system"], order_by="retrieval_weight DESC")
        by_key = {r.get("key"): r for r in rows if r.get("key")}

        preferred_key = "user_summary_long" if prefer_long else "user_summary"
        entry = by_key.get(preferred_key)
        if (not entry or not entry.get("value")) and prefer_long:
            entry = by_key.get("user_summary")
        if entry and entry.get("value"):
            result = entry["value"]
            mp._user_definition_cached = result  # type: ignore[attr-defined]
            return result
    except Exception:
        pass

    mp._user_definition_cached = _FALLBACK  # type: ignore[attr-defined]
    return _FALLBACK


def _ump_build_system_prompt(mp: object) -> str:
    """UMP system prompt — personality voice + template + mode-gate directives.

    Voice line sits at the very top for cache warmth.  The user_definition is
    NOT emitted here — it lives in the user prompt (spec § Prompt Message
    Definitions).  §3b / §6.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    try:
        from services.personality.personality_service import personality_service  # noqa: PLC0415
        from services.system_message_prompt import UnifiedSystemMessagePrompt  # noqa: PLC0415
        template = UnifiedSystemMessagePrompt().get_prompt()
        voice_line = f"When responding; {personality_service.get_voice()}"
        prompt = f"{voice_line}\n\n{template}"
    except Exception as exc:
        _log.warning("[UMP] system prompt build failed: %s", exc)
        return ""

    # Mode-gate steering directives.
    try:
        mode_gate = getattr(mp, "_mode_gate_cached", None)
        if mode_gate is not None:
            additions = mode_gate.get_system_prompt_additions()
            if additions:
                prompt = f"{prompt}\n\n{additions}"
    except Exception as exc:
        _log.debug("[UMP] mode-gate additions failed: %s", exc)

    return prompt


def _ump_build_user_prompt(mp: object) -> str:
    """UMP user-message body for one ACT iteration.

    Section order (spec § Body structure of build_user_prompt):
      1. User definition (identity anchor — in user prompt, not system prompt).
      2. World State block.
      3. ## Previous Messages.
      (blank separator)
      4. Thinking exploration block (high-thinking mode only).
      5. ACT loop trail (empty before any tools have run).
      6. Input line: user: <raw_input>.

    §3b / §6 / spec §4.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    parts: list[str] = []

    # 1. User definition
    user_def = _ump_build_user_definition(mp)
    if user_def:
        parts.append(user_def)

    # 2. World State
    try:
        from services.world_state import world_state  # noqa: PLC0415
        rendered_ws = world_state.render()
        if rendered_ws:
            _log.info(
                "[WorldState] injected rendered block into user prompt (%d chars)",
                len(rendered_ws),
            )
            parts.append(rendered_ws)
    except Exception as exc:
        _log.debug("[UMP] world_state.render failed: %s", exc)

    # 3. Previous Messages
    try:
        prev = mp._flat_get_previous_messages()  # type: ignore[attr-defined]
        if prev:
            parts.append(f"## Previous Messages\n{prev}")
    except Exception as exc:
        _log.debug("[UMP] get_previous_messages failed: %s", exc)

    # Blank separator
    parts.append("")

    # 4. Thinking exploration (high-thinking mode; None when not active)
    exploration = getattr(mp, "thinking_exploration", None)
    if exploration:
        parts.append(
            "## Chain of Thought\n"
            "Below is your initial reaction to this prompt, played back. "
            "Use it as grounding but pivot as needed based on the conversation.\n\n"
            "---\n\n"
            f"{exploration}\n\n"
            "---"
        )

    # 5. ACT loop trail
    try:
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(trail)
    except Exception as exc:
        _log.debug("[UMP] _render_act_trail failed: %s", exc)

    # 6. Input line with optional nudge
    nudge_tag = (getattr(mp, "_metadata", None) or {}).get("nudge_tag") or ""
    turn_line = f"user: {mp._raw_input}"  # type: ignore[attr-defined]
    if nudge_tag:
        turn_line += " " + nudge_tag
    parts.append(turn_line)

    return "\n".join(parts)


def _ump_post_turn(mp: object, response_text: str) -> None:
    """UMP post-turn: proactive skill suggestion only.  No metrics, no phase.

    Fires only on clean ACT exits with 4+ tool-calling iterations.
    Non-blocking (daemon thread inside the service).

    §3b / §4e / §6.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    # loop_exited_cleanly and current_iteration live on the flat mp as
    # _loop_exited_cleanly / _current_iteration (old attrs) — inspect both.
    exited_cleanly = (
        getattr(mp, "_loop_exited_cleanly", False)
        or getattr(mp, "loop_exited_cleanly", False)
    )
    iteration = (
        getattr(mp, "current_iteration", None)
        or getattr(mp, "_current_iteration", 0)
    )
    if exited_cleanly and iteration >= 4:
        try:
            from services.skill_suggestion_message_processor import maybe_suggest_skill  # noqa: PLC0415
            act_trail = getattr(mp, "_act_trail", [])
            raw_input = getattr(mp, "_raw_input", "")
            maybe_suggest_skill(act_trail, raw_input)
        except Exception as exc:
            _log.warning("[POSTTURN] skill suggestion failed: %s", exc)


def make_user_config(metadata: dict[str, Any] | None = None) -> ProcessorConfig:
    """UMP config — conversational user channel.

    broadcast_to='user' (live output), memory_seed=True, suppress_history=False.
    Attachments auto-fire document.upload on turn 0 (no flag needed — presence
    of metadata['attachments'] drives this).  post_turn = skill suggestion only
    (no metrics, no phase — §3b / §4e / §6).
    """
    _metadata = metadata or {}
    return ProcessorConfig(
        channel="user",
        role="user",
        usage_class="chat",
        build_user_prompt=_ump_build_user_prompt,
        build_user_definition=_ump_build_user_definition,
        build_system_prompt=_ump_build_system_prompt,
        always_available=DEFAULT_ALWAYS_AVAILABLE,
        discoverable=DEFAULT_DISCOVERABLE,
        blocked=frozenset(),
        max_iterations=None,
        skip_transcript=False,
        skip_input_row=bool(_metadata.get("hidden_input")),
        suppress_history=False,
        broadcast_to="user",
        memory_seed=True,
        post_turn=_ump_post_turn,
    )


# ── EAMP prompt builders ──────────────────────────────────────────────────────

def _eamp_build_user_definition(agent_name: str, project: str) -> Any:
    """Return a builder callable for the EAMP user definition.

    Returns a zero-arg-relative callable that, when called with mp, returns
    the static agent identity string.  §3b.
    """
    _text = (
        f"The user is {agent_name}, an external agent. "
        f"This conversation is about: {project}."
    )

    def _build(_mp: object) -> str:
        return _text

    return _build


def _eamp_build_system_prompt(agent_name: str, project: str, wrapper_id: str) -> Any:
    """Return a builder callable for the EAMP system prompt.

    Fills in {user_name}, {agent_name}, {project_or_task_name} template
    variables from data_graph and the EAMP constructor args.  §3b.
    """
    _agent_name = agent_name
    _project = project

    def _build(_mp: object) -> str:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            from services.system_message_prompt import ExternalAgentSystemMessagePrompt  # noqa: PLC0415
            body = ExternalAgentSystemMessagePrompt().get_prompt()

            # Substitute {{provider_content_field_name}} if present.
            if "{{provider_content_field_name}}" in body:
                try:
                    from services.providers import Providers  # noqa: PLC0415
                    provider = Providers.instance()._resolve("external_agent")
                    label = getattr(provider, "CONTENT_FIELD_LABEL", None)
                    if label:
                        body = body.replace("{{provider_content_field_name}}", label)
                except Exception:
                    pass

            # Resolve the user's first name from data_graph.
            user_name = "the user"
            try:
                from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
                dgs = get_data_graph_service()
                rows = dgs.fetch(kinds=["system"])
                for row in rows:
                    key = row.get("key") if isinstance(row, dict) else getattr(row, "key", None)
                    val = row.get("value") if isinstance(row, dict) else getattr(row, "value", None)
                    if key == "user_summary" and val:
                        first_word = val.split()[0] if val else ""
                        if first_word:
                            user_name = first_word
                        break
            except Exception:
                pass

            user_def = (
                f"The user is {_agent_name}, an external agent. "
                f"This conversation is about: {_project}."
            )
            body = (
                body
                .replace("{user_name}", user_name)
                .replace("{agent_name}", _agent_name)
                .replace("{project_or_task_name}", _project)
            )
            return f"{user_def}\n\n{body}"
        except Exception as exc:
            _log.warning("[EAMP] system prompt build failed: %s", exc)
            return ""

    return _build


def _eamp_build_user_prompt(mp: object) -> str:
    """EAMP user-message body for one ACT iteration.

    Stripped compared to UMP: no world state, no user definition (it lives
    in the system prompt for EAMP).  Keeps: previous messages, ACT trail,
    input line.  §3b.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    parts: list[str] = []

    # Previous Messages
    try:
        prev = mp._flat_get_previous_messages()  # type: ignore[attr-defined]
        if prev:
            parts.append(f"## Previous Messages\n{prev}")
    except Exception as exc:
        _log.debug("[EAMP] get_previous_messages failed: %s", exc)

    parts.append("")

    # ACT loop trail
    try:
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(trail)
    except Exception as exc:
        _log.debug("[EAMP] _render_act_trail failed: %s", exc)

    # Input line
    parts.append(f"user: {mp._raw_input}")  # type: ignore[attr-defined]

    return "\n".join(parts)


def _make_eamp_post_turn(
    agent_name: str,
    project: str,
    loop_in_human: bool,
) -> Any:
    """Return a post_turn callable for EAMP.

    When loop_in_human=True, dispatches a disclosure message to the user
    channel after the ACT loop completes.  §3b / §4d.
    """
    if not loop_in_human:
        return None

    _agent_name = agent_name
    _project = project

    def _post_turn(mp: object, response_text: str) -> None:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        raw_input = getattr(mp, "_raw_input", "")
        disclosure_input = (
            f"An external agent called '{_agent_name}' just contacted you "
            f"about '{_project}'. "
            f"Here's what they said:\n\n\"{raw_input}\"\n\n"
            f"You replied:\n\n\"{response_text}\"\n\n"
            "Let the user know about this exchange in your own words."
        )
        try:
            from api.chat import dispatch_message  # noqa: PLC0415
            dispatch_message(disclosure_input, source="external_agent", hidden_input=True)
        except Exception as exc:
            _log.warning("[EAMP] disclosure dispatch failed: %s", exc)

    return _post_turn


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
    """
    channel = f"external-agent:{agent_name}"
    return ProcessorConfig(
        channel=channel,
        role="external_agent",
        usage_class="external_agent",
        build_user_prompt=_eamp_build_user_prompt,
        build_user_definition=_eamp_build_user_definition(agent_name, project),
        build_system_prompt=_eamp_build_system_prompt(agent_name, project, wrapper_id),
        always_available=DEFAULT_ALWAYS_AVAILABLE,
        discoverable=DEFAULT_DISCOVERABLE,
        blocked=frozenset(),
        max_iterations=200,
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        broadcast_to=None,
        memory_seed=True,
        post_turn=_make_eamp_post_turn(agent_name, project, loop_in_human),
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

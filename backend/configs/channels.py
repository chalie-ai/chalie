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

T7 wired UMP and EAMP with real implementations.
T8 wired all background/housekeeping channels with real implementations.
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

# ── DMN prompt builders ───────────────────────────────────────────────────────

_EPISODE_RETRIEVAL_WEIGHT_FLOOR = 0.3
_DMN_EPISODE_LOOKBACK_DAYS = 30
_DMN_EPISODE_LIMIT = 50


def _dmn_build_user_definition(_mp: object) -> str:
    """DMN runs as a background process — no human user definition needed."""
    return (
        "The user is 'proactive_thought' — a special background process "
        "that represents your own reflections on recent activity."
    )


def _dmn_fetch_user_synthesis() -> str:
    """Read user synthesis from data_graph.

    Prefers user_summary_long for richer DMN reflection context.
    Falls back to user_summary. Returns '' when neither row exists.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM data_graph "
                "WHERE kind = 'system' "
                "  AND key IN ('user_summary', 'user_summary_long') "
                "  AND active = 1 AND deleted_at IS NULL",
            ).fetchall()
        by_key = {row[0]: row[1] for row in rows if row[1]}
        return by_key.get("user_summary_long") or by_key.get("user_summary") or ""
    except Exception as exc:
        _log.warning("[DMN_CONFIG] _dmn_fetch_user_synthesis failed: %s", exc)
        return ""


def _dmn_fetch_recent_episodes() -> str:
    """Retrieve recent, non-decayed user-channel episodes as a numbered list."""
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, gist, salience, created_at "
                "FROM episodes "
                "WHERE deleted_at IS NULL "
                "  AND channel = 'user' "
                "  AND retrieval_weight >= ? "
                "  AND ( "
                "      last_accessed_at >= datetime('now', ?) "
                "      OR created_at >= datetime('now', ?) "
                "  ) "
                "ORDER BY retrieval_weight DESC, created_at DESC "
                "LIMIT ?",
                (
                    _EPISODE_RETRIEVAL_WEIGHT_FLOOR,
                    f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                    f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                    _DMN_EPISODE_LIMIT,
                ),
            ).fetchall()
        lines = []
        for i, (ep_id, gist, salience, created_at) in enumerate(rows, 1):
            ts = (created_at or "")[:16].replace("T", " ")
            lines.append(f"{i}. [{ts}] (salience={salience}) {gist or ''}")
        return "\n".join(lines)
    except Exception as exc:
        _log.warning("[DMN_CONFIG] _dmn_fetch_recent_episodes failed: %s", exc)
        return ""


def _dmn_build_user_prompt(mp: object) -> str:
    """DMN user-message: user synthesis + filtered recent episodes + ACT trail."""
    parts: list[str] = []
    synthesis = _dmn_fetch_user_synthesis()
    if synthesis:
        parts.append(f"## About the User\n{synthesis}")
    episodes_text = _dmn_fetch_recent_episodes()
    if episodes_text:
        parts.append(f"## Episodes\n{episodes_text}")
    try:
        trail = mp.get_act_loop_trail()  # type: ignore[attr-defined]
        if trail and isinstance(trail, str):
            parts.append(trail)
    except Exception:
        pass
    return "\n\n".join(parts)


def _dmn_build_system_prompt(_mp: object) -> str:
    """DMN system prompt from DMNSystemMessagePrompt."""
    from services.system_message_prompt import DMNSystemMessagePrompt  # noqa: PLC0415
    return DMNSystemMessagePrompt().get_prompt()


DMN_CONFIG = ProcessorConfig(
    channel="dmn",
    role="proactive_thought",
    usage_class="subconscious",
    build_user_prompt=_dmn_build_user_prompt,
    build_user_definition=_dmn_build_user_definition,
    build_system_prompt=_dmn_build_system_prompt,
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
"""DMN background channel.  §3a / §8b.  post_turn=None (metrics moved to gateway §4e)."""

# ── Episode encoder prompt builders ──────────────────────────────────────────


def _episode_encoder_build_user_definition(_mp: object) -> str:
    return (
        "The user is 'episode_encoder' — a background process that "
        "summarises transcript windows into memory snapshots."
    )


def _episode_encoder_build_user_prompt(mp: object) -> str:
    """Episode encoder user-prompt: transcript window + referenced episodes.

    Reads _window and _referenced from the mp instance (set by the caller
    before calling MessageProcessor.process()).
    """
    window = getattr(mp, "_window", "") or ""
    referenced = getattr(mp, "_referenced", "") or ""
    parts = [
        "Transcript window — each line is `[id] (timestamp) role: content`:",
        "",
        window,
    ]
    if referenced:
        parts.extend([
            "",
            "Episodes referenced during these turns (candidates for update / delete):",
            "",
            referenced,
        ])
    return "\n".join(parts)


def _episode_encoder_build_system_prompt(_mp: object) -> str:
    """Episode encoder system prompt from EpisodeEncoderSystemPrompt."""
    from services.system_message_prompt import EpisodeEncoderSystemPrompt  # noqa: PLC0415
    return EpisodeEncoderSystemPrompt().get_prompt()


EPISODE_ENCODER_CONFIG = ProcessorConfig(
    channel="episode_encoder",
    role="episode_encoder",
    usage_class="subconscious",
    build_user_prompt=_episode_encoder_build_user_prompt,
    build_user_definition=_episode_encoder_build_user_definition,
    build_system_prompt=_episode_encoder_build_system_prompt,
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

# ── Skill suggestion prompt builders ─────────────────────────────────────────


def _skill_suggestion_build_user_prompt(mp: object) -> str:
    """Skill suggestion user-prompt: original request + ACT trail.

    Reads _original_trail, _original_input, _iteration_count from mp (set by
    the caller before calling MessageProcessor.process()).
    """
    original_trail = getattr(mp, "_original_trail", []) or []
    original_input = getattr(mp, "_original_input", "") or ""
    iteration_count = getattr(mp, "_iteration_count", len(original_trail))
    parts = [
        f"## Original User Request\n{original_input}",
        f"\n## Completed ACT Loop Trail ({iteration_count} iterations)",
    ]
    for entry in original_trail:
        parts.append(entry)
    try:
        trail = mp.get_act_loop_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(f"\n{trail}")
    except Exception:
        pass
    return "\n".join(parts)


def _skill_suggestion_build_system_prompt(_mp: object) -> str:
    """Skill suggestion system prompt from SkillSuggestionSystemPrompt."""
    from services.system_message_prompt import SkillSuggestionSystemPrompt  # noqa: PLC0415
    return SkillSuggestionSystemPrompt().get_prompt()


SKILL_SUGGESTION_CONFIG = ProcessorConfig(
    channel="skills_building",
    role="skills_building",
    usage_class="subconscious",
    build_user_prompt=_skill_suggestion_build_user_prompt,
    build_user_definition=lambda _mp: "",
    build_system_prompt=_skill_suggestion_build_system_prompt,
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
"""Skill suggestion — housekeeping, suppress_history=True replaces old
get_previous_messages() override.  §3a / AC-26."""

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


# ── Pattern-match prompt builders and post_turn ───────────────────────────────

_TOP_PATTERN_CAP = 50


def _pattern_existing_patterns_block() -> str:
    """Return the top-confidence active behavioral_pattern rows as JSON."""
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT value FROM data_graph "
                "WHERE kind='behavioral_pattern' AND active=1 "
                "AND deleted_at IS NULL "
                "AND json_extract(value, '$.confidence') IS NOT NULL "
                "ORDER BY CAST(json_extract(value, '$.confidence') AS REAL) "
                "DESC LIMIT ?",
                (_TOP_PATTERN_CAP,),
            ).fetchall()
        if not rows:
            return "(none yet)"
        patterns = {}
        for (val,) in rows:
            try:
                d = _json.loads(val) or {}
                name = d.get("name")
                if name:
                    patterns[name] = d.get("summary", "")
            except Exception:
                continue
        return _json.dumps(patterns, indent=2) if patterns else "(none yet)"
    except Exception as exc:
        _log.warning("[PATTERN_CONFIG] existing_patterns_block failed: %s", exc)
        return "(none yet)"


def _pattern_build_user_prompt(mp: object, window_start: int, window_end: int) -> str:
    """Pattern-match user-prompt: transcripts from window + existing patterns + trail."""
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM transcript "
                "WHERE id > ? AND id <= ? "
                "AND content IS NOT NULL AND content != '' "
                "ORDER BY id ASC",
                (window_start, window_end),
            ).fetchall()
        if not rows:
            return "(no transcripts in window)"
        transcript_block = "\n".join(
            f"[id={r[0]} | {r[1]} | {r[3]}] {r[2]}" for r in rows
        )
    except Exception as exc:
        _log.warning("[PATTERN_CONFIG] transcript fetch failed: %s", exc)
        transcript_block = "(transcript fetch failed)"

    existing = _pattern_existing_patterns_block()
    parts = [f"Existing patterns:\n{existing}", transcript_block]
    try:
        trail = mp.get_act_loop_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(trail)
    except Exception:
        pass
    return "\n\n".join(parts)


def _pattern_build_system_prompt(_mp: object) -> str:
    """Pattern-match system prompt (inlined from PatternMatchProcessor.get_system_prompt)."""
    return (
        "You are analysing the user's recent transcripts to detect "
        "repeating behavioural patterns and surface durable life-graph "
        "facts.\n\n"
        "You have ONE forward pass. Emit ALL tool calls in parallel. Do "
        "NOT loop — results are intentionally minimal.\n\n"
        "save_pattern summaries must be 1 sentence and concise. "
        "Examples:\n"
        "- User walks to work every morning at 09:00\n"
        "- User prefers cold-beverages\n"
        "- User goes to the gym daily, except Sundays\n"
        "- User's gym schedule is: Monday weights, Tuesday boxing\n"
        "Do NOT write narratives, episode summaries, or date-stamped "
        "event logs. The summary is a distilled habit, not a story.\n\n"
        "Rules:\n"
        "- save_pattern requires >=2 evidence rows in this batch.\n"
        "- Mirror existing pattern names exactly when reinforcing — "
        "case-sensitive.\n"
        "- Do NOT use save_graph for repeating behaviours — use "
        "save_pattern.\n"
        "- Skip noise. Skip one-offs. Skip ambiguous interpretations.\n"
        "- Emit everything in this single pass."
    )


def _pattern_post_turn(mp: object, _response_text: str) -> None:
    """Confidence decay sweep: -0.005 on untouched active rows; soft-delete at <=0.

    §3b / §4e — no metrics recorded here (metrics moved to send gateway).
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        touched_ids: set = getattr(mp, "_touched_pattern_ids", set()) or set()
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.time_utils import utc_now  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            # Decrement confidence on untouched rows.
            params: list = []
            touched_filter = ""
            if touched_ids:
                placeholders = ",".join("?" * len(touched_ids))
                touched_filter = f"AND id NOT IN ({placeholders})"
                params = list(touched_ids)
            conn.execute(
                f"""
                UPDATE data_graph
                SET value = json_set(
                      value,
                      '$.confidence',
                      MAX(0.0, CAST(json_extract(value, '$.confidence') AS REAL) - 0.005)
                    )
                WHERE kind = 'behavioral_pattern'
                  AND active = 1
                  AND deleted_at IS NULL
                  AND json_extract(value, '$.confidence') IS NOT NULL
                  {touched_filter}
                """,
                params,
            )
            # Soft-delete rows whose confidence dropped to <=0.
            conn.execute(
                """
                UPDATE data_graph
                SET active = 0
                WHERE kind = 'behavioral_pattern'
                  AND active = 1
                  AND CAST(json_extract(value, '$.confidence') AS REAL) <= 0.0
                """,
            )
        save_pattern_calls = getattr(mp, "_save_pattern_calls", 0)
        save_graph_calls = getattr(mp, "_save_graph_calls", 0)
        now_iso = utc_now().isoformat()
        _log.info(
            "[PATTERN_CONFIG] done save_pattern=%d save_graph=%d "
            "touched=%d at=%s",
            save_pattern_calls,
            save_graph_calls,
            len(touched_ids),
            now_iso,
        )
    except Exception as exc:
        _log.warning("[PATTERN_CONFIG] decay sweep failed: %s", exc)


def _pattern_init_instance_state(mp: object) -> None:
    """Initialise per-instance counter/state attrs that SavePattern/SaveGraph read."""
    if not hasattr(mp, "_save_pattern_calls"):
        mp._save_pattern_calls = 0  # type: ignore[attr-defined]
    if not hasattr(mp, "_save_graph_calls"):
        mp._save_graph_calls = 0  # type: ignore[attr-defined]
    if not hasattr(mp, "_save_graph_seen"):
        mp._save_graph_seen = set()  # type: ignore[attr-defined]
    if not hasattr(mp, "_touched_pattern_ids"):
        mp._touched_pattern_ids = set()  # type: ignore[attr-defined]


def make_pattern_config(window_start: int, window_end: int) -> ProcessorConfig:
    """Pattern-match config — per-window background pattern recognition.

    channel/role='pattern_match', suppress_history=True, max_iterations=100.
    post_turn = confidence decay sweep (§3b).

    Counter/state attrs are lazily initialised by build_user_prompt on the first
    call so the caller does not need to pre-set them.
    """
    def _build_user_prompt(mp: object) -> str:
        # Lazy-init per-instance state so SavePattern/SaveGraph find it.
        _pattern_init_instance_state(mp)
        return _pattern_build_user_prompt(mp, window_start, window_end)

    return ProcessorConfig(
        channel="pattern_match",
        role="pattern_match",
        usage_class="subconscious",
        build_user_prompt=_build_user_prompt,
        build_user_definition=lambda _mp: "",
        build_system_prompt=_pattern_build_system_prompt,
        always_available=["save_pattern", "save_graph"],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=100,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=_pattern_post_turn,
    )


# ── Geo-pattern prompt builders and post_turn ─────────────────────────────────


def _geo_pattern_load_transcript_block(window_start: int, window_end: int) -> str:
    """Fetch location-tagged transcripts from the window and format them."""
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at, "
                "location_lat, location_lon, location_name "
                "FROM transcript "
                "WHERE id > ? AND id <= ? "
                "AND location_lat IS NOT NULL AND location_lon IS NOT NULL "
                "AND content IS NOT NULL AND content != '' "
                "ORDER BY id ASC",
                (window_start, window_end),
            ).fetchall()
        if not rows:
            return "(no location-tagged transcripts in window)"
        lines = []
        for r in rows:
            row_id, role, content, created_at, lat, lon, place_name = r
            location_str = f"{lat},{lon}"
            if place_name:
                location_str = f"{lat},{lon} {place_name}"
            lines.append(
                f"[id={row_id} | {role} | {created_at} | {location_str}] {content}"
            )
        return "\n".join(lines)
    except Exception as exc:
        _log.warning("[GEO_CONFIG] transcript block failed: %s", exc)
        return "(transcript fetch failed)"


def _geo_pattern_build_system_prompt(_mp: object) -> str:
    """Geo-pattern system prompt (inlined from GeoPatternProcessor.get_system_prompt)."""
    return (
        "You are analysing the user's recent location-tagged transcripts "
        "to detect geo-spatial behavioural patterns — routines, habits, "
        "and durable facts tied to specific physical places.\n\n"
        "You have ONE forward pass. Emit ALL tool calls in parallel. Do "
        "NOT loop — results are intentionally minimal.\n\n"
        "save_pattern summaries must be 1 sentence and concise. "
        "Examples:\n"
        "- User walks to work every morning at 09:00\n"
        "- User goes to the gym daily, except Sundays\n"
        "- User's gym schedule is: Monday weights, Tuesday boxing\n"
        "Do NOT write narratives, episode summaries, or date-stamped "
        "event logs. The summary is a distilled habit, not a story.\n\n"
        "Rules:\n"
        "- save_pattern requires >=2 evidence rows in this batch.\n"
        "- Mirror existing pattern names exactly when reinforcing — "
        "case-sensitive.\n"
        "- Only emit patterns where physical place is central. Another "
        "processor handles location-independent patterns.\n"
        "- Do NOT use save_graph for repeating behaviours — use "
        "save_pattern.\n"
        "- Skip noise. Skip one-offs. Skip ambiguous interpretations.\n"
        "- Emit everything in this single pass."
    )


def _geo_pattern_post_turn(mp: object, _response_text: str) -> None:
    """Log completion counters. Decay is handled by pattern_match, not geo_pattern.

    §3b — no metrics, no decay sweep.
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.time_utils import utc_now  # noqa: PLC0415
        save_pattern_calls = getattr(mp, "_save_pattern_calls", 0)
        save_graph_calls = getattr(mp, "_save_graph_calls", 0)
        touched_ids = getattr(mp, "_touched_pattern_ids", set()) or set()
        now_iso = utc_now().isoformat()
        _log.info(
            "[GEO_CONFIG] done save_pattern=%d save_graph=%d "
            "touched=%d at=%s",
            save_pattern_calls,
            save_graph_calls,
            len(touched_ids),
            now_iso,
        )
    except Exception as exc:
        _log.warning("[GEO_CONFIG] post_turn logging failed: %s", exc)


def make_geo_config(window_start: int, window_end: int) -> ProcessorConfig:
    """Geo-pattern config — per-window background geo recognition.

    channel/role='geo_pattern', suppress_history=True, max_iterations=30.
    post_turn = log counters only (§3b).

    Counter/state attrs are lazily initialised by build_user_prompt on the first
    call so the caller does not need to pre-set them.
    """
    def _build_user_prompt(mp: object) -> str:
        # Lazy-init per-instance state so SavePattern/SaveGraph find it.
        if not hasattr(mp, "_save_pattern_calls"):
            mp._save_pattern_calls = 0  # type: ignore[attr-defined]
        if not hasattr(mp, "_save_graph_calls"):
            mp._save_graph_calls = 0  # type: ignore[attr-defined]
        if not hasattr(mp, "_save_graph_seen"):
            mp._save_graph_seen = set()  # type: ignore[attr-defined]
        if not hasattr(mp, "_touched_pattern_ids"):
            mp._touched_pattern_ids = set()  # type: ignore[attr-defined]
        # Cache the transcript block across multiple iterations (same as GPP).
        cached = getattr(mp, "_cached_transcript_block", None)
        if cached is None:
            block = _geo_pattern_load_transcript_block(window_start, window_end)
            mp._cached_transcript_block = block  # type: ignore[attr-defined]
        else:
            block = cached
        existing = _pattern_existing_patterns_block()
        parts = [f"Existing patterns:\n{existing}", block]
        try:
            trail = mp.get_act_loop_trail()  # type: ignore[attr-defined]
            if trail:
                parts.append(trail)
        except Exception:
            pass
        return "\n\n".join(parts)

    return ProcessorConfig(
        channel="geo_pattern",
        role="geo_pattern",
        usage_class="subconscious",
        build_user_prompt=_build_user_prompt,
        build_user_definition=lambda _mp: "",
        build_system_prompt=_geo_pattern_build_system_prompt,
        always_available=["save_pattern", "save_graph"],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=30,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=_geo_pattern_post_turn,
    )


# ── User-summary prompt builders and post_turn ────────────────────────────────

_MAX_TRAIT_ROWS = 200
_MAX_PATTERN_ROWS = 25


def _format_pattern_line(content: dict) -> str:
    """Render a single behavioral_pattern content dict as a compact one-liner."""
    name = content.get("name", "unknown")
    freq = content.get("frequency", "?")
    anchor = content.get("time_anchor") or ""
    summary = content.get("summary", "")
    confidence = content.get("confidence", 0)
    last_seen = (content.get("last_seen_at") or "")[:10] or "?"
    anchor_part = f" @ {anchor}" if anchor else ""
    return (
        f"{name} ({freq}{anchor_part}): {summary} "
        f"[confidence={confidence}, last {last_seen}]"
    )


def _user_summary_build_user_prompt(_mp: object) -> str:
    """User-summary user-prompt: user_specific traits + behavioral_patterns."""
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)

    # Section 1: user_specific traits
    try:
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        rows = get_data_graph_service().fetch(
            kinds=["user_specific"],
            limit=_MAX_TRAIT_ROWS,
            order_by="retrieval_weight DESC",
        )
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] trait fetch failed: %s", exc)
        rows = []

    if not rows:
        facts_section = "Facts:\n(no facts available)"
    else:
        lines = [
            f"{r['key']}: {r['value']}"
            for r in rows
            if r.get("key") and r.get("value")
        ]
        facts_section = "Facts:\n" + "\n".join(lines) if lines else "Facts:\n(no facts available)"

    # Section 2: active behavioral_patterns
    active_patterns = []
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            pattern_rows = conn.execute(
                """
                SELECT value
                FROM data_graph
                WHERE kind = 'behavioral_pattern'
                  AND active = 1
                  AND deleted_at IS NULL
                ORDER BY last_confirmed_at DESC
                LIMIT ?
                """,
                (_MAX_PATTERN_ROWS,),
            ).fetchall()
        for (value_json,) in pattern_rows:
            try:
                content = _json.loads(value_json)
                if content:
                    active_patterns.append(content)
            except Exception:
                continue
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] active pattern fetch failed: %s", exc)

    if not active_patterns:
        return facts_section

    pattern_lines = [_format_pattern_line(p) for p in active_patterns]
    patterns_section = (
        "## Behavioural patterns (frequency, last seen)\n"
        + "\n".join(f"- {line}" for line in pattern_lines)
    )
    return facts_section + "\n\n" + patterns_section


def _user_summary_build_system_prompt(_mp: object) -> str:
    """User-summary system prompt from UserSummarySystemPrompt."""
    from services.system_message_prompt import UserSummarySystemPrompt  # noqa: PLC0415
    return UserSummarySystemPrompt().get_prompt()


def _user_summary_post_turn(_mp: object, response_text: str) -> None:
    """Parse JSON {short, long} from response_text and write to data_graph.

    AC-29: receives response_text directly — no on_store hook, no _last_response cache.
    §3b: post_turn(mp, response_text) is the only callback.
    """
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)

    text = (response_text or "").strip()
    if not text:
        _log.warning("[USER_SUMMARY_CONFIG] post_turn: empty LLM response — skipping write")
        return

    # Strip markdown code fences if the model wrapped the JSON.
    stripped = text.removeprefix("```json").removeprefix("```").lstrip()
    stripped = stripped.removesuffix("```").rstrip()

    try:
        parsed = _json.loads(stripped)
    except _json.JSONDecodeError as exc:
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: JSON parse failed (%s) — skipping write. raw=%r",
            exc,
            text[:200],
        )
        return

    if not isinstance(parsed, dict):
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: parsed value is not a dict — skipping write"
        )
        return

    short = (parsed.get("short") or "").strip()
    long_ = (parsed.get("long") or "").strip()

    if not short or not long_:
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: 'short' or 'long' missing/empty — skipping write"
        )
        return

    try:
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        dgs = get_data_graph_service()
        # Write user_summary_long FIRST so crash recovery works correctly.
        # See UserSummaryProcessor.post_turn() docstring for the ordering rationale.
        dgs.store(
            kind="system",
            key="user_summary_long",
            value=long_,
            source="user_summary_config",
        )
        dgs.store(
            kind="system",
            key="user_summary",
            value=short,
            source="user_summary_config",
        )
        _log.info(
            "[USER_SUMMARY_CONFIG] post_turn: wrote user_summary (%d chars) and "
            "user_summary_long (%d chars)",
            len(short),
            len(long_),
        )
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] post_turn: data_graph write failed: %s", exc)


def _should_synthesise() -> bool:
    """Return True when re-synthesis is warranted.

    Logic:
    - latest_trait_ts = MAX(last_confirmed_at) WHERE kind IN
      ('user_specific', 'behavioral_pattern') AND active=1
    - current_summary = row WHERE kind='system' AND key='user_summary' AND active=1

    Cases:
    - No traits or patterns at all → False (nothing to synthesise).
    - Traits/patterns exist, no summary → True (first synthesis needed).
    - Both exist → True only when latest trait/pattern is newer than the summary row.
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.time_utils import parse_utc  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(last_confirmed_at)
                FROM data_graph
                WHERE kind IN ('user_specific', 'behavioral_pattern')
                  AND active = 1
                  AND deleted_at IS NULL
                """
            ).fetchone()
            latest_trait_ts = row[0] if row else None
            if latest_trait_ts is None:
                return False
            summary_row = conn.execute(
                """
                SELECT last_confirmed_at
                FROM data_graph
                WHERE kind = 'system'
                  AND key = 'user_summary'
                  AND active = 1
                  AND deleted_at IS NULL
                LIMIT 1
                """
            ).fetchone()
        if summary_row is None:
            return True
        try:
            trait_dt = parse_utc(latest_trait_ts)
            summary_dt = parse_utc(summary_row[0])
        except Exception as exc:
            _log.error(
                "[USER_SUMMARY_CONFIG] _should_synthesise: parse_utc failed "
                "trait_ts=%r summary_ts=%r: %s",
                latest_trait_ts,
                summary_row[0],
                exc,
            )
            return False
        return trait_dt > summary_dt
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] _should_synthesise failed: %s", exc)
        return False


def make_user_summary_config() -> ProcessorConfig:
    """User-summary config — one-shot user synthesis.

    channel/role='user_summary', suppress_history=True, max_iterations=1.
    post_turn parses {short, long} → data_graph (§3b / AC-29).

    The caller gates on _should_synthesise() BEFORE calling
    MessageProcessor.process() — §3c / O1.
    """
    return ProcessorConfig(
        channel="user_summary",
        role="user_summary",
        usage_class="subconscious",
        build_user_prompt=_user_summary_build_user_prompt,
        build_user_definition=lambda _mp: "You are a synthesiser. The user is a real human whose traits you are distilling.",
        build_system_prompt=_user_summary_build_system_prompt,
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=_user_summary_post_turn,
    )


# ── Super-episode prompt builders ────────────────────────────────────────────


def _super_episode_build_system_prompt(_mp: object) -> str:
    """Super-episode system prompt from SuperEpisodeEncoderSystemPrompt."""
    from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt  # noqa: PLC0415
    return SuperEpisodeEncoderSystemPrompt().get_prompt()


def make_super_episode_config(
    channel: str,
    sources: list[Any],
    spans: Any,
) -> ProcessorConfig:
    """Super-episode encoder config — per-cluster episode synthesis.

    channel/role='super_episode_encoder', suppress_history=True, max_iterations=1.
    post_turn = None (caller owns episode write — §3b / O2).

    sources and spans are captured at factory time so the build_user_prompt
    closure is self-contained per cluster (§3c: cluster loop moves to caller).

    Args:
        channel: The user channel being consolidated (e.g. 'user').
        sources: List of episode dicts for this cluster (each with 'id', 'gist').
        spans:   Raw transcript spans string covering these episodes.
    """
    _sources = sources
    _spans = spans

    def _build_user_prompt(_mp: object) -> str:
        src = "\n\n".join(f"[{e['id']}] {e['gist']}" for e in _sources)
        return (
            f"Source episodes:\n\n{src}\n\n"
            f"Raw transcript spans covering these episodes:\n\n{_spans}"
        )

    def _build_user_definition(_mp: object) -> str:
        return (
            "The user is 'super_episode_encoder' — a background process that "
            "consolidates clusters of related episodes into a single super-episode."
        )

    return ProcessorConfig(
        channel="super_episode_encoder",
        role="super_episode_encoder",
        usage_class="subconscious",
        build_user_prompt=_build_user_prompt,
        build_user_definition=_build_user_definition,
        build_system_prompt=_super_episode_build_system_prompt,
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

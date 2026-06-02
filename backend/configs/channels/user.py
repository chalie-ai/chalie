from __future__ import annotations

from typing import Any

from services.processor_config import ProcessorConfig

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE, DEFAULT_DISCOVERABLE

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
        prev = mp.get_previous_messages()  # type: ignore[attr-defined]
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
    exited_cleanly = getattr(mp, "loop_exited_cleanly", False)
    iteration = getattr(mp, "current_iteration", 0)
    if exited_cleanly and iteration >= 4:
        try:
            from services.skill_suggestion_message_processor import maybe_suggest_skill  # noqa: PLC0415
            rendered = mp._render_act_trail()  # type: ignore[attr-defined]
            act_trail = rendered.split("\n") if rendered else []
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

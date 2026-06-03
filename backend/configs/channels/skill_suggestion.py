from __future__ import annotations

from services.processor_config import ProcessorConfig

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
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(f"\n{trail}")
    except Exception:
        pass
    return "\n".join(parts)


def _skill_suggestion_build_system_prompt(_mp: object) -> str:
    """Skill suggestion system prompt from SkillSuggestionSystemPrompt."""
    from services.system_message_prompt import SkillSuggestionSystemPrompt  # noqa: PLC0415
    return SkillSuggestionSystemPrompt().get_prompt()


class SkillSuggestionConfig(ProcessorConfig):
    """Skill suggestion — housekeeping, suppress_history=True replaces old
    get_previous_messages() override.  §3a / AC-26."""

    def __init__(self) -> None:
        super().__init__(
            channel="skills_building",
            role="skills_building",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
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

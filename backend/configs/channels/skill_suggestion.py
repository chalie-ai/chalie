from __future__ import annotations

from services.processor_config import ProcessorConfig
from services.message_processor import MessageProcessor


class SkillSuggestionConfig(ProcessorConfig):
    """§3a / AC-26 — suppress_history=True replaces the old
    get_previous_messages() override."""

    def __init__(self) -> None:
        super().__init__(
            channel="skills_building",
            role="skills_building",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["skill_manager"],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=5,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: MessageProcessor) -> str:
        return ""

    def get_user_prompt(self, mp: MessageProcessor) -> str:
        """Reads _original_trail, _original_input, _iteration_count from mp
        (set by the caller before MessageProcessor.process())."""
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
            trail = mp._render_act_trail()
            if trail:
                parts.append(f"\n{trail}")
        except Exception:
            pass
        return "\n".join(parts)

    def get_system_prompt(self, mp: MessageProcessor) -> str:
        """OLD assembly ``f"\n\n{body}"`` — this channel's get_user_definition
        is empty so the leading blank lines are reproduced for parity."""
        from services.system_message_prompt import SkillSuggestionSystemPrompt  # noqa: PLC0415
        return f"\n\n{SkillSuggestionSystemPrompt().get_prompt()}"

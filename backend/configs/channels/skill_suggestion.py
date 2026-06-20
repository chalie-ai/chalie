from __future__ import annotations

from typing import TYPE_CHECKING

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor


class SkillSuggestionConfig(ProcessorConfig):
    """§3a / AC-26 — suppress_history=True replaces the old"""

    def __init__(self) -> None:
        super().__init__(
            channel="skills_building",
            role="skills_building",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["skill_manager"],
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return ""

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """Derive the triggering turn's act-trail and iteration count from the"""
        from services.act_trail import ActTrail  # noqa: PLC0415
        trigger_channel = getattr(mp, "_trigger_channel", None)
        trigger_turn_id = getattr(mp, "_trigger_turn_id", None)
        rows: list[dict[str, object]] = []
        if trigger_channel is not None and trigger_turn_id is not None:
            rows = ActTrail().fetch_by_turn(trigger_channel, trigger_turn_id)
        iteration_count = len(rows)
        original_input = getattr(mp, "_raw_input", "") or ""
        parts = [
            f"## Original User Request\n{original_input}",
            f"\n## Completed ACT Loop Trail ({iteration_count} iterations)",
        ]
        for row in rows:
            parts.append(ActTrail.render(row))
        return "\n".join(parts)

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        """OLD assembly ``f"\n\n{body}"`` — this channel's get_user_definition
        is empty so the leading blank lines are reproduced for parity."""
        from services.system_message_prompt import SkillSuggestionSystemPrompt  # noqa: PLC0415
        return f"\n\n{SkillSuggestionSystemPrompt().get_prompt()}"

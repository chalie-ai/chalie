from __future__ import annotations

from typing import TYPE_CHECKING

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

_SYSTEM_PROMPT = """You map behavioral patterns to skill playbooks.

Given a list of the user's behavioral patterns and a list of available skills,
identify which patterns are relevant to which skills and produce a personalisation
rule for each match.

A personalisation rule is a single sentence describing how the skill should be
adapted based on the pattern. Only produce rules where the pattern genuinely
informs how the skill should be executed differently.

Respond with a JSON array of objects:
[{"skill_id": <int>, "pattern_name": "<str>", "rule": "<str>"}]

If no patterns match any skills, respond with an empty array: []"""


class SkillAssociationConfig(ProcessorConfig):
    """Own MP loop, no tools, no transcript."""

    def __init__(self) -> None:
        super().__init__(
            channel="skill_association",
            role="skill_association",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return ""

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        return mp._raw_input

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        return _SYSTEM_PROMPT

from __future__ import annotations

from services.processor_config import ProcessorConfig

_SYSTEM_PROMPT = (
    "Extract a handover summary of what is available in the user-message. "
    "Keep your response concise."
)


class TrailHandoverConfig(ProcessorConfig):
    """Act-trail handover compaction — bounded loop, no tools, no transcript. §3.5."""

    def __init__(self) -> None:
        super().__init__(
            channel="compaction",
            role="compaction",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=30,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        return _SYSTEM_PROMPT

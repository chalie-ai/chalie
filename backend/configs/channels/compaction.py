from __future__ import annotations

from services.processor_config import ProcessorConfig


class CompactionConfig(ProcessorConfig):
    """Continuity compaction — bounded loop, no tools, no transcript writes.  §3a."""

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
            post_turn=None,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        """System prompt for continuity (history) compaction.  §3a / §4a."""
        from services.system_message_prompt import ContinuityCompactionSystemPrompt  # noqa: PLC0415
        return ContinuityCompactionSystemPrompt().get_prompt()

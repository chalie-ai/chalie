from __future__ import annotations

from typing import TYPE_CHECKING

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor


class EpisodeEncoderConfig(ProcessorConfig):
    """Episode encoder — one-shot, no tools, no transcript writes.  §3a."""

    def __init__(self) -> None:
        super().__init__(
            channel="episode_encoder",
            role="episode_encoder",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return (
            "The user is 'episode_encoder' — a background process that "
            "summarises transcript windows into memory snapshots."
        )

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
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

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        from services.system_message_prompt import EpisodeEncoderSystemPrompt  # noqa: PLC0415
        return (
            f"{self.get_user_definition(mp)}\n\n"
            f"{EpisodeEncoderSystemPrompt().get_prompt()}"
        )

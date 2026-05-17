"""
EpisodeEncoderProcessor — one-shot internal processor for episodic memory encoding.

Reads a formatted transcript window (plus any referenced episodes) and asks
the LLM to return a JSON array of episode snapshots. The caller owns parsing
and storage — this processor returns the raw LLM text via send().

"""

from services.message_processor import MessageProcessor
from services.system_message_prompt import EpisodeEncoderSystemPrompt


class EpisodeEncoderProcessor(MessageProcessor):
    """Internal processor that encodes a transcript window into episode snapshots.

    One-shot: MAX_ITERATIONS=1, no tools, no transcript writes.
    The caller is responsible for parsing the returned JSON and persisting
    any resulting episodes via EpisodicService.

    Usage::

        response = EpisodeEncoderProcessor(window_str, referenced_str).send()
        snapshots = json.loads(response) or []
    """

    CHANNEL = 'episode_encoder'
    ROLE = 'episode_encoder'
    USAGE_CLASS = 'subconscious'
    JOB = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = EpisodeEncoderSystemPrompt
    ALWAYS_AVAILABLE: list[str] = []
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS = 1
    SKIP_TRANSCRIPT_WRITE = True

    def __init__(
        self,
        transcript_window: str,
        referenced_episodes: str,
        metadata: dict | None = None,
    ):
        super().__init__(raw_input='', metadata=metadata)
        self._window = transcript_window
        self._referenced = referenced_episodes

    def get_user_definition(self) -> str:
        return (
            "The user is 'episode_encoder' — a background process that "
            "summarises transcript windows into memory snapshots."
        )

    def get_user_prompt(self) -> str:
        parts = [
            "Transcript window — each line is `[id] (timestamp) role: content`:",
            "",
            self._window,
        ]
        if self._referenced:
            parts.extend([
                "",
                "Episodes referenced during these turns (candidates for update / delete):",
                "",
                self._referenced,
            ])
        return "\n".join(parts)

    def post_turn(self) -> None:
        """No post-turn fan-out — caller owns all downstream work."""
        pass

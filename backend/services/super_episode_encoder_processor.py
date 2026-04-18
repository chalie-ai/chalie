"""
SuperEpisodeEncoderProcessor — one-shot internal processor for super-episode synthesis.

Receives a list of apex episode dicts and the raw transcript spans that produced them,
asks the LLM to return a single consolidated JSON object summarising them all.

The caller owns all downstream work: parsing, embedding, salience, storage, and
setting consolidated_into back-pointers on source episodes.

North star: /Volumes/llm/chalie-plans/message-processing.md
Plan: /Volumes/llm/chalie-plans/v0.3.3/episodic-simplification.md § Commit C
"""

from services.message_processor import MessageProcessor
from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt


class SuperEpisodeEncoderProcessor(MessageProcessor):
    """Internal processor that synthesises a cluster of apex episodes into one super-episode.

    One-shot: MAX_ITERATIONS=1, no tools, no transcript writes.
    The caller is responsible for parsing the returned JSON and persisting
    the resulting super-episode via EpisodicService.

    Usage::

        response = SuperEpisodeEncoderProcessor(sources, spans).send()
        super_ep = json.loads(response)  # single dict
    """

    CHANNEL = 'super_episode_encoder'
    ROLE = 'super_episode_encoder'
    JOB = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = SuperEpisodeEncoderSystemPrompt
    NATIVE_TOOLS: list[str] = []
    MAX_ITERATIONS = 1
    MAX_TIMEOUT = 120  # seconds
    SKIP_TRANSCRIPT_WRITE = True

    def __init__(
        self,
        source_episodes: list[dict],
        transcript_spans: str,
        metadata: dict | None = None,
    ):
        super().__init__(raw_input='', metadata=metadata)
        self._sources = source_episodes
        self._spans = transcript_spans

    def getUserDefinition(self) -> str:
        return (
            "The user is 'super_episode_encoder' — a background process that "
            "consolidates clusters of related episodes into a single super-episode."
        )

    def getUserPrompt(self) -> str:
        src = "\n\n".join(f"[{e['id']}] {e['gist']}" for e in self._sources)
        return (
            f"Source episodes:\n\n{src}\n\n"
            f"Raw transcript spans covering these episodes:\n\n{self._spans}"
        )

    def postTurn(self) -> None:
        """No post-turn fan-out — caller owns all downstream work."""
        pass

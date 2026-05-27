"""
SkillSuggestionMessageProcessor — fully subconscious proactive skill creation.

When a user's ACT loop completes with 4+ tool-calling iterations, this processor
runs a background ACT loop with ``skill_manager`` available.  It analyses the
completed trail, eliminates dead ends, and calls ``skill_manager`` with
``action=create`` if the workflow is reusable.  No output is ever sent to the
user — the processor operates entirely in the background.

Entry point: ``maybe_suggest_skill(act_trail, raw_input)`` — non-blocking,
never raises.
"""

import logging
import threading

from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor
from services.system_message_prompt import SkillSuggestionSystemPrompt

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_SUGGEST]"

_SKILLS_DB_PATH = FileMapperService.get_skills_db_path()


class SkillSuggestionMessageProcessor(MessageProcessor):
    """Background processor that analyses completed ACT trails for reusable workflows.

    Runs a standard ACT loop with ``skill_manager`` as the sole tool.  The
    system prompt instructs the LLM to analyse the original trail, eliminate
    dead ends, and call ``skill_manager`` with ``action=create`` when the
    workflow is reusable.  If not reusable, the processor exits without
    calling any tools.

    No result is dispatched to the user — the processor is fully subconscious.
    """

    CHANNEL = 'skills_building'
    ROLE = 'skill_suggestion'
    USAGE_CLASS = 'subconscious'
    LOG_LABEL = 'skill_suggestion'
    SYSTEM_PROMPT_CLASS = SkillSuggestionSystemPrompt
    ALWAYS_AVAILABLE: list[str] = ['skill_manager']
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS = 5
    SKIP_TRANSCRIPT_WRITE = False

    def __init__(
        self,
        original_trail: list[str],
        original_input: str,
        iteration_count: int,
        metadata: dict | None = None,
    ):
        super().__init__(raw_input='', metadata=metadata)
        self._original_trail = original_trail
        self._original_input = original_input
        self._iteration_count = iteration_count

    def get_user_definition(self) -> str:
        """No user definition — internal background processor."""
        return ""

    def get_user_prompt(self) -> str:
        """Build the analysis prompt from the original request and ACT trail."""
        parts = [
            f"## Original User Request\n{self._original_input}",
            f"\n## Completed ACT Loop Trail ({self._iteration_count} iterations)",
        ]
        for entry in self._original_trail:
            parts.append(entry)

        trail = self.get_act_loop_trail()
        if trail:
            parts.append(f"\n{trail}")

        return '\n'.join(parts)

    def get_previous_messages(self, token_budget: int | None = None) -> str:
        """Each run is independent — no prior context injected."""
        return ''


def maybe_suggest_skill(act_trail: list[str], raw_input: str) -> None:
    """Fire background skill analysis for a completed ACT loop.

    Non-blocking.  Never raises.  Skips immediately when skills.sqlite is
    absent.  Logs the threshold-met event before spawning the daemon thread.
    """
    if not act_trail:
        return

    if not _SKILLS_DB_PATH.exists():
        return

    iteration_count = len(act_trail)
    logger.info(
        "%s threshold met — analysing (%d iterations)",
        _LOG_PREFIX,
        iteration_count,
    )

    t = threading.Thread(
        target=_run_suggestion_processor,
        args=(act_trail, raw_input, iteration_count),
        daemon=True,
        name="skill-suggest",
    )
    t.start()


def _run_suggestion_processor(
    act_trail: list[str],
    raw_input: str,
    iteration_count: int,
) -> None:
    """Thread target: instantiate and run the processor. Never raises."""
    try:
        processor = SkillSuggestionMessageProcessor(
            original_trail=act_trail,
            original_input=raw_input,
            iteration_count=iteration_count,
        )
        processor.send()
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)

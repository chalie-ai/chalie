"""
SkillSuggestionMessageProcessor — proactive skill creation after complex ACT loops.

When a user's ACT loop completes with 4+ tool-calling iterations, this processor
runs a background ACT loop with ``skill_builder`` available.  It analyses the
completed trail, eliminates dead ends, and calls ``skill_builder`` with
``action=create`` if the workflow is reusable.  The result is dispatched to the
user via a fresh UMP turn.

Entry point: ``maybe_suggest_skill(act_trail, raw_input)`` — non-blocking,
never raises.
"""

import logging
import threading
import time

from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor
from services.system_message_prompt import SkillSuggestionSystemPrompt

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_SUGGEST]"

_SKILLS_DB_PATH = FileMapperService.get_skills_db_path()

# Module-level cooldown: seconds between successive suggestions.
_COOLDOWN_SECONDS = 300

# Shared state — protected by GIL (single float write/read is atomic in CPython).
_last_suggested_at: float = 0.0


class SkillSuggestionMessageProcessor(MessageProcessor):
    """Background processor that analyses completed ACT trails for reusable workflows.

    Runs a standard ACT loop with ``skill_builder`` as the sole tool.  The
    system prompt instructs the LLM to analyse the original trail, eliminate
    dead ends, and call ``skill_builder`` with ``action=create`` when the
    workflow is reusable.  If not reusable, the processor exits without
    calling any tools.

    ``post_turn()`` dispatches the result to the user only when
    ``skill_builder`` was actually invoked (i.e., a skill was created).
    """

    CHANNEL = 'skill_suggestion'
    ROLE = 'skill_suggestion'
    USAGE_CLASS = 'subconscious'
    LOG_LABEL = 'skill_suggestion'
    SYSTEM_PROMPT_CLASS = SkillSuggestionSystemPrompt
    ALWAYS_AVAILABLE: list[str] = ['skill_builder']
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS = 5
    SKIP_TRANSCRIPT_WRITE = True

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
        self._final_response: str = ''

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

        # The processor's own ACT trail (skill_builder calls, if any).
        trail = self.get_act_loop_trail()
        if trail:
            parts.append(f"\n{trail}")

        return '\n'.join(parts)

    def store(self, llm_response: str) -> None:
        """Capture the final response for post_turn dispatch."""
        self._final_response = llm_response or ''
        super().store(llm_response)

    def post_turn(self) -> None:
        """Dispatch result to the user if skill_builder was called."""
        skill_created = any(
            '[skill_builder(' in entry for entry in self._act_trail
        )
        if not skill_created:
            logger.info("%s verdict: SKIP (skill_builder not called)", _LOG_PREFIX)
            return

        if not self._final_response:
            return

        global _last_suggested_at
        _last_suggested_at = time.monotonic()

        dispatch_input = (
            f"You noticed the user's last task involved {self._iteration_count} "
            f"tool-calling steps. After analysing the workflow, you created a "
            f"reusable skill from it. Here is your analysis:\n\n"
            f"{self._final_response}\n\n"
            f"Let the user know you saved this skill and briefly explain when "
            f"it will be useful."
        )

        from api.chat import dispatch_message
        dispatch_message(
            dispatch_input,
            source='skill_suggestion',
            hidden_input=True,
        )
        logger.info("%s dispatched", _LOG_PREFIX)


def maybe_suggest_skill(act_trail: list[str], raw_input: str) -> None:
    """Fire background skill analysis for a completed ACT loop.

    Non-blocking.  Never raises.  Skips immediately when cooldown is active
    or skills.sqlite is absent.  Logs the threshold-met event before spawning
    the daemon thread.
    """
    if not act_trail:
        return

    if not _SKILLS_DB_PATH.exists():
        return

    global _last_suggested_at
    now = time.monotonic()
    if now - _last_suggested_at < _COOLDOWN_SECONDS:
        logger.info("%s cooldown active — skipping", _LOG_PREFIX)
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

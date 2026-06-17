import logging
import threading

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_SUGGEST]"

_SKILLS_DB_PATH = FileMapperService.get_skills_db_path()


def maybe_suggest_skill(act_trail: list[str], raw_input: str) -> None:
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
    try:
        from configs.channels import SkillSuggestionConfig
        from services.message_processor import MessageProcessor

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "", None)
        mp.config = SkillSuggestionConfig()
        mp.uid = None
        mp.current_iteration = 0
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp._original_trail = act_trail
        mp._original_input = raw_input
        mp._iteration_count = iteration_count
        mp._run()
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)

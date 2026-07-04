import logging
import threading
from typing import TYPE_CHECKING, cast

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from typing import Protocol

    class _TriggerCtx(Protocol):
        _trigger_channel: str
        _trigger_turn_id: int


logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_SUGGEST]"

_SKILLS_DB_PATH = FileMapperService.get_skills_db_path()


def maybe_suggest_skill(
    act_trail: list[str],
    raw_input: str,
    trigger_channel: str,
    trigger_turn_id: int,
) -> None:
    if not act_trail:
        return

    if not _SKILLS_DB_PATH.exists():
        return

    logger.info(
        "%s threshold met — analysing (%d iterations)",
        _LOG_PREFIX,
        len(act_trail),
    )

    t = threading.Thread(
        target=_run_suggestion_processor,
        args=(raw_input, trigger_channel, trigger_turn_id),
        daemon=True,
        name="skill-suggest",
    )
    t.start()


def _run_suggestion_processor(
    raw_input: str,
    trigger_channel: str,
    trigger_turn_id: int,
) -> None:
    try:
        from configs.channels import SkillSuggestionConfig
        from services.message_processor import MessageProcessor

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, SkillSuggestionConfig(), raw_input=raw_input)
        mp.uid = None
        mp.thinking_level = "low"
        # Trigger context — the user turn that fired this suggestion pass.
        # get_user_prompt reads these to derive the real act-trail and query
        # from the DB instead of the suggestion MP's own (empty) turn.
        cast("_TriggerCtx", mp)._trigger_channel = trigger_channel
        cast("_TriggerCtx", mp)._trigger_turn_id = trigger_turn_id
        mp._run()
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)

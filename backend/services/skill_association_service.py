"""
SkillAssociationService — Layer 2 of the Self-Refining Skill Library.

Maps the user's active behavioural patterns to curated skill playbooks and
writes personalisation rules into skill_associations in skills.sqlite.

Called by SubconsciousWorker after every PatternMatchProcessor pass.
"""

import json
import logging
from typing import cast

from models.behavioral_pattern import BehavioralPattern
from models.skill import Skill
from models.skill_association import SkillAssociation
from services.database import Database
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SKILL_ASSOC]"

_SKILLS_DB = FileMapperService.get_skills_db_path()

class SkillAssociationService:
    """Run LLM-driven association passes between behavioural patterns and skills.

    Writes personalisation rules to skill_associations in skills.sqlite.
    """

    def run_pass(self, touched_pattern_ids: set[int]) -> int:
        if not _SKILLS_DB.exists():
            logger.info(f"{LOG_PREFIX} skills.sqlite not found — skipping")
            return 0

        if not touched_pattern_ids:
            logger.info(f"{LOG_PREFIX} no touched patterns — skipping")
            return 0

        patterns = self._load_patterns(touched_pattern_ids)
        if not patterns:
            logger.info(f"{LOG_PREFIX} no patterns found for touched IDs — skipping")
            return 0

        skills = self._load_skill_index()
        if not skills:
            logger.info(f"{LOG_PREFIX} no skills in DB — skipping")
            return 0

        associations = self._request_associations(patterns, skills)
        if associations is None:
            return 0

        valid_skill_ids = {s[0] for s in skills}
        pattern_names = {p[0] for p in patterns}
        written = self._write_associations(associations, valid_skill_ids, pattern_names)

        logger.info(f"{LOG_PREFIX} wrote {written} associations")
        return written

    def _load_patterns(self, row_ids: set[int]) -> list[tuple[str, str]]:
        rows = BehavioralPattern.live().filter_in("id", sorted(row_ids)).get()
        return cast(list[tuple[str, str]], [(row.key, row.value) for row in rows])

    def _load_skill_index(self) -> list[tuple[str, str, str]]:
        rows = Skill.all().get()
        return cast(
            list[tuple[str, str, str]],
            [(row.id, row.title, row.use_for) for row in rows],
        )

    def _request_associations(
        self,
        patterns: list[tuple[str, str]],
        skills: list[tuple[str, str, str]],
    ) -> list[dict[str, object]] | None:
        pattern_list = [
            {key: json.loads(value).get("summary", value) if value else value}
            for key, value in patterns
        ]
        skill_list = [
            {"id": sid, "title": title, "use_for": use_for}
            for sid, title, use_for in skills
        ]
        user_prompt = (
            f"## Behavioral Patterns\n{json.dumps(pattern_list)}\n\n"
            f"## Available Skills\n{json.dumps(skill_list)}"
        )

        from controllers.message_processor import MessageProcessor
        from configs.channels import SkillAssociationConfig
        try:
            text = MessageProcessor.process(SkillAssociationConfig(), raw_input=user_prompt).result()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "context" in exc_str or "token" in exc_str or "length" in exc_str:
                logger.error(
                    f"{LOG_PREFIX} prompt exceeds provider context window — "
                    f"patterns={len(patterns)} skills={len(skills)}: {exc}"
                )
            else:
                logger.error(f"{LOG_PREFIX} LLM call failed: {exc}")
            return None

        return _parse_associations(text)

    def _write_associations(
        self,
        associations: list[dict[str, object]],
        valid_skill_ids: set[str],
        pattern_names: set[str],
    ) -> int:
        written = 0

        with Database.transaction(str(_SKILLS_DB)):
            for assoc in associations:
                sid = assoc.get("skill_id")
                pname = assoc.get("pattern_name")
                rule = assoc.get("rule")
                if not (sid and pname and rule):
                    continue
                if sid not in valid_skill_ids:
                    continue
                if pname not in pattern_names:
                    continue
                SkillAssociation.upsert(cast(int, sid), pname, cast(str, rule))
                written += 1

        return written


def _parse_associations(text: str) -> list[dict[str, object]] | None:
    if not text:
        return None
    try:
        payload = text.strip()
        if '```' in payload:
            parts = payload.split('```')
            if len(parts) >= 3:
                payload = parts[1]
            else:
                payload = parts[1]
            if payload.startswith('json'):
                payload = payload[4:]
        result = json.loads(payload.strip())
    except (json.JSONDecodeError, IndexError) as exc:
        logger.warning(f"{LOG_PREFIX} failed to parse LLM response: {exc}")
        return None

    if not isinstance(result, list):
        logger.warning(f"{LOG_PREFIX} LLM returned non-list: {type(result)}")
        return None

    return cast(list[dict[str, object]], result)

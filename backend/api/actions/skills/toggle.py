"""Skill toggle action — nested resource under /api/skills/toggle.

Covers:
- POST /api/skills/toggle/<id>  → post (flip enabled)
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import NotFoundError
from api.request import Request
from api.response.skills import SkillToggleResponse
from models.skill import Skill as SkillModel
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class SkillToggle(Action):
    """Action for flipping a skill's enabled state."""

    id_type: ClassVar[type[int] | type[str]] = int

    def slug(self) -> str:
        return "skills"

    def verb(self) -> str:
        return "toggle"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            return self.not_allowed()

        if not FileMapperService.get_skills_db_path().exists():
            return SkillToggleResponse.failure("skills database unavailable"), 503

        skill_id = cast(int, id)
        skill = SkillModel.filter("id", skill_id).first()
        if skill is None:
            raise NotFoundError("Skill not found")

        new_enabled = 0 if skill.enabled else 1
        skill.enabled = new_enabled
        skill.save()

        logger.info("[SKILLS API] Toggled skill id=%d enabled=%d", skill_id, new_enabled)
        return SkillToggleResponse(skill_id=skill_id, enabled=bool(new_enabled)).single()

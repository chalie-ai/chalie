"""Skill copy action — nested resource under /api/skills/copy.

Covers:
- POST /api/skills/copy/<id>  → post (copy a curated skill into a user skill)

Copying disables the original curated skill (source="curated") and indexes
the new user copy, mirroring the legacy handler exactly.
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, NotFoundError
from api.endpoints.skills import _index_new_skill
from api.request import Request
from api.response.skills import SkillResponse
from models.skill import Skill as SkillModel
from services.database import Database
from services.file_mapper_service import FileMapperService
from utils.skills_io import DEFAULT_VERSION, ensure_user_skills_dir, skill_yaml_path, write_skill_file

logger = logging.getLogger(__name__)


class SkillCopy(Action):
    """Action for copying a curated skill into an editable user skill."""

    id_type: ClassVar[type[int] | type[str]] = int
    # Unlike most Action verbs this one creates a resource (the user copy),
    # so its POST genuinely returns 201.
    _post_may_create: ClassVar[bool] = True
    response_dto = {
        "post": DocumentedResponse(
            SkillResponse,
            extras=(
                (409, "A user copy of this skill already exists"),
                (422, "Only curated skills can be copied"),
                (503, "Skills database unavailable"),
            ),
        )
    }

    def slug(self) -> str:
        return "skills"

    def verb(self) -> str:
        return "copy"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            return self.not_allowed()

        if not FileMapperService.get_skills_db_path().exists():
            return SkillResponse.failure("skills database unavailable"), 503

        skill_id = cast(int, id)
        skill = SkillModel.filter("id", skill_id).first()
        if skill is None:
            raise NotFoundError("Skill not found")
        if skill.source != "curated":
            return SkillResponse.failure("Only curated skills can be copied"), 422

        base_title = skill.title
        copy_title = f"{base_title} (Custom)"
        tags = skill.tags or ""

        if SkillModel.find_by_title_ci(copy_title) is not None:
            return SkillResponse.failure(f"A user copy named '{copy_title}' already exists"), 409

        with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
            copy = SkillModel(
                title=copy_title,
                use_for=skill.use_for,
                content=skill.content,
                tags=tags,
                version=DEFAULT_VERSION,
                source="user",
                based_on=skill_id,
            ).save()
            new_id = cast(int, copy.id)

            skill.enabled = 0
            skill.save()

            _index_new_skill(conn, new_id, copy_title, skill.use_for, tags)

        ensure_user_skills_dir()
        write_skill_file(
            skill_yaml_path(copy_title),
            {
                "title": copy_title,
                "use_for": skill.use_for,
                "content": skill.content,
                "tags": tags,
                "version": DEFAULT_VERSION,
            },
        )

        logger.info(
            "[SKILLS API] Copied curated skill id=%d '%s' -> user skill id=%d '%s'",
            skill_id, base_title, new_id, copy_title,
        )
        return self._to_response(copy).single(), 201

    def _to_response(self, skill: SkillModel) -> SkillResponse:
        return SkillResponse(
            id=cast(int, skill.id),
            title=skill.title,
            use_for=skill.use_for,
            content=skill.content,
            tags=skill.tags or "",
            version=skill.version,
            source=skill.source,
            enabled=bool(skill.enabled),
            based_on=skill.based_on,
        )

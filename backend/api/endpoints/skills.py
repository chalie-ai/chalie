"""Skills endpoint — migrated from the legacy skills namespace.

Covers the CRUD routes over skills.sqlite (via the
:class:`~models.skill.Skill` active-record model) plus user-skill YAML
write-back to ``data/skills/user/``:

- GET    /api/skills/all   → get_all (listing only — associations moved to
                              the ``skills/associations`` action)
- POST   /api/skills/<id>  → post (create when id=-1, update otherwise)
- DELETE /api/skills/<id>  → delete

The embedding search index (``skill_search_entries``/``skill_search_vec``/
``skill_search_fts``) is written through ``utils.build_skills_db.index_skill``
/ ``utils.skills_io.remove_search_entries`` on the same raw connection the
model write lands on, inside one ``Database.transaction`` block, so a skill
row and its index entries commit atomically.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import EndpointError, ForbiddenError, NotFoundError
from api.request import Request
from api.request.skills import SkillRequest
from api.response.skills import SkillResponse
from models.skill import Skill as SkillModel
from services.database import Database
from services.file_mapper_service import FileMapperService
from utils.skills_io import (
    DEFAULT_VERSION,
    ensure_user_skills_dir,
    remove_search_entries,
    skill_yaml_path,
    write_skill_file,
)

logger = logging.getLogger(__name__)


def _index_new_skill(conn: sqlite3.Connection, skill_id: int, title: str, use_for: str, tags: str) -> None:
    try:
        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, use_for, tags)
    except Exception as exc:
        logger.error("[SKILLS API] Failed to index skill %d: %s", skill_id, exc)
        raise


class Skills(Endpoint):
    """CRUD endpoint for skills."""

    id_type: ClassVar[type[int] | type[str]] = int
    request_dto: ClassVar[type[Request] | None] = SkillRequest
    response_dto = {
        "get_all": DocumentedResponse(SkillResponse, listing=True),
        "post": DocumentedResponse(
            SkillResponse,
            extras=(
                (409, "A user skill with this title already exists"),
                (503, "Skills database unavailable"),
            ),
        ),
        "delete": DocumentedResponse(extras=((503, "Skills database unavailable"),)),
    }

    def slug(self) -> str:
        return "skills"

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return SkillResponse.listing([], page, limit, 0)

        rows = SkillModel.order_by("source, title").get()
        items = [self._to_response(row) for row in rows]
        total = len(items)
        start = (page - 1) * limit
        return SkillResponse.listing(items[start : start + limit], page, limit, total)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        dto = cast(SkillRequest, data)
        if self.is_create(id):
            return self._create(dto)
        return self._update(cast(int, id), dto)

    def delete(self, id: int | str) -> ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return SkillResponse.failure("skills database unavailable"), 503

        skill_id = cast(int, id)
        skill = SkillModel.filter("id", skill_id).first()
        if skill is None:
            raise NotFoundError("Skill not found")
        if skill.source != "user":
            raise ForbiddenError("Only user-created skills can be deleted")

        title = skill.title
        path = skill_yaml_path(title)

        with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
            remove_search_entries(conn, skill_id)
            skill.delete()

        if path.exists() and path.resolve().is_relative_to(FileMapperService.get_user_skills_path().resolve()):
            path.unlink()

        logger.info("[SKILLS API] Deleted skill id=%d '%s'", skill_id, title)
        return "", 204

    def _create(self, dto: SkillRequest) -> ResponseReturnValue:
        if dto.title is None:
            raise EndpointError("title is required")
        if dto.use_for is None:
            raise EndpointError("use_for is required")
        if dto.content is None:
            raise EndpointError("content is required")

        if not FileMapperService.get_skills_db_path().exists():
            return SkillResponse.failure("skills database unavailable"), 503

        if SkillModel.find_by_title_ci(dto.title) is not None:
            return SkillResponse.failure(f"A user skill named '{dto.title}' already exists"), 409

        title = dto.title
        use_for = dto.use_for
        content = dto.content
        tags = dto.tags if dto.tags is not None else ""

        with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
            skill = SkillModel(
                title=title,
                use_for=use_for,
                content=content,
                tags=tags,
                version=DEFAULT_VERSION,
                source="user",
            ).save()
            skill_id = cast(int, skill.id)

            _index_new_skill(conn, skill_id, title, use_for, tags)

        ensure_user_skills_dir()
        write_skill_file(
            skill_yaml_path(title),
            {
                "title": title,
                "use_for": use_for,
                "content": content,
                "tags": tags,
                "version": DEFAULT_VERSION,
            },
        )

        logger.info("[SKILLS API] Created skill '%s' (id=%d)", title, skill_id)
        return self._to_response(skill).single(), 201

    def _update(self, skill_id: int, dto: SkillRequest) -> ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return SkillResponse.failure("skills database unavailable"), 503

        skill = SkillModel.filter("id", skill_id).first()
        if skill is None:
            raise NotFoundError("Skill not found")
        if skill.source != "user":
            raise ForbiddenError("Only user-created skills can be edited")

        title = skill.title
        use_for = dto.use_for if dto.use_for is not None else skill.use_for
        content = dto.content if dto.content is not None else skill.content
        tags = dto.tags if dto.tags is not None else (skill.tags or "")
        version = skill.version + 1

        with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
            skill.use_for = use_for
            skill.content = content
            skill.tags = tags
            skill.version = version
            skill.save()

            remove_search_entries(conn, skill_id)
            _index_new_skill(conn, skill_id, title, use_for, tags)

        ensure_user_skills_dir()
        write_skill_file(
            skill_yaml_path(title),
            {
                "title": title,
                "use_for": use_for,
                "content": content,
                "tags": tags,
                "version": version,
            },
        )

        logger.info("[SKILLS API] Updated skill id=%d '%s' to v%d", skill_id, title, version)
        return self._to_response(skill).single()

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

"""Skills namespace — Brain Skills tab API.

CRUD over skills.sqlite (via the :class:`~models.skill.Skill` /
:class:`~models.skill_association.SkillAssociation` active-record models)
plus user-skill YAML write-back to data/skills/user/. DTO-typed through the
foundation boundary decorators (``@expects``/``@responds``) following the
lists reference shape.

The embedding search index (``skill_search_entries``/``skill_search_vec``/
``skill_search_fts``) is out of scope for the two models above — it is
written through ``utils.build_skills_db.index_skill`` /
``utils.skills_io.remove_search_entries`` on a raw connection, obtained the
same way the models reach skills.sqlite (``Database.conn`` on the skills db
path), grouped with the model write in one ``Database.transaction`` block so
a skill row and its index entries commit atomically.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from models.skill import Skill as SkillModel
from models.skill_association import SkillAssociation as SkillAssociationModel
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.time_utils import parse_utc
from utils.skills_io import (
    DEFAULT_VERSION,
    ensure_user_skills_dir,
    remove_search_entries,
    skill_yaml_path,
    write_skill_file,
)
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.skill import Skill, SkillCreate, SkillUpdate
from .dto.skill_association import SkillAssociation
from .dto.skill_list import SkillListResult
from .dto.skill_toggle import SkillToggleResult

logger = logging.getLogger(__name__)

skills_ns = Namespace("skills", description="Brain Skills tab API", path="/api/skills")

register_dto(
    skills_ns,
    Skill,
    SkillCreate,
    SkillUpdate,
    SkillAssociation,
    SkillListResult,
    SkillToggleResult,
    Error,
)

_S = skills_ns.models

_NOT_FOUND = "Skill not found"
_DB_UNAVAILABLE = "skills database unavailable"
_ONLY_USER_EDIT = "Only user-created skills can be edited"
_ONLY_USER_DELETE = "Only user-created skills can be deleted"
_ONLY_CURATED_COPY = "Only curated skills can be copied"

# Write handlers open ``Database.transaction`` on the same path the models bind
# ``skills.sqlite`` writes to (``Skill._bound_connection`` /
# ``SkillAssociation._bound_connection``) so a model write and a raw
# search-index write land on the same connection and commit atomically.


def _row_to_dict(skill: SkillModel) -> dict[str, object]:
    return {
        "id": skill.id,
        "title": skill.title,
        "use_for": skill.use_for,
        "content": skill.content,
        "tags": skill.tags or "",
        "version": skill.version,
        "source": skill.source,
        "enabled": bool(skill.enabled),
        "based_on": skill.based_on,
    }


def _load_associations() -> list[dict[str, object]]:
    rows = SkillAssociationModel.with_skill_titles()
    return [
        {
            "skill_id": r["skill_id"],
            "skill_title": r["skill_title"],
            "pattern_name": r["pattern_name"],
            "rule": r["rule"],
            "created_at": parse_utc(cast(str, r["created_at"])),
        }
        for r in rows
    ]


def _index_new_skill(conn: sqlite3.Connection, skill_id: int, title: str, use_for: str, tags: str) -> None:
    try:
        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, use_for, tags)
    except Exception as exc:
        logger.error("[SKILLS API] Failed to index skill %d: %s", skill_id, exc)
        raise


@skills_ns.route("")
class SkillsListResource(Resource):
    @require_session
    @skills_ns.response(200, "All skills + associations", model=_S["SkillListResult"])
    @skills_ns.response(500, "Failed to load skills", model=_S["Error"])
    @responds(SkillListResult, code=200)
    def get(self) -> SkillListResult | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return SkillListResult(skills=[], associations=[])

        try:
            rows = SkillModel.order_by("source, title").get()
            skills = [Skill.model_validate(_row_to_dict(r)) for r in rows]
            associations = [SkillAssociation.model_validate(a) for a in _load_associations()]

            return SkillListResult(skills=skills, associations=associations)
        except Exception as exc:
            logger.error("[SKILLS API] GET /api/skills failed: %s", exc)
            return error("Failed to load skills", 500)

    @require_session
    @skills_ns.expect(_S["SkillCreate"])
    @skills_ns.response(201, "Created", model=_S["Skill"])
    @skills_ns.response(409, "A user skill with this title already exists", model=_S["Error"])
    @skills_ns.response(422, "Validation failed", model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to create skill", model=_S["Error"])
    @responds(Skill, code=201)
    @expects(SkillCreate)
    def post(self, dto: SkillCreate) -> Skill | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return error(_DB_UNAVAILABLE, 503)

        try:
            if SkillModel.find_by_title_ci(dto.title) is not None:
                return error(f"A user skill named '{dto.title}' already exists", 409)

            with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
                skill = SkillModel(
                    title=dto.title,
                    use_for=dto.use_for,
                    content=dto.content,
                    tags=dto.tags,
                    version=DEFAULT_VERSION,
                    source="user",
                ).save()
                skill_id = cast(int, skill.id)

                _index_new_skill(conn, skill_id, dto.title, dto.use_for, dto.tags)

            ensure_user_skills_dir()
            write_skill_file(
                skill_yaml_path(dto.title),
                {
                    "title": dto.title,
                    "use_for": dto.use_for,
                    "content": dto.content,
                    "tags": dto.tags,
                    "version": DEFAULT_VERSION,
                },
            )

            logger.info("[SKILLS API] Created skill '%s' (id=%d)", dto.title, skill_id)
            return Skill.model_validate(_row_to_dict(skill))
        except Exception as exc:
            logger.error("[SKILLS API] POST /api/skills failed: %s", exc)
            return error("Failed to create skill", 500)


@skills_ns.route("/<int:skill_id>")
class SkillItemResource(Resource):
    @require_session
    @skills_ns.param("skill_id", "Skill id")
    @skills_ns.expect(_S["SkillUpdate"])
    @skills_ns.response(200, "Updated skill", model=_S["Skill"])
    @skills_ns.response(403, _ONLY_USER_EDIT, model=_S["Error"])
    @skills_ns.response(404, _NOT_FOUND, model=_S["Error"])
    @skills_ns.response(422, "Validation failed", model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to update skill", model=_S["Error"])
    @responds(Skill, code=200)
    @expects(SkillUpdate)
    def put(self, skill_id: int, dto: SkillUpdate) -> Skill | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return error(_DB_UNAVAILABLE, 503)

        try:
            skill = SkillModel.filter("id", skill_id).first()
            if skill is None:
                return error(_NOT_FOUND, 404)
            if skill.source != "user":
                return error(_ONLY_USER_EDIT, 403)

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
            return Skill.model_validate(_row_to_dict(skill))
        except Exception as exc:
            logger.error("[SKILLS API] PUT /api/skills/%d failed: %s", skill_id, exc)
            return error("Failed to update skill", 500)

    @require_session
    @skills_ns.param("skill_id", "Skill id")
    @skills_ns.response(204, "Deleted")
    @skills_ns.response(403, _ONLY_USER_DELETE, model=_S["Error"])
    @skills_ns.response(404, _NOT_FOUND, model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to delete skill", model=_S["Error"])
    @responds(code=204)
    def delete(self, skill_id: int) -> None | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return error(_DB_UNAVAILABLE, 503)

        try:
            skill = SkillModel.filter("id", skill_id).first()
            if skill is None:
                return error(_NOT_FOUND, 404)
            if skill.source != "user":
                return error(_ONLY_USER_DELETE, 403)

            title = skill.title
            path = skill_yaml_path(title)

            with Database.transaction(str(FileMapperService.get_skills_db_path())) as conn:
                remove_search_entries(conn, skill_id)
                skill.delete()

            if path.exists() and path.resolve().is_relative_to(FileMapperService.get_user_skills_path().resolve()):
                path.unlink()

            logger.info("[SKILLS API] Deleted skill id=%d '%s'", skill_id, title)
            return None
        except Exception as exc:
            logger.error("[SKILLS API] DELETE /api/skills/%d failed: %s", skill_id, exc)
            return error("Failed to delete skill", 500)


@skills_ns.route("/<int:skill_id>/toggle")
class SkillToggleResource(Resource):
    @require_session
    @skills_ns.param("skill_id", "Skill id")
    @skills_ns.response(200, "Toggled", model=_S["SkillToggleResult"])
    @skills_ns.response(404, _NOT_FOUND, model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to toggle skill", model=_S["Error"])
    @responds(SkillToggleResult, code=200)
    def put(self, skill_id: int) -> SkillToggleResult | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return error(_DB_UNAVAILABLE, 503)

        try:
            skill = SkillModel.filter("id", skill_id).first()
            if skill is None:
                return error(_NOT_FOUND, 404)

            new_enabled = 0 if skill.enabled else 1
            skill.enabled = new_enabled
            skill.save()

            logger.info("[SKILLS API] Toggled skill id=%d enabled=%d", skill_id, new_enabled)
            return SkillToggleResult(skill_id=skill_id, enabled=bool(new_enabled))
        except Exception as exc:
            logger.error("[SKILLS API] PUT /api/skills/%d/toggle failed: %s", skill_id, exc)
            return error("Failed to toggle skill", 500)


@skills_ns.route("/<int:skill_id>/copy")
class SkillCopyResource(Resource):
    @require_session
    @skills_ns.param("skill_id", "Skill id")
    @skills_ns.response(201, "Copied", model=_S["Skill"])
    @skills_ns.response(404, _NOT_FOUND, model=_S["Error"])
    @skills_ns.response(409, "A user copy with this title already exists", model=_S["Error"])
    @skills_ns.response(422, _ONLY_CURATED_COPY, model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to copy skill", model=_S["Error"])
    @responds(Skill, code=201)
    def post(self, skill_id: int) -> Skill | ResponseReturnValue:
        if not FileMapperService.get_skills_db_path().exists():
            return error(_DB_UNAVAILABLE, 503)

        try:
            skill = SkillModel.filter("id", skill_id).first()
            if skill is None:
                return error(_NOT_FOUND, 404)
            if skill.source != "curated":
                return error(_ONLY_CURATED_COPY, 422)

            base_title = skill.title
            copy_title = f"{base_title} (Custom)"
            tags = skill.tags or ""

            if SkillModel.find_by_title_ci(copy_title) is not None:
                return error(f"A user copy named '{copy_title}' already exists", 409)

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
            return Skill.model_validate(_row_to_dict(copy))
        except Exception as exc:
            logger.error("[SKILLS API] POST /api/skills/%d/copy failed: %s", skill_id, exc)
            return error("Failed to copy skill", 500)

"""Skills namespace — Brain Skills tab API.

CRUD over skills.sqlite plus user-skill YAML write-back to data/skills/user/.
DTO-typed through the foundation boundary decorators (``@expects``/``@responds``)
following the lists reference shape.
"""

from __future__ import annotations

import logging
import sqlite3

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.time_utils import parse_utc

from utils.skills_io import (
    DEFAULT_VERSION,
    SKILLS_DB_PATH,
    USER_SKILLS_DIR,
    ensure_user_skills_dir,
    open_skills_db,
    remove_search_entries,
    skill_yaml_path,
    write_skill_file,
)

from .auth import require_session
from .dto import Error, expects, register_dto, responds
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


def _open_db() -> sqlite3.Connection:
    """Open skills.sqlite with row_factory for API dict conversion."""
    return open_skills_db(row_factory=True)


def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "title": row["title"],
        "use_for": row["use_for"],
        "content": row["content"],
        "tags": row["tags"] or "",
        "version": row["version"],
        "source": row["source"],
        "enabled": bool(row["enabled"]),
        "based_on": row["based_on"],
    }


def _load_associations(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        "SELECT sa.skill_id, s.title AS skill_title, sa.pattern_name, sa.rule, sa.created_at "
        "FROM skill_associations sa "
        "JOIN skills s ON s.id = sa.skill_id "
        "ORDER BY sa.created_at DESC"
    ).fetchall()
    return [
        {
            "skill_id": r["skill_id"],
            "skill_title": r["skill_title"],
            "pattern_name": r["pattern_name"],
            "rule": r["rule"],
            "created_at": parse_utc(r["created_at"]),
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


def _error(message: str, status: int) -> ResponseReturnValue:
    """Build a uniform non-2xx ``Error`` body carrying its own status code."""
    return Error(error=message).model_dump(mode="json"), status


@skills_ns.route("")
class SkillsListResource(Resource):
    @require_session
    @skills_ns.response(200, "All skills + associations", model=_S["SkillListResult"])
    @skills_ns.response(500, "Failed to load skills", model=_S["Error"])
    @responds(SkillListResult, code=200)
    def get(self) -> SkillListResult | ResponseReturnValue:
        if not SKILLS_DB_PATH.exists():
            return SkillListResult(skills=[], associations=[])

        try:
            conn = _open_db()
            try:
                rows = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, "
                    "source, enabled, based_on "
                    "FROM skills ORDER BY source, title"
                ).fetchall()
                skills = [Skill.model_validate(_row_to_dict(r)) for r in rows]
                associations = [SkillAssociation.model_validate(a) for a in _load_associations(conn)]
            finally:
                conn.close()

            return SkillListResult(skills=skills, associations=associations)
        except Exception as exc:
            logger.error("[SKILLS API] GET /api/skills failed: %s", exc)
            return _error("Failed to load skills", 500)

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
        if not SKILLS_DB_PATH.exists():
            return _error(_DB_UNAVAILABLE, 503)

        try:
            conn = _open_db()
            try:
                existing = conn.execute(
                    "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                    (dto.title,),
                ).fetchone()
                if existing is not None:
                    return _error(f"A user skill named '{dto.title}' already exists", 409)

                conn.execute(
                    "INSERT INTO skills(title, use_for, content, tags, version, source) "
                    "VALUES (?, ?, ?, ?, ?, 'user')",
                    (dto.title, dto.use_for, dto.content, dto.tags, DEFAULT_VERSION),
                )
                skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                _index_new_skill(conn, skill_id, dto.title, dto.use_for, dto.tags)
                conn.commit()

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

                row = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, "
                    "source, enabled, based_on FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
            finally:
                conn.close()

            logger.info("[SKILLS API] Created skill '%s' (id=%d)", dto.title, skill_id)
            return Skill.model_validate(_row_to_dict(row))
        except Exception as exc:
            logger.error("[SKILLS API] POST /api/skills failed: %s", exc)
            return _error("Failed to create skill", 500)


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
        if not SKILLS_DB_PATH.exists():
            return _error(_DB_UNAVAILABLE, 503)

        try:
            conn = _open_db()
            try:
                row = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, "
                    "source, enabled, based_on FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
                if row is None:
                    return _error(_NOT_FOUND, 404)
                if row["source"] != "user":
                    return _error(_ONLY_USER_EDIT, 403)

                title = row["title"]
                use_for = dto.use_for if dto.use_for is not None else row["use_for"]
                content = dto.content if dto.content is not None else row["content"]
                tags = dto.tags if dto.tags is not None else (row["tags"] or "")
                version = row["version"] + 1

                conn.execute(
                    "UPDATE skills SET use_for=?, content=?, tags=?, version=? "
                    "WHERE id=?",
                    (use_for, content, tags, version, skill_id),
                )

                remove_search_entries(conn, skill_id)
                _index_new_skill(conn, skill_id, title, use_for, tags)
                conn.commit()

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

                row = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, "
                    "source, enabled, based_on FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
            finally:
                conn.close()

            logger.info("[SKILLS API] Updated skill id=%d '%s' to v%d", skill_id, title, version)
            return Skill.model_validate(_row_to_dict(row))
        except Exception as exc:
            logger.error("[SKILLS API] PUT /api/skills/%d failed: %s", skill_id, exc)
            return _error("Failed to update skill", 500)

    @require_session
    @skills_ns.param("skill_id", "Skill id")
    @skills_ns.response(204, "Deleted")
    @skills_ns.response(403, _ONLY_USER_DELETE, model=_S["Error"])
    @skills_ns.response(404, _NOT_FOUND, model=_S["Error"])
    @skills_ns.response(503, _DB_UNAVAILABLE, model=_S["Error"])
    @skills_ns.response(500, "Failed to delete skill", model=_S["Error"])
    @responds(code=204)
    def delete(self, skill_id: int) -> None | ResponseReturnValue:
        if not SKILLS_DB_PATH.exists():
            return _error(_DB_UNAVAILABLE, 503)

        try:
            conn = _open_db()
            try:
                row = conn.execute(
                    "SELECT id, title, source FROM skills WHERE id = ?", (skill_id,)
                ).fetchone()
                if row is None:
                    return _error(_NOT_FOUND, 404)
                if row["source"] != "user":
                    return _error(_ONLY_USER_DELETE, 403)

                title = row["title"]
                path = skill_yaml_path(title)

                remove_search_entries(conn, skill_id)
                conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
                conn.commit()

                if path.exists() and path.resolve().is_relative_to(USER_SKILLS_DIR.resolve()):
                    path.unlink()
            finally:
                conn.close()

            logger.info("[SKILLS API] Deleted skill id=%d '%s'", skill_id, title)
            return None
        except Exception as exc:
            logger.error("[SKILLS API] DELETE /api/skills/%d failed: %s", skill_id, exc)
            return _error("Failed to delete skill", 500)


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
        if not SKILLS_DB_PATH.exists():
            return _error(_DB_UNAVAILABLE, 503)

        try:
            conn = _open_db()
            try:
                row = conn.execute(
                    "SELECT id, enabled FROM skills WHERE id = ?", (skill_id,)
                ).fetchone()
                if row is None:
                    return _error(_NOT_FOUND, 404)

                new_enabled = 0 if row["enabled"] else 1
                conn.execute(
                    "UPDATE skills SET enabled = ? WHERE id = ?", (new_enabled, skill_id)
                )
                conn.commit()
            finally:
                conn.close()

            logger.info("[SKILLS API] Toggled skill id=%d enabled=%d", skill_id, new_enabled)
            return SkillToggleResult(skill_id=skill_id, enabled=bool(new_enabled))
        except Exception as exc:
            logger.error("[SKILLS API] PUT /api/skills/%d/toggle failed: %s", skill_id, exc)
            return _error("Failed to toggle skill", 500)


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
        if not SKILLS_DB_PATH.exists():
            return _error(_DB_UNAVAILABLE, 503)

        try:
            conn = _open_db()
            try:
                row = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, source "
                    "FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
                if row is None:
                    return _error(_NOT_FOUND, 404)
                if row["source"] != "curated":
                    return _error(_ONLY_CURATED_COPY, 422)

                base_title = row["title"]
                copy_title = f"{base_title} (Custom)"
                tags = row["tags"] or ""

                existing_copy = conn.execute(
                    "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                    (copy_title,),
                ).fetchone()
                if existing_copy is not None:
                    return _error(f"A user copy named '{copy_title}' already exists", 409)

                conn.execute(
                    "INSERT INTO skills(title, use_for, content, tags, version, source, based_on) "
                    "VALUES (?, ?, ?, ?, ?, 'user', ?)",
                    (copy_title, row["use_for"], row["content"], tags, DEFAULT_VERSION, skill_id),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,))

                _index_new_skill(conn, new_id, copy_title, row["use_for"], tags)
                conn.commit()

                ensure_user_skills_dir()
                write_skill_file(
                    skill_yaml_path(copy_title),
                    {
                        "title": copy_title,
                        "use_for": row["use_for"],
                        "content": row["content"],
                        "tags": tags,
                        "version": DEFAULT_VERSION,
                    },
                )

                new_row = conn.execute(
                    "SELECT id, title, use_for, content, tags, version, "
                    "source, enabled, based_on FROM skills WHERE id = ?",
                    (new_id,),
                ).fetchone()
            finally:
                conn.close()

            logger.info(
                "[SKILLS API] Copied curated skill id=%d '%s' -> user skill id=%d '%s'",
                skill_id, base_title, new_id, copy_title,
            )
            return Skill.model_validate(_row_to_dict(new_row))
        except Exception as exc:
            logger.error("[SKILLS API] POST /api/skills/%d/copy failed: %s", skill_id, exc)
            return _error("Failed to copy skill", 500)

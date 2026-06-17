"""
Skills blueprint — Brain Skills tab API.

Provides CRUD endpoints for managing skills in skills.sqlite plus
user-skill YAML files in data/skills/user/.
"""

import logging
import sqlite3

from flask import Blueprint, jsonify, request

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

logger = logging.getLogger(__name__)

skills_bp = Blueprint("skills", __name__, url_prefix="/api/skills")


# ── Internal helpers ───────────────────────────────────────────────────────────


def _open_db() -> sqlite3.Connection:
    """Open skills.sqlite with row_factory for API dict conversion."""
    return open_skills_db(row_factory=True)


def _row_to_dict(row: sqlite3.Row) -> dict:
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


def _load_associations(conn: sqlite3.Connection) -> list[dict]:
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
            "created_at": r["created_at"],
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


# ── Endpoints ──────────────────────────────────────────────────────────────────


@skills_bp.route("", methods=["GET"])
@require_session
def list_skills():
    if not SKILLS_DB_PATH.exists():
        return jsonify({"skills": [], "associations": []}), 200

    try:
        conn = _open_db()
        try:
            rows = conn.execute(
                "SELECT id, title, use_for, content, tags, version, "
                "source, enabled, based_on "
                "FROM skills ORDER BY source, title"
            ).fetchall()
            skills = [_row_to_dict(r) for r in rows]
            associations = _load_associations(conn)
        finally:
            conn.close()

        return jsonify({"skills": skills, "associations": associations}), 200
    except Exception as exc:
        logger.error("[SKILLS API] GET /api/skills failed: %s", exc)
        return jsonify({"error": "Failed to load skills"}), 500


@skills_bp.route("", methods=["POST"])
@require_session
def create_skill():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    use_for = (data.get("use_for") or "").strip()
    content = (data.get("content") or "").strip()

    if not title:
        return jsonify({"error": "title is required"}), 400
    if not use_for:
        return jsonify({"error": "use_for is required"}), 400
    if not content:
        return jsonify({"error": "content is required"}), 400

    if not SKILLS_DB_PATH.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        tags = (data.get("tags") or "").strip()

        conn = _open_db()
        try:
            existing = conn.execute(
                "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                (title,),
            ).fetchone()
            if existing is not None:
                return jsonify({"error": f"A user skill named '{title}' already exists"}), 409

            conn.execute(
                "INSERT INTO skills(title, use_for, content, tags, version, source) "
                "VALUES (?, ?, ?, ?, ?, 'user')",
                (title, use_for, content, tags, DEFAULT_VERSION),
            )
            skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            _index_new_skill(conn, skill_id, title, use_for, tags)
            conn.commit()

            ensure_user_skills_dir()
            write_skill_file(skill_yaml_path(title), {
                "title": title, "use_for": use_for, "content": content,
                "tags": tags, "version": DEFAULT_VERSION,
            })

            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, "
                "source, enabled, based_on FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
            skill = _row_to_dict(row)
        finally:
            conn.close()

        logger.info("[SKILLS API] Created skill '%s' (id=%d)", title, skill_id)
        return jsonify({"skill": skill}), 201
    except Exception as exc:
        logger.error("[SKILLS API] POST /api/skills failed: %s", exc)
        return jsonify({"error": "Failed to create skill"}), 500


@skills_bp.route("/<int:skill_id>", methods=["PUT"])
@require_session
def update_skill(skill_id: int):
    data = request.get_json(silent=True) or {}

    if not SKILLS_DB_PATH.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, "
                "source, enabled, based_on FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
            if row is None:
                return jsonify({"error": "Skill not found"}), 404
            if row["source"] != "user":
                return jsonify({"error": "Only user-created skills can be edited"}), 403

            title = row["title"]
            updated = {
                "title": title,
                "use_for": (data.get("use_for") or "").strip() or row["use_for"],
                "content": (data.get("content") or "").strip() or row["content"],
                "tags": (data.get("tags") if data.get("tags") is not None else row["tags"]) or "",
                "version": row["version"] + 1,
            }

            conn.execute(
                "UPDATE skills SET use_for=?, content=?, tags=?, version=? "
                "WHERE id=?",
                (updated["use_for"], updated["content"], updated["tags"],
                 updated["version"], skill_id),
            )

            remove_search_entries(conn, skill_id)
            _index_new_skill(conn, skill_id, title, updated["use_for"], updated["tags"])
            conn.commit()

            ensure_user_skills_dir()
            write_skill_file(skill_yaml_path(title), updated)

            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, "
                "source, enabled, based_on FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
            skill = _row_to_dict(row)
        finally:
            conn.close()

        logger.info("[SKILLS API] Updated skill id=%d '%s' to v%d", skill_id, title, updated["version"])
        return jsonify({"skill": skill}), 200
    except Exception as exc:
        logger.error("[SKILLS API] PUT /api/skills/%d failed: %s", skill_id, exc)
        return jsonify({"error": "Failed to update skill"}), 500


@skills_bp.route("/<int:skill_id>", methods=["DELETE"])
@require_session
def delete_skill(skill_id: int):
    if not SKILLS_DB_PATH.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, title, source FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            if row is None:
                return jsonify({"error": "Skill not found"}), 404
            if row["source"] != "user":
                return jsonify({"error": "Only user-created skills can be deleted"}), 403

            title = row["title"]
            path = skill_yaml_path(title)

            remove_search_entries(conn, skill_id)
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            conn.commit()

            if path.exists() and path.resolve().is_relative_to(
                USER_SKILLS_DIR.resolve()
            ):
                path.unlink()
        finally:
            conn.close()

        logger.info("[SKILLS API] Deleted skill id=%d '%s'", skill_id, title)
        return jsonify({"deleted": True}), 200
    except Exception as exc:
        logger.error("[SKILLS API] DELETE /api/skills/%d failed: %s", skill_id, exc)
        return jsonify({"error": "Failed to delete skill"}), 500


@skills_bp.route("/<int:skill_id>/toggle", methods=["PUT"])
@require_session
def toggle_skill(skill_id: int):
    if not SKILLS_DB_PATH.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, enabled FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            if row is None:
                return jsonify({"error": "Skill not found"}), 404

            new_enabled = 0 if row["enabled"] else 1
            conn.execute("UPDATE skills SET enabled = ? WHERE id = ?", (new_enabled, skill_id))
            conn.commit()
        finally:
            conn.close()

        logger.info("[SKILLS API] Toggled skill id=%d enabled=%d", skill_id, new_enabled)
        return jsonify({"skill_id": skill_id, "enabled": bool(new_enabled)}), 200
    except Exception as exc:
        logger.error("[SKILLS API] PUT /api/skills/%d/toggle failed: %s", skill_id, exc)
        return jsonify({"error": "Failed to toggle skill"}), 500


@skills_bp.route("/<int:skill_id>/copy", methods=["POST"])
@require_session
def copy_skill(skill_id: int):
    if not SKILLS_DB_PATH.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, source "
                "FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
            if row is None:
                return jsonify({"error": "Skill not found"}), 404
            if row["source"] != "curated":
                return jsonify({"error": "Only curated skills can be copied"}), 400

            base_title = row["title"]
            copy_title = f"{base_title} (Custom)"
            tags = row["tags"] or ""

            existing_copy = conn.execute(
                "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                (copy_title,),
            ).fetchone()
            if existing_copy is not None:
                return jsonify({"error": f"A user copy named '{copy_title}' already exists"}), 409

            conn.execute(
                "INSERT INTO skills(title, use_for, content, tags, version, source, based_on) "
                "VALUES (?, ?, ?, ?, ?, 'user', ?)",
                (copy_title, row["use_for"], row["content"], tags,
                 DEFAULT_VERSION, skill_id),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,))

            _index_new_skill(conn, new_id, copy_title, row["use_for"], tags)
            conn.commit()

            ensure_user_skills_dir()
            write_skill_file(skill_yaml_path(copy_title), {
                "title": copy_title,
                "use_for": row["use_for"],
                "content": row["content"],
                "tags": tags,
                "version": DEFAULT_VERSION,
            })

            new_row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, "
                "source, enabled, based_on FROM skills WHERE id = ?",
                (new_id,),
            ).fetchone()
            skill = _row_to_dict(new_row)
        finally:
            conn.close()

        logger.info(
            "[SKILLS API] Copied curated skill id=%d '%s' -> user skill id=%d '%s'",
            skill_id, base_title, new_id, copy_title,
        )
        return jsonify({"skill": skill}), 201
    except Exception as exc:
        logger.error("[SKILLS API] POST /api/skills/%d/copy failed: %s", skill_id, exc)
        return jsonify({"error": "Failed to copy skill"}), 500

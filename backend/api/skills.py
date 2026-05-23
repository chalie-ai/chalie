"""
Skills blueprint — Brain Skills tab API.

Provides CRUD endpoints for managing skills in skills.sqlite plus
user-skill YAML files in data/skills/user/.
"""

import logging
import re
import sqlite3
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

_SKILLS_DB = Path(__file__).resolve().parent.parent / "abilities" / "assets" / "skills.sqlite"
_USER_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills" / "user"

_SLUG_MAX_LENGTH = 64
_DEFAULT_VERSION = 1

skills_bp = Blueprint("skills", __name__, url_prefix="/api/skills")


# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        try:
            conn.load_extension("vec0")
        except Exception:
            pass


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_SKILLS_DB))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)
    return conn


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:_SLUG_MAX_LENGTH]


def _skill_path(title: str) -> Path:
    return _USER_SKILLS_DIR / f"{_slugify(title)}.yaml"


def _ensure_user_skills_dir() -> None:
    _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def _write_skill_file(path: Path, meta: dict) -> None:
    frontmatter = {
        "title": meta["title"],
        "use_for": meta["use_for"],
        "tags": meta.get("tags", ""),
        "version": meta.get("version", _DEFAULT_VERSION),
        "related_abilities": meta.get("related_abilities", ""),
    }
    body = meta.get("content", "")
    content = "---\n" + yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True) + "---\n\n" + body + "\n"
    path.write_text(content, encoding="utf-8")


def _remove_search_entries(conn: sqlite3.Connection, skill_id: int) -> None:
    entry_ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM skill_search_entries WHERE skill_id = ?", (skill_id,)
        ).fetchall()
    ]
    for eid in entry_ids:
        conn.execute("DELETE FROM skill_search_vec WHERE rowid = ?", (eid,))
        conn.execute("DELETE FROM skill_search_fts WHERE rowid = ?", (eid,))
    conn.execute("DELETE FROM skill_search_entries WHERE skill_id = ?", (skill_id,))


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "use_for": row["use_for"],
        "content": row["content"],
        "tags": row["tags"] or "",
        "version": row["version"],
        "related_abilities": row["related_abilities"] or "",
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
    """Return all skills from skills.sqlite grouped with associations.

    Response 200::

        {
          "skills": [...],
          "associations": [...]
        }
    """
    if not _SKILLS_DB.exists():
        return jsonify({"skills": [], "associations": []}), 200

    try:
        conn = _open_db()
        try:
            rows = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, "
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
    """Create a new user skill.

    Request body::

        {"title": "...", "use_for": "...", "content": "...", "tags": "...", "related_abilities": "..."}

    Response 201::

        {"skill": {...}}
    """
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

    if not _SKILLS_DB.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        tags = (data.get("tags") or "").strip()
        related_abilities = (data.get("related_abilities") or "").strip()

        conn = _open_db()
        try:
            existing = conn.execute(
                "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                (title,),
            ).fetchone()
            if existing is not None:
                return jsonify({"error": f"A user skill named '{title}' already exists"}), 409

            conn.execute(
                "INSERT INTO skills(title, use_for, content, tags, version, related_abilities, source) "
                "VALUES (?, ?, ?, ?, ?, ?, 'user')",
                (title, use_for, content, tags, _DEFAULT_VERSION, related_abilities),
            )
            skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()

            _ensure_user_skills_dir()
            _write_skill_file(_skill_path(title), {
                "title": title, "use_for": use_for, "content": content,
                "tags": tags, "version": _DEFAULT_VERSION, "related_abilities": related_abilities,
            })

            _index_new_skill(conn, skill_id, title, use_for, tags)

            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, "
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
    """Update a user-created skill.

    Only source='user' skills can be edited.

    Request body::

        {"use_for": "...", "content": "...", "tags": "...", "related_abilities": "..."}

    Response 200::

        {"skill": {...}}
    """
    data = request.get_json(silent=True) or {}

    if not _SKILLS_DB.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, "
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
                "related_abilities": (
                    data.get("related_abilities")
                    if data.get("related_abilities") is not None
                    else row["related_abilities"]
                ) or "",
            }

            conn.execute(
                "UPDATE skills SET use_for=?, content=?, tags=?, version=?, related_abilities=? "
                "WHERE id=?",
                (updated["use_for"], updated["content"], updated["tags"],
                 updated["version"], updated["related_abilities"], skill_id),
            )
            conn.commit()

            _ensure_user_skills_dir()
            _write_skill_file(_skill_path(title), updated)

            _remove_search_entries(conn, skill_id)
            _index_new_skill(conn, skill_id, title, updated["use_for"], updated["tags"])

            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, "
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
    """Delete a user-created skill.

    Only source='user' skills can be deleted.

    Response 200::

        {"deleted": true}
    """
    if not _SKILLS_DB.exists():
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
            skill_path = _skill_path(title)

            _remove_search_entries(conn, skill_id)
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            conn.commit()

            if skill_path.exists():
                skill_path.unlink()
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
    """Toggle the enabled/disabled state of any skill.

    Response 200::

        {"skill_id": 1, "enabled": false}
    """
    if not _SKILLS_DB.exists():
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
    """Copy a curated skill as a new user skill and disable the original.

    Creates a user copy with based_on set to the curated skill id.
    Disables the curated original.

    Response 201::

        {"skill": {...}}
    """
    if not _SKILLS_DB.exists():
        return jsonify({"error": "skills database unavailable"}), 503

    try:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, source "
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
            related_abilities = row["related_abilities"] or ""

            existing_copy = conn.execute(
                "SELECT id FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
                (copy_title,),
            ).fetchone()
            if existing_copy is not None:
                return jsonify({"error": f"A user copy named '{copy_title}' already exists"}), 409

            conn.execute(
                "INSERT INTO skills(title, use_for, content, tags, version, related_abilities, source, based_on) "
                "VALUES (?, ?, ?, ?, ?, ?, 'user', ?)",
                (copy_title, row["use_for"], row["content"], tags,
                 _DEFAULT_VERSION, related_abilities, skill_id),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,))
            conn.commit()

            _ensure_user_skills_dir()
            _write_skill_file(_skill_path(copy_title), {
                "title": copy_title,
                "use_for": row["use_for"],
                "content": row["content"],
                "tags": tags,
                "version": _DEFAULT_VERSION,
                "related_abilities": related_abilities,
            })

            _index_new_skill(conn, new_id, copy_title, row["use_for"], tags)

            new_row = conn.execute(
                "SELECT id, title, use_for, content, tags, version, related_abilities, "
                "source, enabled, based_on FROM skills WHERE id = ?",
                (new_id,),
            ).fetchone()
            skill = _row_to_dict(new_row)
        finally:
            conn.close()

        logger.info(
            "[SKILLS API] Copied curated skill id=%d '%s' → user skill id=%d '%s'",
            skill_id, base_title, new_id, copy_title,
        )
        return jsonify({"skill": skill}), 201
    except Exception as exc:
        logger.error("[SKILLS API] POST /api/skills/%d/copy failed: %s", skill_id, exc)
        return jsonify({"error": "Failed to copy skill"}), 500

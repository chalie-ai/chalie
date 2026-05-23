"""
SkillBuilderAbility — create, edit, delete, and list user-defined skill playbooks.

User skills are stored as .yaml files in data/skills/user/ with the same
frontmatter format as curated skills in backend/abilities/skills/. On
create/edit they are also indexed into skills.sqlite for find_skills routing.
Only user-created skills (source='user') can be edited or deleted.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import ClassVar

import yaml

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_BUILDER]"
_USER_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills" / "user"
_DB_PATH = Path(__file__).resolve().parent / "assets" / "skills.sqlite"

_DEFAULT_VERSION = 1
_SLUG_MAX_LENGTH = 64


class SkillBuilderAbility(Ability):
    NAME = "skill_builder"
    SEARCH_TOOLTIP = "create, edit, delete, or list user-defined skill playbooks"
    SUMMARY = (
        "Create, edit, delete, or list custom skill playbooks. "
        "Use this to save step-by-step procedures that Chalie should follow for recurring tasks."
    )
    EXAMPLES = [
        "create a skill for tracking my weekly expenses",
        "save a playbook for how I like to research new topics",
        "add a skill called Morning Briefing that checks weather and news",
        "edit my Track Flights skill to also check for delays",
        "delete the skill I created for meal planning",
        "show me all my custom skills",
        "list the skills I've created",
        "what custom playbooks do I have",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit", "delete", "list"],
                "description": (
                    "create: save a new user skill playbook. "
                    "edit: update an existing user skill (identified by title). "
                    "delete: remove a user skill by title. "
                    "list: list all skills (both curated and user-created)."
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "The skill title — a short noun phrase describing what the skill does "
                    "(e.g. 'Track Package Delivery', 'Weekly Expense Review'). "
                    "Required for create, edit, and delete."
                ),
            },
            "use_for": {
                "type": "string",
                "description": (
                    "One sentence describing when to use this skill "
                    "(e.g. 'tracking package delivery status and estimated arrival times'). "
                    "Required for create."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "The skill instructions as numbered steps. Each step should be a clear, "
                    "actionable instruction that references specific tools when applicable "
                    "(e.g. 'Use the search tool to...', 'Check your memory for...'). "
                    "Structure: numbered steps (1. 2. 3.) describing the procedure. "
                    "Optionally end with a Preferences section for user-specific defaults. "
                    "Required for create."
                ),
            },
            "tags": {
                "type": "string",
                "description": (
                    "Comma-separated keywords that describe the skill's domain "
                    "(e.g. 'logistics, packages, delivery, tracking'). Optional."
                ),
            },
            "related_abilities": {
                "type": "string",
                "description": (
                    "Comma-separated list of Chalie tool names this skill uses "
                    "(e.g. 'search, schedule, memory'). Optional."
                ),
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 15

    _DB_PATH: ClassVar[Path] = _DB_PATH
    _USER_SKILLS_DIR: ClassVar[Path] = _USER_SKILLS_DIR

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "list")
        logger.info("%s action=%s channel=%s", _LOG_PREFIX, action, channel)

        try:
            if action == "create":
                text = _handle_create(params)
            elif action == "edit":
                text = _handle_edit(params)
            elif action == "delete":
                text = _handle_delete(params)
            elif action == "list":
                text = _handle_list(params)
            else:
                text = _skill_tag(
                    "skill_builder",
                    error=f"unknown-action:{action}",
                    valid="create,edit,delete,list",
                )
        except Exception as exc:
            logger.exception("%s Error in %s: %s", _LOG_PREFIX, action, exc)
            text = _skill_tag("skill_builder", action=action, error=str(exc)[:200])

        return {"text": text}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _slugify(title: str) -> str:
    """Convert a skill title to a safe filename slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:_SLUG_MAX_LENGTH]


def _skill_path(title: str) -> Path:
    return _USER_SKILLS_DIR / f"{_slugify(title)}.yaml"


def _ensure_user_skills_dir() -> None:
    _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def _write_skill_file(path: Path, meta: dict) -> None:
    """Write a skill metadata dict to a YAML frontmatter file."""
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


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension("vec0")


def _open_skills_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")
    _load_sqlite_vec(conn)
    return conn


def _find_user_skill_by_title(conn: sqlite3.Connection, title: str) -> dict | None:
    """Return the skill row for a user-created skill matching title (case-insensitive)."""
    row = conn.execute(
        "SELECT id, title, use_for, content, tags, version, related_abilities "
        "FROM skills WHERE source = 'user' AND lower(title) = lower(?)",
        (title,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "use_for": row[2],
        "content": row[3],
        "tags": row[4],
        "version": row[5],
        "related_abilities": row[6],
    }


def _remove_search_entries(conn: sqlite3.Connection, skill_id: int) -> None:
    """Remove all search index entries for a skill (CASCADE handles vec/fts)."""
    entry_ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM skill_search_entries WHERE skill_id = ?", (skill_id,)
        ).fetchall()
    ]
    for eid in entry_ids:
        conn.execute("DELETE FROM skill_search_vec WHERE rowid = ?", (eid,))
        conn.execute("DELETE FROM skill_search_fts WHERE rowid = ?", (eid,))
    conn.execute("DELETE FROM skill_search_entries WHERE skill_id = ?", (skill_id,))


# ── Action handlers ────────────────────────────────────────────────────────────


def _handle_create(params: dict) -> str:
    title = (params.get("title") or "").strip()
    use_for = (params.get("use_for") or "").strip()
    content = (params.get("content") or "").strip()

    if not title:
        return _skill_tag("skill_builder", action="create", error="title-required")
    if not use_for:
        return _skill_tag("skill_builder", action="create", error="use_for-required")
    if not content:
        return _skill_tag("skill_builder", action="create", error="content-required")

    if not _DB_PATH.exists():
        return _skill_tag("skill_builder", action="create", error="skill-db-unavailable")

    conn = _open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is not None:
            return _skill_tag(
                "skill_builder",
                action="create",
                error=f"skill-already-exists:{title}",
                hint="Use action=edit to update an existing skill",
            )

        tags = (params.get("tags") or "").strip()
        related_abilities = (params.get("related_abilities") or "").strip()
        meta = {
            "title": title,
            "use_for": use_for,
            "content": content,
            "tags": tags,
            "version": _DEFAULT_VERSION,
            "related_abilities": related_abilities,
        }

        _ensure_user_skills_dir()
        skill_path = _skill_path(title)
        _write_skill_file(skill_path, meta)

        conn.execute(
            "INSERT INTO skills(title, use_for, content, tags, version, related_abilities, source) "
            "VALUES (?, ?, ?, ?, ?, ?, 'user')",
            (title, use_for, content, tags, _DEFAULT_VERSION, related_abilities),
        )
        skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, use_for, tags)

        logger.info("%s Created skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, skill_path.name)
        return _skill_tag(
            "skill_builder",
            f'Skill "{title}" created and indexed.',
            action="create",
            status="ok",
            skill_id=skill_id,
        )
    finally:
        conn.close()


def _handle_edit(params: dict) -> str:
    title = (params.get("title") or "").strip()
    if not title:
        return _skill_tag("skill_builder", action="edit", error="title-required")

    if not _DB_PATH.exists():
        return _skill_tag("skill_builder", action="edit", error="skill-db-unavailable")

    conn = _open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is None:
            return _skill_tag(
                "skill_builder",
                action="edit",
                error=f"skill-not-found:{title}",
                hint="Only user-created skills can be edited",
            )

        skill_id = existing["id"]
        updated_meta = {
            "title": title,
            "use_for": (params.get("use_for") or "").strip() or existing["use_for"],
            "content": (params.get("content") or "").strip() or existing["content"],
            "tags": (params.get("tags") or "").strip() if params.get("tags") is not None else (existing["tags"] or ""),
            "version": existing["version"] + 1,
            "related_abilities": (
                (params.get("related_abilities") or "").strip()
                if params.get("related_abilities") is not None
                else (existing["related_abilities"] or "")
            ),
        }

        _ensure_user_skills_dir()
        skill_path = _skill_path(title)
        _write_skill_file(skill_path, updated_meta)

        conn.execute(
            "UPDATE skills SET use_for=?, content=?, tags=?, version=?, related_abilities=? "
            "WHERE id=?",
            (
                updated_meta["use_for"],
                updated_meta["content"],
                updated_meta["tags"],
                updated_meta["version"],
                updated_meta["related_abilities"],
                skill_id,
            ),
        )

        _remove_search_entries(conn, skill_id)

        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, updated_meta["use_for"], updated_meta["tags"])

        logger.info("%s Updated skill '%s' (id=%d, version=%d)", _LOG_PREFIX, title, skill_id, updated_meta["version"])
        return _skill_tag(
            "skill_builder",
            f'Skill "{title}" updated to version {updated_meta["version"]}.',
            action="edit",
            status="ok",
            skill_id=skill_id,
        )
    finally:
        conn.close()


def _handle_delete(params: dict) -> str:
    title = (params.get("title") or "").strip()
    if not title:
        return _skill_tag("skill_builder", action="delete", error="title-required")

    if not _DB_PATH.exists():
        return _skill_tag("skill_builder", action="delete", error="skill-db-unavailable")

    conn = _open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is None:
            return _skill_tag(
                "skill_builder",
                action="delete",
                error=f"skill-not-found:{title}",
                hint="Only user-created skills can be deleted",
            )

        skill_id = existing["id"]
        skill_path = _skill_path(title)

        _remove_search_entries(conn, skill_id)
        conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        conn.commit()

        if skill_path.exists():
            skill_path.unlink()

        logger.info("%s Deleted skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, skill_path.name)
        return _skill_tag(
            "skill_builder",
            f'Skill "{title}" deleted.',
            action="delete",
            status="ok",
        )
    finally:
        conn.close()


def _handle_list(params: dict) -> str:  # noqa: ARG001
    if not _DB_PATH.exists():
        return _skill_tag("skill_builder", action="list", error="skill-db-unavailable")

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        rows = conn.execute(
            "SELECT id, title, use_for, tags, version, source, enabled "
            "FROM skills ORDER BY source, title"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return _skill_tag("skill_builder", "No skills found.", action="list", found=0)

    lines = []
    for row in rows:
        skill_id, title, use_for, tags, version, source, enabled = row
        status = "enabled" if enabled else "disabled"
        tags_display = f" [{tags}]" if tags else ""
        lines.append(f"- [{source}] {title} (v{version}, {status}){tags_display}")
        lines.append(f"  {use_for}")

    body = "\n".join(lines)
    return _skill_tag("skill_builder", body, action="list", found=len(rows))

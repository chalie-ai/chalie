"""
SkillBuilderAbility — create, edit, delete, and list user-defined skill playbooks.

User skills are stored as .yaml files in data/skills/user/ with the same
frontmatter format as curated skills in backend/abilities/skills/. On
create/edit they are also indexed into skills.sqlite for find_skills routing.
Only user-created skills (source='user') can be edited or deleted.
"""

import logging
import sqlite3

from abilities._base import Ability
from services.file_mapper_service import FileMapperService
from services.innate_skills._tag import tag as _skill_tag
from utils.skills_io import (
    DEFAULT_VERSION,
    SKILLS_DB_PATH,
    ensure_user_skills_dir,
    open_skills_db,
    remove_search_entries,
    skill_yaml_path,
    write_skill_file,
)

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_BUILDER]"

# Internal/meta tools that should not appear in skill content guidance.
_META_TOOLS = frozenset({
    "find_tools", "find_skills", "skill_builder",
    "subagent", "review_tool_calls", "review_transcript",
    "save_graph", "save_pattern",
})


def _discover_tool_names() -> str:
    """Auto-discover user-facing ability names from the abilities directory."""
    abilities_dir = FileMapperService.get_abilities_path()
    names = sorted(
        p.stem for p in abilities_dir.glob("*.py")
        if not p.name.startswith("_") and p.stem not in _META_TOOLS
    )
    return ", ".join(names)


class SkillBuilderAbility(Ability):
    NAME = "skill_builder"
    SEARCH_TOOLTIP = "create, edit, delete, or list user-defined skill playbooks"
    POLICY_CATEGORY = "Skills"
    POLICY_LABELS = {
        "create": "Create custom skill",
        "delete": "Delete custom skill",
        "edit": "Edit custom skill",
        "list": "List custom skills",
    }
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
                    "The skill body as numbered steps (1. 2. 3. …). "
                    "Each step MUST: start with a verb, reference a tool name in backticks "
                    "(e.g. `memory`, `search`, `document`), and describe one clear action. "
                    f"Available tools: {_discover_tool_names()}. "
                    "Pattern — good: '1. Use `memory` to recall dietary preferences and restrictions.' "
                    "Pattern — bad: '1. Think about what the user might want.' (no tool, vague). "
                    "Aim for 5–10 steps. Steps should build logically: recall context → "
                    "gather data → process → produce output → persist results. "
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
        },
        "required": ["action"],
    }
    TIMEOUT = 15

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


def _find_user_skill_by_title(conn: sqlite3.Connection, title: str) -> dict | None:
    """Return the skill row for a user-created skill matching title (case-insensitive)."""
    row = conn.execute(
        "SELECT id, title, use_for, content, tags, version "
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
    }


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

    if not SKILLS_DB_PATH.exists():
        return _skill_tag("skill_builder", action="create", error="skill-db-unavailable")

    conn = open_skills_db()
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
        meta = {
            "title": title,
            "use_for": use_for,
            "content": content,
            "tags": tags,
            "version": DEFAULT_VERSION,
        }

        conn.execute(
            "INSERT INTO skills(title, use_for, content, tags, version, source) "
            "VALUES (?, ?, ?, ?, ?, 'user')",
            (title, use_for, content, tags, DEFAULT_VERSION),
        )
        skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, use_for, tags)

        conn.commit()

        ensure_user_skills_dir()
        path = skill_yaml_path(title)
        write_skill_file(path, meta)

        logger.info("%s Created skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, path.name)
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

    if not SKILLS_DB_PATH.exists():
        return _skill_tag("skill_builder", action="edit", error="skill-db-unavailable")

    conn = open_skills_db()
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
        }

        conn.execute(
            "UPDATE skills SET use_for=?, content=?, tags=?, version=? "
            "WHERE id=?",
            (
                updated_meta["use_for"],
                updated_meta["content"],
                updated_meta["tags"],
                updated_meta["version"],
                skill_id,
            ),
        )

        remove_search_entries(conn, skill_id)

        from services.embedding_service import EmbeddingService
        from utils.build_skills_db import index_skill
        emb_service = EmbeddingService()
        index_skill(conn, emb_service, skill_id, title, updated_meta["use_for"], updated_meta["tags"])

        conn.commit()

        ensure_user_skills_dir()
        write_skill_file(skill_yaml_path(title), updated_meta)

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

    if not SKILLS_DB_PATH.exists():
        return _skill_tag("skill_builder", action="delete", error="skill-db-unavailable")

    conn = open_skills_db()
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
        path = skill_yaml_path(title)

        remove_search_entries(conn, skill_id)
        conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        conn.commit()

        if path.exists():
            path.unlink()

        logger.info("%s Deleted skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, path.name)
        return _skill_tag(
            "skill_builder",
            f'Skill "{title}" deleted.',
            action="delete",
            status="ok",
        )
    finally:
        conn.close()


def _handle_list(params: dict) -> str:  # noqa: ARG001
    if not SKILLS_DB_PATH.exists():
        return _skill_tag("skill_builder", action="list", error="skill-db-unavailable")

    conn = sqlite3.connect(str(SKILLS_DB_PATH))
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

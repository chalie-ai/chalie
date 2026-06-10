"""Skill playbook CRUD — the merged ``skill_builder`` / ``skill_manager`` pair.

User skills are stored as .yaml files in data/skills/user/ with the same
frontmatter format as curated skills in backend/abilities/skills/. On
create/edit they are also indexed into skills.sqlite for find_skills routing.
Only user-created skills (source='user') can be edited or deleted.

ONE module, ONE behaviour: :class:`SkillBuilderAbility` owns every handler and
``run()``. :class:`SkillManagerAbility` is the SYSTEM-policy variant used by the
background ``SkillSuggestionMessageProcessor`` — it differs ONLY in its name
(``skill_manager``) and ``SYSTEM = True``; it shares the parent's handlers
verbatim. There is NO per-name content rewriting: the dispatcher renders the
wire envelope under ``get_name()``, so the ACT trail already shows the right
identity. A skill whose body legitimately contains the substring
``skill_builder`` is therefore stored and returned byte-identical under either
tool — the old blanket ``content.replace('skill_builder', 'skill_manager')``
that corrupted such bodies is gone.
"""

import logging
import sqlite3
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from services.file_mapper_service import FileMapperService
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
    "find_tools", "find_skills", "skill_builder", "skill_manager",
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
    # The ACTION_REQUIRED pre-gate (consulted by the dispatcher BEFORE the policy
    # gate and BEFORE run()): an unknown action → one unknown-action error whose
    # valid= names all four real actions; a known action missing required params →
    # one missing-params error naming ALL of them. edit/delete need only a title
    # (the existence/ownership checks live in run()); list needs nothing.
    ACTION_REQUIRED: ClassVar[dict] = {
        "create": ("title", "use_for", "content"),
        "edit": ("title",),
        "delete": ("title",),
        "list": (),
    }

    # SYSTEM=False on the user-facing tool; the SYSTEM variant flips this. Declared
    # here so both names carry a deterministic, introspectable policy identity.
    SYSTEM: ClassVar[bool] = False

    def get_name(self) -> str:
        return "skill_builder"

    def get_summary(self) -> str:
        return (
            "Create, edit, delete, or list custom skill playbooks. "
            "Use this to save step-by-step procedures that Chalie should follow for recurring tasks."
        )

    def get_examples(self) -> list[str]:
        return [
            "create a skill for tracking my weekly expenses",
            "save a playbook for how I like to research new topics",
            "add a skill called Morning Briefing that checks weather and news",
            "edit my Track Flights skill to also check for delays",
            "delete the skill I created for meal planning",
            "show me all my custom skills",
            "list the skills I've created",
            "what custom playbooks do I have",
        ]

    def get_search_tooltip(self) -> str:
        return "create, edit, delete, or list user-defined skill playbooks"

    _PARAMETERS: ClassVar[dict] = {
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

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        # The ACTION_REQUIRED pre-gate has already rejected an unknown action and
        # any missing required params before this point, so action is one of the
        # four real actions and its required params are present. No try/except
        # swallow: an unexpected failure bubbles to the dispatcher's _run, which
        # renders it as code=unhandled-exception (errors must surface).
        action = params.get("action", "list")
        config = getattr(self.mp, "config", None)
        channel = getattr(config, "channel", None)
        logger.info("%s action=%s channel=%s", _LOG_PREFIX, action, channel)

        if action == "create":
            return _handle_create(params)
        if action == "edit":
            return _handle_edit(params)
        if action == "delete":
            return _handle_delete(params)
        return _handle_list(params)


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


def _handle_create(params: dict) -> ToolResult:
    # title / use_for / content presence is guaranteed by the ACTION_REQUIRED
    # pre-gate; here we only normalise.
    title = (params.get("title") or "").strip()
    use_for = (params.get("use_for") or "").strip()
    content = (params.get("content") or "").strip()

    if not SKILLS_DB_PATH.exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="create",
        )

    conn = open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is not None:
            return ToolResult.err(
                f'A skill titled "{title}" already exists.',
                code="skill-already-exists",
                hint="Use action=edit to update an existing skill",
                action="create",
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
        return ToolResult.ok(
            f'Skill "{title}" created and indexed.',
            action="create",
            skill_id=skill_id,
        )
    finally:
        conn.close()


def _handle_edit(params: dict) -> ToolResult:
    title = (params.get("title") or "").strip()  # presence guaranteed by pre-gate

    if not SKILLS_DB_PATH.exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="edit",
        )

    conn = open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is None:
            return ToolResult.err(
                f'No user skill titled "{title}" was found.',
                code="skill-not-found",
                hint="Only user-created skills can be edited",
                action="edit",
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
        return ToolResult.ok(
            f'Skill "{title}" updated to version {updated_meta["version"]}.',
            action="edit",
            skill_id=skill_id,
        )
    finally:
        conn.close()


def _handle_delete(params: dict) -> ToolResult:
    title = (params.get("title") or "").strip()  # presence guaranteed by pre-gate

    if not SKILLS_DB_PATH.exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="delete",
        )

    conn = open_skills_db()
    try:
        existing = _find_user_skill_by_title(conn, title)
        if existing is None:
            return ToolResult.err(
                f'No user skill titled "{title}" was found.',
                code="skill-not-found",
                hint="Only user-created skills can be deleted",
                action="delete",
            )

        skill_id = existing["id"]
        path = skill_yaml_path(title)

        remove_search_entries(conn, skill_id)
        conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        conn.commit()

        if path.exists():
            path.unlink()

        logger.info("%s Deleted skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, path.name)
        return ToolResult.ok(
            f'Skill "{title}" deleted.',
            action="delete",
        )
    finally:
        conn.close()


def _handle_list(params: dict) -> ToolResult:  # noqa: ARG001
    if not SKILLS_DB_PATH.exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="list",
        )

    conn = sqlite3.connect(str(SKILLS_DB_PATH))
    try:
        rows = conn.execute(
            "SELECT id, title, use_for, tags, version, source, enabled "
            "FROM skills ORDER BY source, title"
        ).fetchall()
    finally:
        conn.close()

    # Structured rows (JSON), not prose: a weak model can read each skill's
    # fields directly. count meta mirrors len(body) so the model sees the total
    # without parsing.
    skills = [
        {
            "id": skill_id,
            "title": title,
            "use_for": use_for,
            "tags": tags or "",
            "version": version,
            "source": source,
            "enabled": bool(enabled),
        }
        for skill_id, title, use_for, tags, version, source, enabled in rows
    ]
    return ToolResult.ok(skills, action="list", count=len(skills))


class SkillManagerAbility(SkillBuilderAbility):
    """SYSTEM-policy variant of :class:`SkillBuilderAbility`.

    Used exclusively by ``SkillSuggestionMessageProcessor`` (the background
    skill-creation loop). It inherits EVERYTHING — handlers, ``run()``, all five
    metadata getters, ``ACTION_REQUIRED`` — from the parent unchanged; it differs
    ONLY in ``get_name()`` (so the ACT trail and the policy gate see
    ``skill_manager``) and ``SYSTEM = True`` (the tool is in
    ``PolicyManager.INTERNAL`` and bypasses the gate).

    There is NO per-name content rewriting: the dispatcher renders the envelope
    under ``get_name()`` already, so a skill body containing the literal string
    ``skill_builder`` survives a ``skill_manager`` operation byte-identical.
    """

    SYSTEM: ClassVar[bool] = True

    def get_name(self) -> str:
        return "skill_manager"

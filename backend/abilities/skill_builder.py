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
wire envelope under the ability's ``NAME``, so the ACT trail already shows the
right identity. A skill whose body legitimately contains the substring
``skill_builder`` is therefore stored and returned byte-identical under either
tool — the old blanket ``content.replace('skill_builder', 'skill_manager')``
that corrupted such bodies is gone.
"""

import logging
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.channels import Channel
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.skill_builder_params_bag import (
    SkillBuilderCreateParams,
    SkillBuilderDeleteParams,
    SkillBuilderEditParams,
    SkillBuilderListParams,
    SkillBuilderParamsBag,
    SkillBuilderReadParams,
)
from models.skill import Skill
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

_LOG_PREFIX = "[SKILL_BUILDER]"

# Internal/meta tools that should not appear in skill content guidance.
_META_TOOLS = frozenset({
    "find_tools", "find_skills", "skill_builder", "skill_manager",
    "review_tool_calls", "review_transcript",
    "save_graph", "save_pattern",
})


def _discover_tool_names() -> str:
    abilities_dir = FileMapperService.get_abilities_path()
    names = sorted(
        p.stem for p in abilities_dir.glob("*.py")
        if not p.name.startswith("_") and p.stem not in _META_TOOLS
    )
    return ", ".join(names)


class SkillBuilderAbility(Ability[SkillBuilderParamsBag]):
    # The ACTION_REQUIRED pre-gate (consulted by the dispatcher BEFORE the policy
    # gate and BEFORE run()): an unknown action → one unknown-action error whose
    # valid= names all five real actions; a known action missing required params →
    # one missing-params error naming ALL of them. edit/delete/read need only a
    # title (the existence/ownership checks live in run()); list needs nothing.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "create": (Keys.title_, Keys.use_for, Keys.content),
        "edit": (Keys.title_,),
        "delete": (Keys.title_,),
        "read": (Keys.title_,),
        "list": (),
    }

    # The typed input contract: the dispatch seam builds the bag via
    # SkillBuilderParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SkillBuilderParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("skills", "create skill", "playbook", "skill playbooks")

    # SYSTEM=False on the user-facing tool; the SYSTEM variant flips this. Declared
    # here so both names carry a deterministic, introspectable policy identity.
    SYSTEM: ClassVar[bool] = False
    NAME: ClassVar[str] = "skill_builder"

    def get_summary(self) -> str:
        return (
            "Create, edit, delete, read, or list custom skill playbooks. "
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
            "show me the steps in my Morning Briefing skill",
            "list the skills I've created",
        ]

    def get_search_tooltip(self) -> str:
        return "create, edit, delete, read, or list user-defined skill playbooks"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["create", "edit", "delete", "read", "list"],
                "description": (
                    "create: save a new user skill playbook. "
                    "edit: update an existing user skill (identified by title). "
                    "delete: remove a user skill by title. "
                    "read: show the full content (every step) of one skill by title. "
                    "list: list all skills (both curated and user-created), titles and use_for only."
                ),
            },
            Keys.title_: {
                "type": "string",
                "description": (
                    "The skill title — a short noun phrase describing what the skill does "
                    "(e.g. 'Track Package Delivery', 'Weekly Expense Review'). "
                    "Required for create, edit, delete, and read."
                ),
            },
            Keys.use_for: {
                "type": "string",
                "description": (
                    "One sentence describing when to use this skill "
                    "(e.g. 'tracking package delivery status and estimated arrival times'). "
                    "Required for create."
                ),
            },
            Keys.content: {
                "type": "string",
                "description": (
                    "The skill body as numbered steps (1. 2. 3. …). "
                    "Each step MUST: start with a verb, reference a tool name in backticks "
                    "(e.g. `memory`, `search`, `file_write`), and describe one clear action. "
                    f"Available tools: {_discover_tool_names()}. "
                    "Pattern — good: '1. Use `memory` to recall dietary preferences and restrictions.' "
                    "Pattern — bad: '1. Think about what the user might want.' (no tool, vague). "
                    "Aim for 5–10 steps. Steps should build logically: recall context → "
                    "gather data → process → produce output → persist results. "
                    "Optionally end with a Preferences section for user-specific defaults. "
                    "Required for create."
                ),
            },
            Keys.tags: {
                "type": "string",
                "description": (
                    "Comma-separated keywords that describe the skill's domain "
                    "(e.g. 'logistics, packages, delivery, tracking'). Optional."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SkillBuilderParamsBag) -> ToolResult:
        # The bag is built (= validated) at the dispatch seam, so every leaf that
        # reaches this point carries proven fields. No try/except swallow: an
        # unexpected failure bubbles to the dispatcher's _run, which renders it
        # as code=unhandled-exception (errors must surface).
        mp = self.mp
        if mp is None:
            raise RuntimeError("skill_builder.run() dispatched without a bound MessageProcessor")

        channel = mp.config.channel
        logger.info("%s bag=%s channel=%s", _LOG_PREFIX, type(params).__name__, channel)

        if isinstance(params, SkillBuilderEditParams):
            result = _handle_edit(params)
        elif isinstance(params, SkillBuilderCreateParams):
            result = _handle_create(params)
        elif isinstance(params, SkillBuilderDeleteParams):
            result = _handle_delete(params)
        elif isinstance(params, SkillBuilderReadParams):
            result = _handle_read(params)
        elif isinstance(params, SkillBuilderListParams):
            result = _handle_list()
        else:
            return ToolResult.err(
                f"Unknown skill_builder params bag: {type(params).__name__}.",
                code="unknown-action",
                valid=("create", "edit", "delete", "read", "list"),
            )

        # The background suggestion loop (channel 'skills_building') saves exactly
        # ONE skill per turn: the instant a create/edit succeeds, halt the recursive
        # ACT loop so the model cannot keep emitting near-duplicate writes. Other
        # channels (a user explicitly building a skill) are unaffected.
        wrote = isinstance(params, (SkillBuilderCreateParams, SkillBuilderEditParams))
        if channel == Channel.SKILLS_BUILDING and wrote and result.status == "success":
            mp.turn_execution_service.cancel()
        return result


# ── Action handlers ────────────────────────────────────────────────────────────


def _handle_create(params: SkillBuilderCreateParams) -> ToolResult:
    title = params.title
    use_for = params.use_for
    content = params.content

    if not FileMapperService.get_skills_db_path().exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="create",
        )

    existing = Skill.find_by_title_ci(title, source="user")
    if existing is not None:
        return ToolResult.err(
            f'A skill titled "{title}" already exists.',
            code="skill-already-exists",
            hint="Use action=edit to update an existing skill",
            action="create",
        )

    meta = {
        "title": title,
        "use_for": use_for,
        "content": content,
        "tags": params.tags,
        "version": DEFAULT_VERSION,
    }

    # Column list intentionally matches the DDL default columns (enabled,
    # based_on) left unset: save() only INSERTs the fields set on the
    # instance, the same explicit (title, use_for, content, tags, version,
    # source) list the raw INSERT used, so `enabled`/`based_on` still fall
    # through to their DDL defaults exactly as before.
    skill = Skill(
        title=title,
        use_for=use_for,
        content=content,
        tags=params.tags,
        version=DEFAULT_VERSION,
        source="user",
    ).save()
    skill_id: int = cast("int", skill.id)

    conn = Database.conn(str(FileMapperService.get_skills_db_path()))
    from services.embedding_service import EmbeddingService
    from utils.build_skills_db import index_skill
    emb_service = EmbeddingService()
    index_skill(conn, emb_service, skill_id, title, use_for, params.tags)

    conn.commit()

    ensure_user_skills_dir()
    path = skill_yaml_path(title)
    write_skill_file(path, cast("dict[str, str | int]", meta))

    logger.info("%s Created skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, path.name)
    return ToolResult.ok(
        f'Skill "{title}" created and indexed.',
        action="create",
        skill_id=skill_id,
    )


def _handle_edit(params: SkillBuilderEditParams) -> ToolResult:
    title = params.title

    if not FileMapperService.get_skills_db_path().exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="edit",
        )

    existing = Skill.find_by_title_ci(title, source="user")
    if existing is None:
        return ToolResult.err(
            f'No user skill titled "{title}" was found.',
            code="skill-not-found",
            hint="Only user-created skills can be edited",
            action="edit",
        )

    skill_id: int = cast("int", existing.id)
    updated_meta: dict[str, object] = {
        "title": title,
        "use_for": params.use_for or existing.use_for,
        "content": params.content or existing.content,
        "tags": params.tags if params.tags is not None else (existing.tags or ""),
        "version": existing.version + 1,
    }

    Skill.filter("id", skill_id).update(
        use_for=updated_meta["use_for"],
        content=updated_meta["content"],
        tags=updated_meta["tags"],
        version=updated_meta["version"],
    )

    conn = Database.conn(str(FileMapperService.get_skills_db_path()))
    remove_search_entries(conn, skill_id)

    from services.embedding_service import EmbeddingService
    from utils.build_skills_db import index_skill
    emb_service = EmbeddingService()
    index_skill(conn, emb_service, skill_id, title, cast("str", updated_meta["use_for"]), cast("str", updated_meta["tags"]))

    conn.commit()

    ensure_user_skills_dir()
    write_skill_file(skill_yaml_path(title), cast("dict[str, str | int]", updated_meta))

    logger.info("%s Updated skill '%s' (id=%d, version=%d)", _LOG_PREFIX, title, skill_id, updated_meta["version"])
    return ToolResult.ok(
        f'Skill "{title}" updated to version {updated_meta["version"]}.',
        action="edit",
        skill_id=skill_id,
    )


def _handle_delete(params: SkillBuilderDeleteParams) -> ToolResult:
    title = params.title

    if not FileMapperService.get_skills_db_path().exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="delete",
        )

    existing = Skill.find_by_title_ci(title, source="user")
    if existing is None:
        return ToolResult.err(
            f'No user skill titled "{title}" was found.',
            code="skill-not-found",
            hint="Only user-created skills can be deleted",
            action="delete",
        )

    skill_id: int = cast("int", existing.id)
    path = skill_yaml_path(title)

    conn = Database.conn(str(FileMapperService.get_skills_db_path()))
    remove_search_entries(conn, skill_id)
    Skill.filter("id", skill_id).delete()
    conn.commit()

    if path.exists():
        path.unlink()

    logger.info("%s Deleted skill '%s' (id=%d, file=%s)", _LOG_PREFIX, title, skill_id, path.name)
    return ToolResult.ok(
        f'Skill "{title}" deleted.',
        action="delete",
    )


def _handle_list() -> ToolResult:
    if not FileMapperService.get_skills_db_path().exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="list",
        )

    rows = Skill.order_by("source, title").select(
        "id", "title", "use_for", "tags", "version", "source", "enabled"
    )

    # Structured rows (JSON), not prose: a weak model can read each skill's
    # fields directly. count meta mirrors len(body) so the model sees the total
    # without parsing.
    skills = [
        {
            "id": row["id"],
            "title": row["title"],
            "use_for": row["use_for"],
            "tags": row["tags"] or "",
            "version": row["version"],
            "source": row["source"],
            "enabled": bool(row["enabled"]),
        }
        for row in rows
    ]
    if not skills:
        return ToolResult.no_results(action="list")
    return ToolResult.ok(skills, action="list", count=len(skills))


def _handle_read(params: SkillBuilderReadParams) -> ToolResult:
    title = params.title

    if not FileMapperService.get_skills_db_path().exists():
        return ToolResult.err(
            "The skill store is unavailable.",
            code="skill-db-unavailable",
            action="read",
        )

    # Match any source; prefer the editable user copy when a title collides
    # with a curated skill. Unlike list, this returns the full `content` so
    # the model can read a skill's steps before action=edit merges into them.
    skill = Skill.find_by_title_ci_prefer_user(title)

    if skill is None:
        return ToolResult.err(
            f'No skill titled "{title}" was found.',
            code="skill-not-found",
            hint="Call action=list to see the exact titles that exist",
            action="read",
        )

    return ToolResult.ok(
        {
            "id": skill.id,
            "title": skill.title,
            "use_for": skill.use_for,
            "content": skill.content,
            "tags": skill.tags or "",
            "version": skill.version,
            "source": skill.source,
            "enabled": bool(skill.enabled),
        },
        action="read",
    )


class SkillManagerAbility(SkillBuilderAbility):
    """SYSTEM-policy variant of :class:`SkillBuilderAbility`.

    Used exclusively by ``SkillSuggestionMessageProcessor`` (the background
    skill-creation loop). It inherits EVERYTHING — handlers, ``run()``, all five
    metadata getters, ``ACTION_REQUIRED`` — from the parent unchanged; it differs
    ONLY in ``NAME`` (so the ACT trail and the policy gate see
    ``skill_manager``) and ``SYSTEM = True`` (the tool is in
    ``PolicyManager.INTERNAL`` and bypasses the gate).

    There is NO per-name content rewriting: the dispatcher renders the envelope
    under ``NAME`` already, so a skill body containing the literal string
    ``skill_builder`` survives a ``skill_manager`` operation byte-identical.
    """

    SYSTEM: ClassVar[bool] = True
    # Override the parent's discoverability: skill_manager is pinned exclusively on
    # SkillSuggestionConfig.always_available; skill_builder (the parent) stays discoverable.
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "skill_manager"

import shutil
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor as _MP
    from services.processor_config import ProcessorConfig as _PC

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

pytestmark_unit = pytest.mark.unit
pytestmark_integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_DB = FileMapperService.get_skills_db_path()


def _copy_db(tmp_path: Path) -> Path:
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_DB), str(dest))
    return dest


def _skill_count(db_path: Path, source: str = "user") -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM skills WHERE source = ?", (source,)
        ).fetchone()
        return cast(int, row[0])
    finally:
        conn.close()


def _get_skill_row(db_path: Path, title: str) -> dict[str, object] | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, title, use_for, content, tags, version, source "
            "FROM skills WHERE lower(title) = lower(?)",
            (title,),
        ).fetchone()
        if row is None:
            return None
        return {k: row[i] for i, k in enumerate(("id", "title", "use_for", "content", "tags", "version", "source"))}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fixture: per-test isolated skills DB + YAML dir
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    db_path = _copy_db(tmp_path)
    yaml_dir = tmp_path / "user_skills"
    yaml_dir.mkdir()

    import utils.skills_io as sio
    import abilities.skill_builder as sb

    monkeypatch.setattr(sio, "SKILLS_DB_PATH", db_path)
    monkeypatch.setattr(sio, "USER_SKILLS_DIR", yaml_dir)
    monkeypatch.setattr(sb, "SKILLS_DB_PATH", db_path)

    return {"db_path": db_path, "yaml_dir": yaml_dir}


# ===========================================================================
# UNIT TESTS — pure functions in utils/skills_io.py
# ===========================================================================


@pytest.mark.unit
class TestSlugifyTitle:

    def test_spaces_become_hyphens(self) -> None:
        from utils.skills_io import slugify_title

        assert slugify_title("Track Package Delivery") == "track-package-delivery"

    def test_special_chars_become_hyphens(self) -> None:
        from utils.skills_io import slugify_title

        assert slugify_title("Morning & Evening!") == "morning-evening"

    def test_truncated_at_64_chars(self) -> None:
        from utils.skills_io import slugify_title

        long_title = "a" * 100
        result = slugify_title(long_title)
        assert len(result) == 64

    def test_leading_trailing_hyphens_stripped(self) -> None:
        from utils.skills_io import slugify_title

        result = slugify_title("  -- hello world --  ")
        assert not result.startswith("-")
        assert not result.endswith("-")
        assert "hello" in result


@pytest.mark.unit
class TestSkillYamlPath:

    def test_returns_path_in_user_skills_dir(self) -> None:
        from utils.skills_io import USER_SKILLS_DIR, skill_yaml_path

        path = skill_yaml_path("My Custom Skill")
        assert path.parent == USER_SKILLS_DIR

    def test_filename_is_slugified_title_with_yaml_extension(self) -> None:
        from utils.skills_io import skill_yaml_path, slugify_title

        title = "Weekly Expense Review"
        path = skill_yaml_path(title)
        assert path.name == f"{slugify_title(title)}.yaml"
        assert path.suffix == ".yaml"


# ===========================================================================
# INTEGRATION TESTS — SkillBuilderAbility.execute() against real DB
# ===========================================================================


@pytest.mark.integration
class TestSkillBuilderCreateValidation:

    def test_create_existing_db_succeeds(self, skill_env: dict[str, Path]) -> None:
        from abilities._result import ToolResult
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        result = ability.run({
            "action": "create",
            "title": "My Skill",
            "use_for": "tracking things",
            "content": "1. Use `memory` to recall context.",
        })

        assert isinstance(result, ToolResult)
        assert result.status == "success"
        assert result.meta["action"] == "create"


@pytest.mark.integration
class TestSkillBuilderLifecycle:

    _TITLE = "Test Weekly Review"
    _USE_FOR = "conducting structured weekly reviews"
    _CONTENT = "1. Summarise completed tasks.\n2. Plan next week."

    def test_create_inserts_row_and_writes_yaml(self, skill_env: dict[str, Path]) -> None:
        from abilities.skill_builder import SkillBuilderAbility

        db_path = skill_env["db_path"]
        yaml_dir = skill_env["yaml_dir"]
        before = _skill_count(db_path)

        ability = SkillBuilderAbility()
        result = ability.run({
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
                "tags": "productivity, planning",
            })

        assert result.status == "success"
        assert result.meta["action"] == "create"
        assert _skill_count(db_path) == before + 1

        row = _get_skill_row(db_path, self._TITLE)
        assert row is not None
        assert row["source"] == "user"
        assert row["version"] == 1
        assert row["use_for"] == self._USE_FOR

        yaml_files = list(yaml_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        content = yaml_files[0].read_text(encoding="utf-8")
        assert self._TITLE in content
        assert self._USE_FOR in content

    def test_create_duplicate_title_returns_error(self, skill_env: dict[str, Path]) -> None:
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        params: dict[str, object] = {
            "action": "create",
            "title": self._TITLE,
            "use_for": self._USE_FOR,
            "content": self._CONTENT,
        }
        ability.run(params)
        result = ability.run(params)

        assert result.status == "error"
        assert result.code == "skill-already-exists"

    def test_edit_increments_version_and_updates_content(self, skill_env: dict[str, Path]) -> None:
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        ability.run({
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            })

        new_content = "1. Summarise tasks.\n2. Review blockers.\n3. Plan next week."
        result = ability.run({"action": "edit", "title": self._TITLE, "content": new_content})

        assert result.status == "success"
        assert result.meta["action"] == "edit"

        row = _get_skill_row(skill_env["db_path"], self._TITLE)
        assert row is not None
        assert row["version"] == 2
        assert row["content"] == new_content

    def test_delete_removes_row_and_yaml_file(self, skill_env: dict[str, Path]) -> None:
        from abilities.skill_builder import SkillBuilderAbility

        db_path = skill_env["db_path"]
        ability = SkillBuilderAbility()
        ability.run({
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            })

        before = _skill_count(db_path)
        result = ability.run({"action": "delete", "title": self._TITLE})

        assert result.status == "success"
        assert result.meta["action"] == "delete"
        assert _skill_count(db_path) == before - 1
        assert _get_skill_row(db_path, self._TITLE) is None

        yaml_files = list(skill_env["yaml_dir"].glob("*.yaml"))
        assert len(yaml_files) == 0

    def test_list_includes_created_skill(self, skill_env: dict[str, Path]) -> None:
        from abilities.skill_builder import SkillBuilderAbility
        from collections.abc import Sequence

        ability = SkillBuilderAbility()
        ability.run({
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            })

        result = ability.run({"action": "list"})

        assert result.status == "success"
        assert result.meta["action"] == "list"
        # list now returns structured JSON rows, not prose.
        body_rows = cast(Sequence[dict[str, object]], result.body)
        titles = [row["title"] for row in body_rows]
        assert self._TITLE in titles
        mine = next(row for row in body_rows if row["title"] == self._TITLE)
        assert mine["source"] == "user"


# ===========================================================================
# Migrated from test_ability_skills_tool_result.py (TKT-975)
# Ability-specific business-logic tests that pin the TKT-896 regression.
# ===========================================================================


def _fetch_content_from_db(db_path: Path, title: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT content FROM skills WHERE lower(title) = lower(?)", (title,)
        ).fetchone()
        return cast(str, row[0]) if row else None
    finally:
        conn.close()


def _mp_for_skill_test(config: "_PC", db: sqlite3.Connection) -> "_MP":
    from services.message_processor import MessageProcessor
    from tests._tool_result_harness import seed_transcript

    mp = MessageProcessor("manage my skills")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    pc = getattr(config, "policy_channel", None)
    mp.uid = seed_transcript(db, pc.value if pc else "chat", "manage my skills")
    return mp


def _skill_head(rendered: str, tool: str) -> str:
    from tests._tool_result_harness import head
    return head(rendered, tool)


def _skill_body(rendered: str, tool: str) -> object:
    from tests._tool_result_harness import body
    return body(rendered, tool)


@pytest.mark.unit
def test_skill_body_containing_skill_builder_survives_manager_op_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection
) -> None:
    import abilities.skill_builder as sb
    import utils.skills_io as sio
    from abilities._dispatcher import ToolDispatcher
    from configs.channels import SkillSuggestionConfig, UserConfig

    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_DB), str(dest))
    yaml_dir = tmp_path / "user_skills"
    yaml_dir.mkdir()
    monkeypatch.setattr(sio, "SKILLS_DB_PATH", dest)
    monkeypatch.setattr(sio, "USER_SKILLS_DIR", yaml_dir)
    monkeypatch.setattr(sb, "SKILLS_DB_PATH", dest)

    title = "Tool Authoring Reminder"
    content = (
        "1. Use `skill_builder` to create the playbook for X.\n"
        "2. Remember: use skill_builder for X, never skill_builder for Y.\n"
        "3. Use `memory` to recall the user's preferences."
    )

    # Create through the real production hot path (skill_builder, user channel).
    builder_mp = _mp_for_skill_test(UserConfig({}), db)
    created = ToolDispatcher(builder_mp).dispatch(
        "skill_builder",
        {
            "action": "create",
            "title": title,
            "use_for": "authoring new skills",
            "content": content,
            "act_summary": "x",
        },
    )
    assert _skill_head(created, "skill_builder").startswith("[skill_builder(status=success")
    assert _fetch_content_from_db(dest, title) == content  # stored verbatim

    # Now run the SYSTEM variant (skill_manager) over the same DB via real dispatch.
    mgr_mp = _mp_for_skill_test(SkillSuggestionConfig(), db)
    listed = ToolDispatcher(mgr_mp).dispatch(
        "skill_manager", {"action": "list", "act_summary": "x"}
    )
    assert _skill_head(listed, "skill_manager").startswith("[skill_manager(status=success")

    # The acceptance bar: content survives the manager operation byte-identical —
    # "skill_builder" was NOT silently rewritten to "skill_manager".
    after = _fetch_content_from_db(dest, title)
    assert after == content
    assert "skill_builder" in (after or "")
    assert "skill_manager" not in (after or "")


# ===========================================================================
# Feature test — SkillSuggestionConfig.get_user_prompt derives real query+trail
# ===========================================================================


@pytest.mark.unit
def test_skill_suggestion_prompt_contains_real_query_and_trail(db: sqlite3.Connection) -> None:
    """SkillSuggestionConfig.get_user_prompt must render the actual user query"""
    from services.transcript_service import Transcript
    from services.act_trail import ActTrail
    from configs.channels import SkillSuggestionConfig
    from services.message_processor import MessageProcessor
    import threading

    # 1. Seed a user transcript row — this is the triggering turn.
    user_query = "Find me a good recipe for pasta carbonara and set a 20-minute timer"
    row_id = Transcript.write_input_row("user", "user", user_query)
    trigger_turn_id = Transcript.turn_id_of_row(row_id)

    # 2. Seed two tool_calls rows anchored to that transcript row.
    ActTrail().record(
        tool_name="web_search",
        params={"query": "pasta carbonara recipe", "act_summary": "searching recipes"},
        result="[web_search(status=success)]\n...\n[end:web_search]",
        transcript_id=row_id,
        summary="Searching for recipes",
    )
    ActTrail().record(
        tool_name="timer",
        params={"duration": 1200, "act_summary": "starting timer"},
        result="[timer(status=success)]\n...\n[end:timer]",
        transcript_id=row_id,
        summary="Starting 20-min timer",
    )

    # 3. Build the suggestion MP exactly as _run_suggestion_processor does.
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, user_query, None)
    mp.config = SkillSuggestionConfig()
    mp.uid = None
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    setattr(mp, "_trigger_channel", "user")
    setattr(mp, "_trigger_turn_id", trigger_turn_id)

    # 4. Call the real get_user_prompt.
    prompt = mp.config.get_user_prompt(mp)

    # 5. Assert both the user query and the tool-call trail are present — not empty.
    assert user_query in prompt, (
        f"Original user query not found in suggestion prompt.\nPrompt:\n{prompt}"
    )
    assert "web_search" in prompt, (
        f"web_search tool call not found in suggestion prompt.\nPrompt:\n{prompt}"
    )
    assert "timer" in prompt, (
        f"timer tool call not found in suggestion prompt.\nPrompt:\n{prompt}"
    )
    assert "2 iterations" in prompt, (
        f"Iteration count not found in suggestion prompt.\nPrompt:\n{prompt}"
    )


# ===========================================================================
# Feature tests — action=read surfaces content; skills_building loop self-halts
# ===========================================================================


@pytest.mark.unit
def test_read_returns_full_content_that_list_omits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection
) -> None:
    """action=read surfaces a skill's full step content — which action=list omits —
    so the suggestion model can inspect a candidate's steps before action=edit."""
    from collections.abc import Sequence

    import abilities.skill_builder as sb
    import utils.skills_io as sio
    from abilities._dispatcher import ToolDispatcher
    from configs.channels import SkillSuggestionConfig
    from tests._tool_result_harness import parse_body

    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_DB), str(dest))
    yaml_dir = tmp_path / "user_skills"
    yaml_dir.mkdir()
    monkeypatch.setattr(sio, "SKILLS_DB_PATH", dest)
    monkeypatch.setattr(sio, "USER_SKILLS_DIR", yaml_dir)
    monkeypatch.setattr(sb, "SKILLS_DB_PATH", dest)

    title = "Morning Briefing Routine"
    content = (
        "1. Use `memory` to recall the user's briefing preferences.\n"
        "2. Use `web_search` to fetch the latest headlines.\n"
        "3. Use `weather` to fetch today's forecast."
    )

    # Create through the real dispatch hot path.
    ToolDispatcher(_mp_for_skill_test(SkillSuggestionConfig(), db)).dispatch(
        "skill_manager",
        {
            "action": "create",
            "title": title,
            "use_for": "delivering a morning briefing",
            "content": content,
            "act_summary": "x",
        },
    )

    # read returns the full content + use_for...
    read_rendered = ToolDispatcher(_mp_for_skill_test(SkillSuggestionConfig(), db)).dispatch(
        "skill_manager", {"action": "read", "title": title, "act_summary": "x"}
    )
    assert _skill_head(read_rendered, "skill_manager").startswith("[skill_manager(status=success")
    read_body = cast("dict[str, object]", parse_body(read_rendered, "skill_manager"))
    assert read_body["content"] == content
    assert read_body["use_for"] == "delivering a morning briefing"

    # ...whereas list omits content entirely (titles + use_for only).
    list_rendered = ToolDispatcher(_mp_for_skill_test(SkillSuggestionConfig(), db)).dispatch(
        "skill_manager", {"action": "list", "act_summary": "x"}
    )
    list_body = cast("Sequence[dict[str, object]]", parse_body(list_rendered, "skill_manager"))
    mine = next(r for r in list_body if r["title"] == title and r["source"] == "user")
    assert "content" not in mine


@pytest.mark.unit
def test_skills_building_create_halts_loop_other_channels_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection
) -> None:
    """A successful create on the background suggestion channel sets the turn's
    cancel_event — the signal _step checks to stop recursing — so the loop saves
    exactly one skill. The same create on the user channel must NOT halt the
    user's turn."""
    import abilities.skill_builder as sb
    import utils.skills_io as sio
    from abilities._dispatcher import ToolDispatcher
    from configs.channels import SkillSuggestionConfig, UserConfig

    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_DB), str(dest))
    yaml_dir = tmp_path / "user_skills"
    yaml_dir.mkdir()
    monkeypatch.setattr(sio, "SKILLS_DB_PATH", dest)
    monkeypatch.setattr(sio, "USER_SKILLS_DIR", yaml_dir)
    monkeypatch.setattr(sb, "SKILLS_DB_PATH", dest)

    # Background suggestion loop (skill_manager on skills_building).
    mgr_mp = _mp_for_skill_test(SkillSuggestionConfig(), db)
    assert not mgr_mp.cancel_event.is_set()
    rendered = ToolDispatcher(mgr_mp).dispatch(
        "skill_manager",
        {
            "action": "create",
            "title": "Background Briefing Routine",
            "use_for": "delivering a morning briefing",
            "content": "1. Use `memory` to recall preferences.\n2. Use `weather` to fetch the forecast.",
            "act_summary": "x",
        },
    )
    assert _skill_head(rendered, "skill_manager").startswith("[skill_manager(status=success")
    assert mgr_mp.cancel_event.is_set()  # loop stops after the first write

    # Same create via the user-facing skill_builder on the user channel.
    user_mp = _mp_for_skill_test(UserConfig({}), db)
    assert not user_mp.cancel_event.is_set()
    rendered_user = ToolDispatcher(user_mp).dispatch(
        "skill_builder",
        {
            "action": "create",
            "title": "User Built Skill",
            "use_for": "a user-authored playbook",
            "content": "1. Use `memory` to recall context.",
            "act_summary": "x",
        },
    )
    assert _skill_head(rendered_user, "skill_builder").startswith("[skill_builder(status=success")
    assert not user_mp.cancel_event.is_set()  # user's turn keeps running

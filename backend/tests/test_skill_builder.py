"""Feature tests for TKT-596: SkillBuilderAbility and skills_io helpers.

Two modules are covered:

1. utils/skills_io.py — pure functions: slugify_title, skill_yaml_path.
   Tests are unit-marked (no IO, no collaborators, deterministic).

2. abilities/skill_builder.py — SkillBuilderAbility.execute() lifecycle tests.
   Tests are integration-marked; they run against a real temporary skills.sqlite
   (copied from the production DB and redirected via module-attribute reassignment
   — only the file path is redirected, not any business logic).

Redirection strategy:
  SKILLS_DB_PATH and USER_SKILLS_DIR are module-level Path constants imported
  directly into skill_builder.py. Both modules must be patched so that:
    - The .exists() guard in _handle_* passes against the temp DB copy.
    - open_skills_db() and skill_yaml_path() write to tmp directories.
  Only Path constants are redirected; all SQL, YAML-write, embedding, and
  search-index logic runs through real production code paths.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from services.file_mapper_service import FileMapperService

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
    """Copy the real skills.sqlite into tmp_path; return the copy path."""
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_DB), str(dest))
    return dest


def _skill_count(db_path: Path, source: str = "user") -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM skills WHERE source = ?", (source,)
        ).fetchone()
        return row[0]
    finally:
        conn.close()


def _get_skill_row(db_path: Path, title: str) -> dict | None:
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
def skill_env(tmp_path, monkeypatch):
    """
    Provide a clean, isolated skill environment for each integration test.

    - Copies the real skills.sqlite to tmp_path.
    - Redirects SKILLS_DB_PATH and USER_SKILLS_DIR in both utils.skills_io
      and abilities.skill_builder so all production code writes to tmp_path.
    - The redirect touches only Path constants (infrastructure), not any
      business logic — no production function is mocked.
    """
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
    """slugify_title converts titles to safe filename slugs."""

    def test_spaces_become_hyphens(self):
        from utils.skills_io import slugify_title

        assert slugify_title("Track Package Delivery") == "track-package-delivery"

    def test_special_chars_become_hyphens(self):
        from utils.skills_io import slugify_title

        assert slugify_title("Morning & Evening!") == "morning-evening"

    def test_truncated_at_64_chars(self):
        from utils.skills_io import slugify_title

        long_title = "a" * 100
        result = slugify_title(long_title)
        assert len(result) == 64

    def test_leading_trailing_hyphens_stripped(self):
        from utils.skills_io import slugify_title

        result = slugify_title("  -- hello world --  ")
        assert not result.startswith("-")
        assert not result.endswith("-")
        assert "hello" in result


@pytest.mark.unit
class TestSkillYamlPath:
    """skill_yaml_path returns the correct YAML file path for a title."""

    def test_returns_path_in_user_skills_dir(self):
        from utils.skills_io import USER_SKILLS_DIR, skill_yaml_path

        path = skill_yaml_path("My Custom Skill")
        assert path.parent == USER_SKILLS_DIR

    def test_filename_is_slugified_title_with_yaml_extension(self):
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
    """create action returns error tags when required fields are missing."""

    def test_missing_title_returns_error_tag(self, skill_env):
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        result = ability.execute("test", {"action": "create"}, None)

        assert "error=title-required" in result["text"]

    def test_missing_use_for_returns_error_tag(self, skill_env):
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        result = ability.execute("test", {"action": "create", "title": "My Skill"}, None)

        assert "error=use_for-required" in result["text"]

    def test_missing_content_returns_error_tag(self, skill_env):
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        result = ability.execute(
            "test",
            {"action": "create", "title": "My Skill", "use_for": "tracking things"},
            None,
        )

        assert "error=content-required" in result["text"]


@pytest.mark.integration
class TestSkillBuilderLifecycle:
    """Full create → edit → delete lifecycle against a real skills.sqlite copy."""

    _TITLE = "Test Weekly Review"
    _USE_FOR = "conducting structured weekly reviews"
    _CONTENT = "1. Summarise completed tasks.\n2. Plan next week."

    def test_create_inserts_row_and_writes_yaml(self, skill_env):
        """create action: new user skill row appears in DB and YAML file is written."""
        from abilities.skill_builder import SkillBuilderAbility

        db_path = skill_env["db_path"]
        yaml_dir = skill_env["yaml_dir"]
        before = _skill_count(db_path)

        ability = SkillBuilderAbility()
        result = ability.execute(
            "test",
            {
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
                "tags": "productivity, planning",
            },
            None,
        )

        assert "status=ok" in result["text"]
        assert "action=create" in result["text"]
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

    def test_create_duplicate_title_returns_error(self, skill_env):
        """Creating a skill whose title already exists returns a skill-already-exists error."""
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        params = {
            "action": "create",
            "title": self._TITLE,
            "use_for": self._USE_FOR,
            "content": self._CONTENT,
        }
        ability.execute("test", params, None)
        result = ability.execute("test", params, None)

        assert "skill-already-exists" in result["text"]

    def test_edit_increments_version_and_updates_content(self, skill_env):
        """edit action: version increments and new content is stored in DB and YAML."""
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        ability.execute(
            "test",
            {
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            },
            None,
        )

        new_content = "1. Summarise tasks.\n2. Review blockers.\n3. Plan next week."
        result = ability.execute(
            "test",
            {"action": "edit", "title": self._TITLE, "content": new_content},
            None,
        )

        assert "status=ok" in result["text"]
        assert "action=edit" in result["text"]

        row = _get_skill_row(skill_env["db_path"], self._TITLE)
        assert row["version"] == 2
        assert row["content"] == new_content

    def test_delete_removes_row_and_yaml_file(self, skill_env):
        """delete action: DB row is removed and YAML file is gone."""
        from abilities.skill_builder import SkillBuilderAbility

        db_path = skill_env["db_path"]
        ability = SkillBuilderAbility()
        ability.execute(
            "test",
            {
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            },
            None,
        )

        before = _skill_count(db_path)
        result = ability.execute(
            "test",
            {"action": "delete", "title": self._TITLE},
            None,
        )

        assert "status=ok" in result["text"]
        assert "action=delete" in result["text"]
        assert _skill_count(db_path) == before - 1
        assert _get_skill_row(db_path, self._TITLE) is None

        yaml_files = list(skill_env["yaml_dir"].glob("*.yaml"))
        assert len(yaml_files) == 0

    def test_list_includes_created_skill(self, skill_env):
        """list action: newly created user skill appears in the listing."""
        from abilities.skill_builder import SkillBuilderAbility

        ability = SkillBuilderAbility()
        ability.execute(
            "test",
            {
                "action": "create",
                "title": self._TITLE,
                "use_for": self._USE_FOR,
                "content": self._CONTENT,
            },
            None,
        )

        result = ability.execute("test", {"action": "list"}, None)

        assert "action=list" in result["text"]
        assert self._TITLE in result["text"]
        assert "user" in result["text"]

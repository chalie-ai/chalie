"""Phase 4 dispatch-cutover invariant tests.

Asserts the Phase 4 mechanisms work as designed:
 * pattern_match helpers are plain classes (not Ability subclasses), so even
   when imported they cannot pollute ``Ability.__subclasses__()`` and slip
   into the registry walk.
 * abilities.sqlite — the discovery layer for find_tools — does not contain
   pattern_match entries, so they cannot leak into tool discovery.
 * abilities/ disk layout matches the 15 dispatchable abilities the registry
   walk expects.
"""

import sqlite3
from pathlib import Path

import pytest

import abilities._registry as _reg_module
from abilities._base import Ability

pytestmark = pytest.mark.unit

_BACKEND_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test 1 — pattern_match helpers are plain classes, not Ability subclasses
# ---------------------------------------------------------------------------


def test_pattern_match_helpers_are_not_ability_subclasses():
    """SavePattern / SaveGraph never appear in AbilityRegistry because they
    are plain classes, not Ability subclasses.

    Why this matters: pattern_match helpers are processor-internal — only
    PatternMatchProcessor may invoke them, never the dispatcher / find_tools /
    UMP NATIVE_TOOLS list. If they inherited from Ability, any test (or
    production code) that imported these classes would put them into
    ``Ability.__subclasses__()`` and they would surface as dispatchable
    abilities the next time the registry rebuilt.
    """
    from abilities.pattern_match.save_pattern import SavePattern
    from abilities.pattern_match.save_graph import SaveGraph

    assert not issubclass(SavePattern, Ability), (
        "SavePattern must not inherit from Ability — that would expose it to "
        "the registry walk."
    )
    assert not issubclass(SaveGraph, Ability), (
        "SaveGraph must not inherit from Ability — that would expose it to "
        "the registry walk."
    )

    _reg_module._reset_for_tests()
    try:
        names = {a.NAME for a in _reg_module.AbilityRegistry.all()}
        assert "save_pattern" not in names
        assert "save_graph" not in names

        with pytest.raises(KeyError):
            _reg_module.AbilityRegistry.get("save_pattern")
        with pytest.raises(KeyError):
            _reg_module.AbilityRegistry.get("save_graph")
    finally:
        _reg_module._reset_for_tests()

    # Defense-in-depth: pattern_match must remain a subdirectory so a top-level
    # glob does not pick the modules up regardless of class hierarchy.
    abilities_dir = Path(_reg_module.__file__).resolve().parent
    shallow_py_files = {p.name for p in abilities_dir.glob("*.py")}
    assert "save_pattern.py" not in shallow_py_files
    assert "save_graph.py" not in shallow_py_files
    assert (abilities_dir / "pattern_match").is_dir()


# ---------------------------------------------------------------------------
# Test 2 — abilities.sqlite excludes pattern_match
# ---------------------------------------------------------------------------


def test_abilities_sqlite_excludes_pattern_match():
    """abilities.sqlite contains no row for save_pattern or save_graph.

    The SQLite ability index is the discovery layer for find_tools. If a
    regenerated DB accidentally includes pattern_match entries, those names
    would surface in tool discovery even though the registry excludes them.
    """
    db_path = _BACKEND_DIR / "abilities" / "assets" / "abilities.sqlite"
    assert db_path.exists(), f"abilities.sqlite not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM abilities WHERE name IN ('save_pattern', 'save_graph')"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [], (
        f"abilities.sqlite contains pattern_match entries that must not exist: "
        f"{[r[0] for r in rows]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — abilities/ disk layout has exactly 15 non-underscore top-level modules
# ---------------------------------------------------------------------------

_EXPECTED_ABILITY_MODULE_STEMS = frozenset({
    "browser",
    "code_eval",
    "document",
    "find_tools",
    "goal_pursuit",
    "list",
    "memory",
    "news",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "rich_render",
    "schedule",
    "search",
    "weather",
})


def test_abilities_directory_has_exactly_15_non_underscore_modules():
    """abilities/ contains exactly the 15 Phase 4 top-level modules — no more, no less.

    This mirrors what AbilityRegistry._load() walks: a shallow glob("*.py")
    over abilities/, skipping files that start with "_". The test asserts the
    disk layout, not the runtime registry state, so it is immune to test-
    session contamination from files that import pattern_match abilities.

    If a new ability is intentionally added, update _EXPECTED_ABILITY_MODULE_STEMS
    in this file at the same time.
    """
    abilities_dir = Path(_reg_module.__file__).resolve().parent

    walked = {
        p.stem
        for p in abilities_dir.glob("*.py")
        if not p.name.startswith("_")
    }

    added = walked - _EXPECTED_ABILITY_MODULE_STEMS
    removed = _EXPECTED_ABILITY_MODULE_STEMS - walked

    assert not added, (
        f"Unexpected .py files appeared at abilities/ top level: {sorted(added)}. "
        "Add them to _EXPECTED_ABILITY_MODULE_STEMS if intentional."
    )
    assert not removed, (
        f"Expected ability modules are missing from abilities/: {sorted(removed)}. "
        "Remove them from _EXPECTED_ABILITY_MODULE_STEMS if intentional."
    )
    assert len(walked) == 15

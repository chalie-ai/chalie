"""Phase 4 dispatch-cutover invariant tests."""

import sqlite3
from pathlib import Path

import pytest

import abilities._registry as _reg_module
from abilities._ability import Ability

pytestmark = pytest.mark.unit

_BACKEND_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test 1 — SavePattern / SaveGraph are real, registered Ability subclasses
# ---------------------------------------------------------------------------


def test_save_pattern_save_graph_are_registered_abilities():
    """SavePattern / SaveGraph are first-class Ability subclasses registered"""
    from abilities.save_graph import SaveGraph
    from abilities.save_pattern import SavePattern

    assert issubclass(SavePattern, Ability)
    assert issubclass(SaveGraph, Ability)

    registry_names = {a.get_name() for a in _reg_module.AbilityRegistry.all()}
    assert "save_pattern" in registry_names
    assert "save_graph" in registry_names

    assert isinstance(_reg_module.AbilityRegistry.get("save_pattern"), SavePattern)
    assert isinstance(_reg_module.AbilityRegistry.get("save_graph"), SaveGraph)


# ---------------------------------------------------------------------------
# Test 2 — abilities.sqlite indexes EXACTLY the DISCOVERABLE=True abilities;
#          the non-discoverable pattern-write tools are excluded.
# ---------------------------------------------------------------------------


def test_abilities_sqlite_excludes_non_discoverable_pattern_tools():
    """abilities.sqlite is the find_tools discovery index, built from the"""
    db_path = _BACKEND_DIR / "abilities" / "assets" / "abilities.sqlite"
    assert db_path.exists(), f"abilities.sqlite not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        indexed = {r[0] for r in conn.execute("SELECT name FROM abilities").fetchall()}
    finally:
        conn.close()

    assert "save_pattern" not in indexed and "save_graph" not in indexed, (
        "abilities.sqlite must NOT index the non-discoverable pattern-write tools "
        f"(DISCOVERABLE=False). Indexed pattern tools: "
        f"{sorted(indexed & {'save_pattern', 'save_graph'})}. Rebuild via "
        "`python -m utils.build_ability_db`."
    )

    # Every indexed row must be a DISCOVERABLE=True ability — the index can never
    # carry a non-discoverable tool. (We assert a subset, not equality: when other
    # test modules register DISCOVERABLE test doubles into the live registry, the
    # prebuilt static index legitimately lacks those runtime-only rows.)
    discoverable = _reg_module.AbilityRegistry.discoverable_names()
    assert indexed <= discoverable, (
        "abilities.sqlite indexes a non-discoverable tool — rebuild is stale.\n"
        f"  in index but not discoverable: {sorted(indexed - discoverable)}\n"
        "Rebuild via `python -m utils.build_ability_db`."
    )


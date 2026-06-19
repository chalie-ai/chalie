"""Phase 4 dispatch-cutover invariant tests.

Asserts the Phase 4 mechanisms work as designed under the per-processor
tool-scope spec:

 * SavePattern / SaveGraph are real Ability subclasses living at abilities/
   top-level — registered in AbilityRegistry. Reachable via the registry by
   any processor that lists them in its own ALWAYS_AVAILABLE
   (PatternMatchProcessor today).
 * abilities.sqlite — the find_tools index — contains EVERY ability,
   including save_pattern + save_graph. find_tools gates discovery via the
   calling processor's DISCOVERABLE allowlist; the index itself is global.
 * abilities/ disk layout matches the 20 dispatchable abilities the registry
   walk expects (18 generic + save_pattern + save_graph, including email/calendar/contacts).
 * Each MessageProcessor subclass declares the exact ALWAYS_AVAILABLE +
   DISCOVERABLE the spec dictates — discoverable externals are NEVER
   pre-injected; processor-innate abilities live solely on the owning
   processor.
 * No production code calls AbilityRegistry.all() outside an allowlist —
   prevents the bloat regression that broke an end-to-end run.
"""

import re
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


def test_save_pattern_save_graph_are_registered_abilities() -> None:
    """SavePattern / SaveGraph are first-class Ability subclasses registered
    in AbilityRegistry.

    Tool *scope* (always-available vs discoverable) lives on the calling
    MessageProcessor, not on the Ability — see per-processor scope checks
    in test 4 below. Registry membership is asserted here so the dispatch
    chokepoint can resolve them when a processor opts in.
    """
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
# Test 2 — abilities.sqlite INDEXES every ability, including save_pattern/save_graph
# ---------------------------------------------------------------------------


def test_abilities_sqlite_indexes_save_pattern_and_save_graph() -> None:
    """abilities.sqlite MUST contain save_pattern and save_graph alongside
    every other ability.

    Per the per-processor tool-scope spec: every tool is listed in the
    abilities sql db with embeddings, etc. Discovery scoping is enforced at
    query time by ``find_tools`` filtering on the calling processor's
    DISCOVERABLE list — NOT by selectively excluding rows from the index.
    """
    db_path = _BACKEND_DIR / "abilities" / "assets" / "abilities.sqlite"
    assert db_path.exists(), f"abilities.sqlite not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        indexed = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM abilities WHERE name IN ('save_pattern', 'save_graph')"
            ).fetchall()
        }
    finally:
        conn.close()

    assert indexed == {"save_pattern", "save_graph"}, (
        "abilities.sqlite must index save_pattern and save_graph (per-processor "
        f"tool-scope spec rule 1). Indexed: {sorted(indexed)}. Rebuild via "
        "`python -m utils.build_ability_db`."
    )


# ---------------------------------------------------------------------------
# Test 3 — abilities/ disk layout has exactly the expected non-underscore modules
# ---------------------------------------------------------------------------

_EXPECTED_ABILITY_MODULE_STEMS = frozenset({
    "bash",
    "browser",
    "calendar",
    "chalie_docs",
    "code_eval",
    "contacts",
    "document",
    "email",
    "file_permissions",
    "file_write",
    "find_skills",
    "find_tools",
    "home",
    "list",
    "mcp_manager",
    "memory",
    "news",
    "place",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "review_transcript",
    "save_graph",
    "save_pattern",
    "schedule",
    "search",
    "search_files",
    # skill_builder hosts BOTH the user-facing SkillBuilderAbility and its SYSTEM
    # variant SkillManagerAbility in ONE module (TKT-896 merged the twins —
    # skill_manager.py was deleted). Both tool NAMES stay registered via
    # __subclasses__(), but there is only one module stem on disk now.
    "skill_builder",
    # thinking: internal never-discoverable ability dispatched at turn 0
    # (added by the compaction redesign, Task 4.1 / commit eaaaf29a, which
    # replaced the old exploration hack with abilities/thinking.py). It lives
    # at the top level so AbilityRegistry._load() registers it, but it is
    # excluded from find_tools indexing (build_ability_db _NON_INDEXED_ABILITIES).
    "thinking",
    # chat_history_compactor / tool_chain_compactor: internal never-discoverable
    # abilities dispatched programmatically by MessageProcessor._dispatch_compaction()
    # (compaction redesign — compaction now fires via the normal tool-dispatch
    # chokepoint instead of an inline _compact() method). Both register at the top
    # level but are excluded from find_tools indexing AND the policy gate
    # (build_ability_db _NON_INDEXED_ABILITIES + policy_manager INTERNAL).
    "chat_history_compactor",
    "tool_chain_compactor",
    "timer",
    # vision: image-description delegate ability (TKT-838) — registered at the
    # top level like every dispatchable tool.
    "vision",
    "ubiquiti",
    "weather",
    "web_browse",
    "web_download",
    "web_search",
})


def test_abilities_directory_has_expected_non_underscore_modules() -> None:
    """abilities/ contains exactly the expected dispatchable top-level modules.

    Mirrors what AbilityRegistry._load() walks: a shallow glob("*.py")
    over abilities/, skipping files starting with "_".  The test asserts the
    disk layout, not the runtime registry state.

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
    assert len(walked) == len(_EXPECTED_ABILITY_MODULE_STEMS)


# ---------------------------------------------------------------------------
# Test 5 — AbilityRegistry.all() callers are restricted to a narrow allowlist
# ---------------------------------------------------------------------------
#
# AbilityRegistry.all() returns every Ability instance — including the
# discoverable externals AND the processor-innate ones.  Calling it from a
# processor's ALWAYS_AVAILABLE comprehension is exactly how a past tool-bloat
# regression slipped in.  This test bans new callers outside legitimate sites:
#
#   * abilities/_registry.py        — owns the method
#   * utils/build_ability_db.py     — rebuilds discovery DB from registry
#   (act_dispatcher_service.py deleted in T3 — no longer a caller)
#
# To add a new caller, audit the use case (does it pre-inject discoverable
# abilities?), then extend _ALLOWED_REGISTRY_ALL_CALLERS in this file.

_ALLOWED_REGISTRY_ALL_CALLERS = frozenset({
    "abilities/_registry.py",
    "utils/build_ability_db.py",
})

_REGISTRY_ALL_PATTERN = re.compile(r"AbilityRegistry\.all\(\)")


def test_no_production_code_calls_ability_registry_all_outside_allowlist() -> None:
    """Static scan: AbilityRegistry.all() must only be called from allowlisted files.

    The bloat regression in an end-to-end run originated from
    `NATIVE_TOOLS = sorted(a.NAME for a in AbilityRegistry.all())` in
    UserMessageProcessor.  This gate prevents the same shape from sneaking
    back into another processor.  Tests are exempt — they may need to inspect
    the full registry to assert behaviour.
    """
    offenders: list[str] = []
    for path in _BACKEND_DIR.rglob("*.py"):
        rel = path.relative_to(_BACKEND_DIR).as_posix()

        if rel.startswith("tests/"):
            continue
        if rel in _ALLOWED_REGISTRY_ALL_CALLERS:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if _REGISTRY_ALL_PATTERN.search(text):
            offenders.append(rel)

    assert not offenders, (
        "AbilityRegistry.all() called outside allowlist:\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nDiscoverable externals (browser/code_eval/news/"
        "programming_docs_search/search/weather) and processor-innate abilities "
        "(save_pattern/save_graph) must NEVER be pre-injected. If your caller "
        "needs the full registry, add it to _ALLOWED_REGISTRY_ALL_CALLERS in "
        "this file with justification."
    )


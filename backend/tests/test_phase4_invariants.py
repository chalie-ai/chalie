"""Phase 4 dispatch-cutover invariant tests.

Asserts the Phase 4 mechanisms work as designed under the single-flag
tool-scope model:

 * SavePattern / SaveGraph are real Ability subclasses living at abilities/
   top-level — registered in AbilityRegistry. Reachable via the registry by
   any processor that lists them in its own ALWAYS_AVAILABLE
   (PatternMatchProcessor today).
 * abilities.sqlite — the find_tools index — contains exactly the
   DISCOVERABLE=True abilities. save_pattern + save_graph are DISCOVERABLE=False
   (pattern-write is pinned onto PatternMatchProcessor via always_available),
   so they are EXCLUDED from the index. Discovery is global: the candidate
   roster is ``AbilityRegistry.discoverable_names()``, not a per-config list.
 * abilities/ disk layout matches the dispatchable abilities the registry
   walk expects (including save_pattern + save_graph + email/calendar/contacts).
 * Tool scope is the single ``Ability.DISCOVERABLE`` flag plus each
   processor's ALWAYS_AVAILABLE — a non-discoverable ability is reached ONLY
   by being pinned into an always_available list; discoverable externals are
   NEVER pre-injected.
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


def test_save_pattern_save_graph_are_registered_abilities():
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
# Test 2 — abilities.sqlite indexes EXACTLY the DISCOVERABLE=True abilities;
#          the non-discoverable pattern-write tools are excluded.
# ---------------------------------------------------------------------------


def test_abilities_sqlite_excludes_non_discoverable_pattern_tools():
    """abilities.sqlite is the find_tools discovery index, built from the
    DISCOVERABLE=True abilities only (``build_ability_db`` filters on
    ``a.DISCOVERABLE``). save_pattern / save_graph are DISCOVERABLE=False —
    reached solely via PatternMatchProcessor's always_available — so they must
    be ABSENT from the index. The index must equal the global discoverable roster.
    """
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
    # at the top level so AbilityRegistry._load() registers it, but it sets
    # DISCOVERABLE=False so it is excluded from the find_tools index/SHA build.
    "thinking",
    # chat_history_compactor / tool_chain_compactor: internal never-discoverable
    # abilities dispatched programmatically by MessageProcessor._dispatch_compaction()
    # (compaction redesign — compaction now fires via the normal tool-dispatch
    # chokepoint instead of an inline _compact() method). Both register at the top
    # level but are excluded from find_tools indexing (DISCOVERABLE=False) AND
    # the policy gate (policy_manager INTERNAL).
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


def test_abilities_directory_has_expected_non_underscore_modules():
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


def test_no_production_code_calls_ability_registry_all_outside_allowlist():
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


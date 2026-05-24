"""Phase 4 dispatch-cutover invariant tests.

Asserts the Phase 4 mechanisms work as designed under the per-processor
tool-scope spec:

 * SavePattern / SaveGraph are real Ability subclasses living at abilities/
   top-level — registered in AbilityRegistry. Reachable via the registry by
   any processor that lists them in its own ALWAYS_AVAILABLE
   (PatternMatchProcessor and GeoPatternProcessor today).
 * abilities.sqlite — the find_tools index — contains EVERY ability,
   including save_pattern + save_graph. find_tools gates discovery via the
   calling processor's DISCOVERABLE allowlist; the index itself is global.
 * abilities/ disk layout matches the 25 dispatchable abilities the registry
   walk expects (23 generic + save_pattern + save_graph, including
   email/calendar/contacts/place).
 * Each MessageProcessor subclass declares the exact ALWAYS_AVAILABLE +
   DISCOVERABLE the spec dictates — discoverable externals are NEVER
   pre-injected; processor-innate abilities live solely on the owning
   processor.
 * No production code calls AbilityRegistry.all() outside an allowlist —
   prevents the bloat regression that broke nightly run 346.
"""

import re
import sqlite3
from pathlib import Path

import pytest

import abilities._registry as _reg_module
from abilities._base import Ability
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_BACKEND_DIR = FileMapperService.get_backend_path()


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

    registry_names = {a.NAME for a in _reg_module.AbilityRegistry.all()}
    assert "save_pattern" in registry_names
    assert "save_graph" in registry_names

    assert isinstance(_reg_module.AbilityRegistry.get("save_pattern"), SavePattern)
    assert isinstance(_reg_module.AbilityRegistry.get("save_graph"), SaveGraph)


# ---------------------------------------------------------------------------
# Test 2 — abilities.sqlite INDEXES every ability, including save_pattern/save_graph
# ---------------------------------------------------------------------------


def test_abilities_sqlite_indexes_save_pattern_and_save_graph():
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
    "browser",
    "calendar",
    "chalie_docs",
    "code_eval",
    "contacts",
    "document",
    "email",
    "file_write",
    "find_skills",
    "find_tools",
    "home",
    "list",
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
    "skill_builder",
    "steer",
    "subagent",
    "timer",
    "ubiquiti",
    "weather",
    "web_download",
})


def test_abilities_directory_has_expected_non_underscore_modules():
    """abilities/ contains exactly the expected dispatchable top-level modules.

    Mirrors what AbilityRegistry._load() walks: a shallow glob("*.py")
    over abilities/, skipping files starting with "_".  The test asserts the
    disk layout, not the runtime registry state.  Untracked files are excluded
    so local experiments don't trip the check.

    If a new ability is intentionally added, update _EXPECTED_ABILITY_MODULE_STEMS
    in this file at the same time.
    """
    import subprocess

    abilities_dir = Path(_reg_module.__file__).resolve().parent

    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "--cached", "*.py"],
            cwd=str(abilities_dir),
            text=True,
        ).splitlines()
    )

    walked = {
        p.stem
        for p in abilities_dir.glob("*.py")
        if not p.name.startswith("_") and p.name in tracked
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
# Test 4 — per-processor ALWAYS_AVAILABLE / DISCOVERABLE matches the spec
# ---------------------------------------------------------------------------
#
# Every processor declares two ClassVars:
#   * ALWAYS_AVAILABLE — list of tool names pre-injected into the native
#     tools array on every iteration.
#   * DISCOVERABLE     — list of tool names find_tools is allowed to surface
#     for this processor.  find_tools filters its DB query by this list.
#
# The bug this test prevents: pre-injecting discoverable externals
# (browser, code_eval, news, programming_docs_search, search, weather) into
# a processor's ALWAYS_AVAILABLE that should not own them.  Pre-injection
# bloated tools arrays from ~5kB to ~12.8kB and produced Ollama 500 errors +
# model hallucinations in nightly run 346.

_DEFAULT_ALWAYS = frozenset({"find_skills", "find_tools", "memory"})

_DEFAULT_DISCOVERABLE = frozenset({
    "browser", "calendar", "chalie_docs", "code_eval", "contacts", "document",
    "email", "file_write", "home", "list", "news", "place", "programming_docs_search", "read",
    "review_tool_calls", "review_transcript", "schedule", "search", "search_files",
    "skill_builder", "subagent", "timer", "ubiquiti", "weather", "web_download",
})


_EXPECTED_SCOPE: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # Post tool-tier refactor (67df5d1e): UMP, DMN, and Subagent all inherit
    # the base class defaults — {find_tools, memory} ALWAYS, full set DISCOVERABLE.
    "UserMessageProcessor":      (_DEFAULT_ALWAYS, _DEFAULT_DISCOVERABLE),
    "DMNMessageProcessor":       (_DEFAULT_ALWAYS, _DEFAULT_DISCOVERABLE),
    "SubagentProcessor":         (_DEFAULT_ALWAYS, _DEFAULT_DISCOVERABLE),
    # PMP and GPP own save_pattern + save_graph; nothing discoverable.
    "PatternMatchProcessor":     (frozenset({"save_pattern", "save_graph"}), frozenset()),
    "GeoPatternProcessor":       (frozenset({"save_pattern", "save_graph"}), frozenset()),
    # Background / no tools at all.
    "ContinuityCompactionProcessor":      (frozenset(), frozenset()),
    "SubagentTrailCompactionProcessor":   (frozenset(), frozenset()),
    "EpisodeEncoderProcessor":            (frozenset(), frozenset()),
    "SuperEpisodeEncoderProcessor": (frozenset(), frozenset()),
    "UserSummaryProcessor":        (frozenset(), frozenset()),
}

# Processors that legitimately own save_pattern/save_graph as innate abilities.
_INNATE_OWNING_PROCESSORS = frozenset({"PatternMatchProcessor", "GeoPatternProcessor"})


def test_per_processor_tool_scope_matches_spec():
    """Each MessageProcessor subclass's ALWAYS_AVAILABLE + DISCOVERABLE must
    match the spec exactly.

    Discoverable externals MUST NOT appear in any processor's
    ALWAYS_AVAILABLE — they are surfaced exclusively through find_tools,
    which gates on DISCOVERABLE.  Processor-innate abilities (save_pattern,
    save_graph) MUST NOT appear in any processor's lists except the owning
    ones (PMP + GPP today).
    """
    from services.compaction_message_processor import (
        ContinuityCompactionProcessor,
        SubagentTrailCompactionProcessor,
    )
    from services.dmn_message_processor import DMNMessageProcessor
    from services.episode_encoder_processor import EpisodeEncoderProcessor
    from services.geo_pattern_processor import GeoPatternProcessor
    from services.pattern_match_processor import PatternMatchProcessor
    from services.subagent_processor import SubagentProcessor
    from services.super_episode_encoder_processor import SuperEpisodeEncoderProcessor
    from services.user_message_processor import UserMessageProcessor
    from services.user_summary_processor import UserSummaryProcessor

    processors = {
        "UserMessageProcessor": UserMessageProcessor,
        "DMNMessageProcessor": DMNMessageProcessor,
        "SubagentProcessor": SubagentProcessor,
        "PatternMatchProcessor": PatternMatchProcessor,
        "GeoPatternProcessor": GeoPatternProcessor,
        "ContinuityCompactionProcessor": ContinuityCompactionProcessor,
        "SubagentTrailCompactionProcessor": SubagentTrailCompactionProcessor,
        "EpisodeEncoderProcessor": EpisodeEncoderProcessor,
        "SuperEpisodeEncoderProcessor": SuperEpisodeEncoderProcessor,
        "UserSummaryProcessor": UserSummaryProcessor,
    }

    innate = frozenset({"save_pattern", "save_graph"})

    for name, cls in processors.items():
        always = frozenset(cls.ALWAYS_AVAILABLE)
        discoverable = frozenset(cls.DISCOVERABLE)
        expected_always, expected_discoverable = _EXPECTED_SCOPE[name]

        leaked_externals = always & _DEFAULT_DISCOVERABLE
        assert not leaked_externals, (
            f"{name}.ALWAYS_AVAILABLE leaks discoverable externals "
            f"{sorted(leaked_externals)}. These MUST be surfaced via find_tools "
            "only."
        )

        if name not in _INNATE_OWNING_PROCESSORS:
            leaked_innate = (always | discoverable) & innate
            assert not leaked_innate, (
                f"{name} leaks processor-innate abilities {sorted(leaked_innate)}. "
                "Only PatternMatchProcessor and GeoPatternProcessor own these today."
            )

        assert always == expected_always, (
            f"{name}.ALWAYS_AVAILABLE drifted from spec.\n"
            f"  expected: {sorted(expected_always)}\n"
            f"  actual:   {sorted(always)}\n"
            f"  added:    {sorted(always - expected_always)}\n"
            f"  removed:  {sorted(expected_always - always)}\n"
            "Update processor-tool-scope.md if intentional."
        )
        assert discoverable == expected_discoverable, (
            f"{name}.DISCOVERABLE drifted from spec.\n"
            f"  expected: {sorted(expected_discoverable)}\n"
            f"  actual:   {sorted(discoverable)}\n"
            f"  added:    {sorted(discoverable - expected_discoverable)}\n"
            f"  removed:  {sorted(expected_discoverable - discoverable)}\n"
            "Update processor-tool-scope.md if intentional."
        )


# ---------------------------------------------------------------------------
# Test 5 — AbilityRegistry.all() callers are restricted to a narrow allowlist
# ---------------------------------------------------------------------------
#
# AbilityRegistry.all() returns every Ability instance — including the
# discoverable externals AND the processor-innate ones.  Calling it from a
# processor's ALWAYS_AVAILABLE comprehension is exactly how the run-346 bloat
# regression slipped in.  This test bans new callers outside legitimate sites:
#
#   * abilities/_registry.py        — owns the method
#   * utils/build_ability_db.py     — rebuilds discovery DB from registry
#   * services/act_dispatcher_service.py — populates dispatcher handlers
#
# To add a new caller, audit the use case (does it pre-inject discoverable
# abilities?), then extend _ALLOWED_REGISTRY_ALL_CALLERS in this file.

_ALLOWED_REGISTRY_ALL_CALLERS = frozenset({
    "abilities/_registry.py",
    "utils/build_ability_db.py",
    "services/act_dispatcher_service.py",
    "services/policy_service.py",
})

_REGISTRY_ALL_PATTERN = re.compile(r"AbilityRegistry\.all\(\)")


def test_no_production_code_calls_ability_registry_all_outside_allowlist():
    """Static scan: AbilityRegistry.all() must only be called from allowlisted files.

    The bloat regression in nightly run 346 originated from
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

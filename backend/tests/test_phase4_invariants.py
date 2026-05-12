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
   prevents the bloat regression that broke nightly run 346.
"""

import re
import sqlite3
from pathlib import Path

import pytest

import abilities._registry as _reg_module
from abilities._base import Ability

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
# Test 3 — abilities/ disk layout has exactly 17 non-underscore top-level modules
# ---------------------------------------------------------------------------

_EXPECTED_ABILITY_MODULE_STEMS = frozenset({
    "browser",
    "calendar",
    "code_eval",
    "contacts",
    "document",
    "email",
    "find_tools",
    "list",
    "memory",
    "news",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "save_graph",
    "save_pattern",
    "schedule",
    "search",
    "subagent",
    "timer",
    "weather",
})


def test_abilities_directory_has_exactly_20_non_underscore_modules():
    """abilities/ contains exactly the 20 dispatchable top-level modules.

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
    assert len(walked) == 20


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

_DEFAULT_ALWAYS = frozenset({
    "document", "find_tools", "list", "memory",
    "read", "review_tool_calls", "schedule", "timer",
})

# `subagent` is discoverable in UMP only — DMN/Scheduled/Subagent processors
# must not spawn further subagents, so they subtract it explicitly. Listing it
# here keeps the leak-check honest: any processor that pre-injects `subagent`
# into ALWAYS_AVAILABLE will trip the assertion in
# test_per_processor_tool_scope_matches_spec.
#
# email/calendar/contacts are discoverable in UMP (personal data tools) and
# DMN (background research may need calendar/contacts context), but NOT in
# Scheduled or Subagent processors — those run headless without personal
# data access.
_DEFAULT_DISCOVERABLE = frozenset({
    "browser", "calendar", "code_eval", "contacts", "email",
    "news", "programming_docs_search", "search", "subagent", "weather",
})

# Capability tools (email/calendar/contacts) are personal-data tools — they
# must not be surfaced in headless background processors (Scheduled, Subagent).
_CAPABILITY_TOOLS = frozenset({"email", "calendar", "contacts"})


_EXPECTED_SCOPE: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "UserMessageProcessor":      (_DEFAULT_ALWAYS, _DEFAULT_DISCOVERABLE),
    # DMN has news, search, browser natively per masterplan §4 — no find_tools round-trip.
    # `timer` is dropped: DMN runs in the background without a user-channel
    # surface, so the rich-media card would never render. `subagent` is
    # excluded everywhere except UMP so background processes cannot spawn
    # nested subagents. email/calendar/contacts remain discoverable for DMN
    # — background reflection needs access to personal data context.
    "DMNMessageProcessor":       ((_DEFAULT_ALWAYS - {"timer"}) | {"news", "search", "browser"},
                                   _DEFAULT_DISCOVERABLE - {"news", "search", "browser", "subagent"}),
    # Scheduled has no UI surface — drop `timer`, `schedule`, and personal-data
    # capability tools (email/calendar/contacts are user-initiated only).
    "ScheduledMessageProcessor": (_DEFAULT_ALWAYS - {"schedule", "timer"},
                                   _DEFAULT_DISCOVERABLE - {"subagent"} - _CAPABILITY_TOOLS),
    # SubagentProcessor ALWAYS_AVAILABLE is set per-instance (from agent_type);
    # the class-level attribute is [] (empty). The per-instance value is
    # verified separately in test_subagent_processor.py::test_per_instance_always_available_is_set_from_agent_type.
    # Subagent does not get capability tools — subagents are task-scoped, not
    # user-personal-data-scoped.
    "SubagentProcessor":         (frozenset(), frozenset(
        (set(_DEFAULT_DISCOVERABLE) - {"subagent"} - _CAPABILITY_TOOLS)
        | {"document", "list", "memory", "read", "review_tool_calls", "schedule"}
    )),
    # PMP owns save_pattern + save_graph today; nothing discoverable.
    "PatternMatchProcessor":     (frozenset({"save_pattern", "save_graph"}), frozenset()),
    # Background / no tools at all.
    "ContinuityCompactionProcessor":      (frozenset(), frozenset()),
    "SubagentTrailCompactionProcessor":   (frozenset(), frozenset()),
    "EpisodeEncoderProcessor":            (frozenset(), frozenset()),
    "SuperEpisodeEncoderProcessor": (frozenset(), frozenset()),
    "UserSummaryProcessor":        (frozenset(), frozenset()),
}


def test_per_processor_tool_scope_matches_spec():
    """Each MessageProcessor subclass's ALWAYS_AVAILABLE + DISCOVERABLE must
    match the spec exactly.

    Discoverable externals MUST NOT appear in any processor's
    ALWAYS_AVAILABLE — they are surfaced exclusively through find_tools,
    which gates on DISCOVERABLE.  Processor-innate abilities (save_pattern,
    save_graph) MUST NOT appear in any processor's lists except the owning
    one (PMP today).
    """
    from services.compaction_message_processor import (
        ContinuityCompactionProcessor,
        SubagentTrailCompactionProcessor,
    )
    from services.dmn_message_processor import DMNMessageProcessor
    from services.episode_encoder_processor import EpisodeEncoderProcessor
    from services.pattern_match_processor import PatternMatchProcessor
    from services.scheduled_message_processor import ScheduledMessageProcessor
    from services.subagent_processor import SubagentProcessor
    from services.super_episode_encoder_processor import SuperEpisodeEncoderProcessor
    from services.user_message_processor import UserMessageProcessor
    from services.user_summary_processor import UserSummaryProcessor

    processors = {
        "UserMessageProcessor": UserMessageProcessor,
        "DMNMessageProcessor": DMNMessageProcessor,
        "ScheduledMessageProcessor": ScheduledMessageProcessor,
        "SubagentProcessor": SubagentProcessor,
        "PatternMatchProcessor": PatternMatchProcessor,
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

        # DMNMessageProcessor has news/search/browser promoted to
        # ALWAYS_AVAILABLE per masterplan §4 — not a leak, by design.
        approved_always_externals = {"news", "search", "browser"} if name == "DMNMessageProcessor" else set()
        leaked_externals = always & (_DEFAULT_DISCOVERABLE - approved_always_externals)
        assert not leaked_externals, (
            f"{name}.ALWAYS_AVAILABLE leaks discoverable externals "
            f"{sorted(leaked_externals)}. These MUST be surfaced via find_tools "
            "only."
        )

        if name != "PatternMatchProcessor":
            leaked_innate = (always | discoverable) & innate
            assert not leaked_innate, (
                f"{name} leaks processor-innate abilities {sorted(leaked_innate)}. "
                "Only PatternMatchProcessor owns these today."
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
#   * services/self_model_service.py — innate-skills inventory for /self_model
#
# To add a new caller, audit the use case (does it pre-inject discoverable
# abilities?), then extend _ALLOWED_REGISTRY_ALL_CALLERS in this file.

_ALLOWED_REGISTRY_ALL_CALLERS = frozenset({
    "abilities/_registry.py",
    "utils/build_ability_db.py",
    "services/act_dispatcher_service.py",
    "services/self_model_service.py",
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

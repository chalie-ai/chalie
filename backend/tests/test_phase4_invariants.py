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

# Base class (MessageProcessor) defaults — UMP, DMN, EAMP all inherit these
# without overriding on rc-0.8.0.
_BASE_ALWAYS = frozenset({"find_skills", "find_tools", "memory"})
_BASE_DISCOVERABLE = frozenset({
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
    "home",
    "list",
    "news",
    "place",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "review_transcript",
    "schedule",
    "search",
    "search_files",
    "skill_builder",
    "subagent",
    "timer",
    "ubiquiti",
    "weather",
    "web_download",
})


_EXPECTED_SCOPE: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "UserMessageProcessor":      (_BASE_ALWAYS, _BASE_DISCOVERABLE),
    "DMNMessageProcessor":       (_BASE_ALWAYS, _BASE_DISCOVERABLE),
    # SubagentProcessor is excluded from class-level checks.  Its __init__
    # overrides self.ALWAYS_AVAILABLE per-instance from SUBAGENT_TYPES —
    # the class attribute is the inherited base value and is never operative.
    # See test_subagent_processor_instance_scope below.
    # PMP owns save_pattern + save_graph today; nothing discoverable.
    "PatternMatchProcessor":     (frozenset({"save_pattern", "save_graph"}), frozenset()),
    # Background / no tools at all.
    "ContinuityCompactionProcessor":      (frozenset(), frozenset()),
    "SubagentTrailCompactionProcessor":   (frozenset(), frozenset()),
    "EpisodeEncoderProcessor":            (frozenset(), frozenset()),
    "SuperEpisodeEncoderProcessor": (frozenset(), frozenset()),
    "UserSummaryProcessor":        (frozenset(), frozenset()),
    # SkillSuggestionMessageProcessor runs a background ACT loop with only
    # skill_builder available — its sole purpose is to analyse completed trails
    # and optionally create a reusable skill. Nothing is discoverable.
    "SkillSuggestionMessageProcessor": (frozenset({"skill_builder"}), frozenset()),
    # ExternalAgentMessageProcessor inherits ALWAYS_AVAILABLE and DISCOVERABLE
    # from the MessageProcessor base class without any overrides.
    "ExternalAgentMessageProcessor": (_BASE_ALWAYS, _BASE_DISCOVERABLE),
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
    from services.external_agent_message_processor import ExternalAgentMessageProcessor
    from services.pattern_match_processor import PatternMatchProcessor
    from services.skill_suggestion_message_processor import SkillSuggestionMessageProcessor
    from services.super_episode_encoder_processor import SuperEpisodeEncoderProcessor
    from services.user_message_processor import UserMessageProcessor
    from services.user_summary_processor import UserSummaryProcessor

    processors = {
        "UserMessageProcessor": UserMessageProcessor,
        "DMNMessageProcessor": DMNMessageProcessor,
        "PatternMatchProcessor": PatternMatchProcessor,
        "ContinuityCompactionProcessor": ContinuityCompactionProcessor,
        "SubagentTrailCompactionProcessor": SubagentTrailCompactionProcessor,
        "EpisodeEncoderProcessor": EpisodeEncoderProcessor,
        "SuperEpisodeEncoderProcessor": SuperEpisodeEncoderProcessor,
        "UserSummaryProcessor": UserSummaryProcessor,
        "SkillSuggestionMessageProcessor": SkillSuggestionMessageProcessor,
        "ExternalAgentMessageProcessor": ExternalAgentMessageProcessor,
    }

    innate = frozenset({"save_pattern", "save_graph"})

    for name, cls in processors.items():
        always = frozenset(cls.ALWAYS_AVAILABLE)
        discoverable = frozenset(cls.DISCOVERABLE)
        expected_always, expected_discoverable = _EXPECTED_SCOPE[name]

        # Processor-innate tools (save_pattern, save_graph) must not appear
        # in any processor's ALWAYS or DISCOVERABLE except PMP.
        # The old "leaked externals" check is obsolete on rc-0.8.0 since
        # all processors now use the base {find_skills, find_tools, memory}.

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


# ---------------------------------------------------------------------------
# Test 6 — SubagentProcessor instance-level scope (TKT-609)
# ---------------------------------------------------------------------------


def test_subagent_processor_instance_scope():
    """SubagentProcessor.__init__ sets self.ALWAYS_AVAILABLE from SUBAGENT_TYPES.

    The class-level attribute is inherited from MessageProcessor and is not
    operative.  After construction, each instance's ALWAYS_AVAILABLE must
    equal the ``native_tools`` list for its agent_type.
    """
    from abilities.subagent import SUBAGENT_TYPES
    from services.subagent_processor import SubagentProcessor

    for agent_type, entry in SUBAGENT_TYPES.items():
        proc = SubagentProcessor(raw_input="x", agent_type=agent_type)
        expected = list(entry["native_tools"])
        assert proc.ALWAYS_AVAILABLE == expected, (
            f"SubagentProcessor(agent_type={agent_type!r}): "
            f"instance ALWAYS_AVAILABLE {proc.ALWAYS_AVAILABLE} "
            f"!= native_tools {expected}"
        )


def test_subagent_processor_class_attribute_is_base_value():
    """The class-level ALWAYS_AVAILABLE on SubagentProcessor is the inherited
    base value (find_skills, find_tools, memory), not any type-specific list.
    """
    from services.message_processor import MessageProcessor
    from services.subagent_processor import SubagentProcessor

    class_always = frozenset(SubagentProcessor.ALWAYS_AVAILABLE)
    base_always = frozenset(MessageProcessor.ALWAYS_AVAILABLE)

    assert class_always == base_always, (
        "SubagentProcessor.ALWAYS_AVAILABLE (class attr) should be the "
        f"inherited base value {base_always}; got {class_always}"
    )

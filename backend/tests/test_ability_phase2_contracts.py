"""ABC contract tests for the 14 Phase 2 Ability ports.

Verifies that each concrete Ability subclass satisfies the full ABC contract
and carries the ClassVars mandated by the codebase plan. No execute() calls
are made — those are covered by existing legacy tests on the innate-skill and
tool paths that remain wired in Phase 2.

HTTP boundaries are NOT mocked here because execute() is never called. The
test suite is 100% class-introspection: import the class, inspect ClassVars,
query the registry.

Tool *scope* (always-available vs discoverable) lives on the calling
MessageProcessor, not on the Ability itself — see test_phase4_invariants.py
for those checks. This file only verifies ABC structural contracts:
NAME / SUMMARY / EXAMPLES / INPUT_SCHEMA / TIMEOUT / execute().

  Note for read: _URL_FETCH_TIMEOUT ClassVar = 15 (the HTTP timeout), distinct
  from the ABC TIMEOUT attribute which controls overall ability dispatch.
"""

import gc
import importlib
import sys

import pytest

from abilities._base import Ability
from abilities._registry import AbilityRegistry, _reset_for_tests

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Registry isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the registry before and after every test.

    gc.collect() flushes any locally-scoped subclasses from other test files
    so the lazy load starts from a predictable baseline.
    """
    gc.collect()
    _reset_for_tests()
    yield
    gc.collect()
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import(module_name: str):
    """Import abilities.<module_name>, reloading if already cached."""
    full = f"abilities.{module_name}"
    if full in sys.modules:
        return sys.modules[full]
    return importlib.import_module(full)


def _assert_contract(cls, *, name, timeout):
    """Assert the full ABC contract for *cls*."""
    # Required ClassVars exist and are non-empty
    assert isinstance(cls.NAME, str) and cls.NAME, f"{cls.__name__}.NAME is empty"
    assert isinstance(cls.SUMMARY, str) and cls.SUMMARY, f"{cls.__name__}.SUMMARY is empty"
    assert isinstance(cls.EXAMPLES, list), f"{cls.__name__}.EXAMPLES is not a list"
    assert all(isinstance(e, str) for e in cls.EXAMPLES), f"{cls.__name__}.EXAMPLES must be list[str]"
    assert isinstance(cls.INPUT_SCHEMA, dict), f"{cls.__name__}.INPUT_SCHEMA is not a dict"

    # EXAMPLES count
    assert 6 <= len(cls.EXAMPLES) <= 8, (
        f"{cls.__name__}.EXAMPLES has {len(cls.EXAMPLES)} entries — must be 6-8"
    )

    # NAME matches expected
    assert cls.NAME == name, f"{cls.__name__}.NAME={cls.NAME!r}, expected {name!r}"

    # TIMEOUT matches expected
    assert cls.TIMEOUT == timeout, (
        f"{cls.__name__}.TIMEOUT={cls.TIMEOUT}, expected {timeout}"
    )

    # execute() exists and is callable
    assert callable(getattr(cls, "execute", None)), (
        f"{cls.__name__} has no callable execute()"
    )

    # Properly inherits from Ability
    assert issubclass(cls, Ability), f"{cls.__name__} is not a subclass of Ability"


def _assert_registered(name: str, expected_cls: type):
    """Assert AbilityRegistry.get(name) returns an instance of expected_cls."""
    instance = AbilityRegistry.get(name)
    assert isinstance(instance, expected_cls), (
        f"Registry.get({name!r}) returned {type(instance)}, expected {expected_cls}"
    )


# ===========================================================================
# browser
# ===========================================================================


def test_browser_contract():
    """BrowserAbility satisfies the ABC when Playwright is installed.

    If Playwright is absent the class is skipped by the ImportError guard —
    the module import itself must not raise in either case.
    """
    mod = _import("browser")
    if not hasattr(mod, "BrowserAbility"):
        pytest.skip("Playwright not installed — BrowserAbility guard active")

    cls = mod.BrowserAbility
    _assert_contract(cls, name="browser", timeout=90)


def test_browser_registered_when_playwright_available():
    """AbilityRegistry.get('browser') returns a BrowserAbility when Playwright is present."""
    mod = _import("browser")
    if not hasattr(mod, "BrowserAbility"):
        pytest.skip("Playwright not installed — BrowserAbility guard active")

    _assert_registered("browser", mod.BrowserAbility)


def test_browser_import_does_not_raise_without_playwright():
    """Importing abilities.browser never raises even when playwright is absent."""
    # Temporarily remove playwright from sys.modules to simulate absence
    playwright_cached = sys.modules.pop("playwright", None)
    # Also remove the ability module so it re-evaluates the try/except guard
    mod_cached = sys.modules.pop("abilities.browser", None)

    try:
        importlib.import_module("abilities.browser")
        # No exception — guard worked correctly
    except Exception as exc:
        pytest.fail(f"abilities.browser raised on import without playwright: {exc}")
    finally:
        # Restore originals so other tests are unaffected
        if playwright_cached is not None:
            sys.modules["playwright"] = playwright_cached
        if mod_cached is not None:
            sys.modules["abilities.browser"] = mod_cached


# ===========================================================================
# code_eval
# ===========================================================================


def test_code_eval_contract():
    """CodeEvalAbility satisfies the ABC contract."""
    from abilities.code_eval import CodeEvalAbility
    _assert_contract(CodeEvalAbility, name="code_eval", timeout=15)


def test_code_eval_registered():
    """AbilityRegistry.get('code_eval') returns a CodeEvalAbility."""
    from abilities.code_eval import CodeEvalAbility
    _assert_registered("code_eval", CodeEvalAbility)


def test_code_eval_restricted_globals_is_dict_with_builtins():
    """CodeEvalAbility._RESTRICTED_GLOBALS is a dict containing a __builtins__ entry."""
    from abilities.code_eval import CodeEvalAbility
    rg = CodeEvalAbility._RESTRICTED_GLOBALS
    assert isinstance(rg, dict), "_RESTRICTED_GLOBALS must be a dict"
    assert "__builtins__" in rg, "_RESTRICTED_GLOBALS must contain '__builtins__'"


# ===========================================================================
# document
# ===========================================================================


def test_document_contract():
    """DocumentAbility satisfies the ABC contract."""
    from abilities.document import DocumentAbility
    _assert_contract(DocumentAbility, name="document", timeout=10)


def test_document_registered():
    """AbilityRegistry.get('document') returns a DocumentAbility."""
    from abilities.document import DocumentAbility
    _assert_registered("document", DocumentAbility)


def test_document_module_level_create_document_artifacts_callable():
    """create_document_artifacts is a module-level callable (legacy callers depend on this)."""
    import abilities.document as doc_mod
    assert callable(getattr(doc_mod, "create_document_artifacts", None)), (
        "abilities.document.create_document_artifacts must be a module-level callable"
    )


# ===========================================================================
# find_tools
# ===========================================================================


def test_find_tools_contract():
    """FindToolsAbility satisfies the ABC contract."""
    from abilities.find_tools import FindToolsAbility
    _assert_contract(FindToolsAbility, name="find_tools", timeout=10)


def test_find_tools_registered():
    """AbilityRegistry.get('find_tools') returns a FindToolsAbility."""
    from abilities.find_tools import FindToolsAbility
    _assert_registered("find_tools", FindToolsAbility)


def test_find_tools_abilities_db_path_ends_with_expected_suffix():
    """FindToolsAbility._ABILITIES_DB_PATH ends with assets/abilities.sqlite."""
    from abilities.find_tools import FindToolsAbility
    db_path = FindToolsAbility._ABILITIES_DB_PATH
    assert str(db_path).endswith("assets/abilities.sqlite"), (
        f"_ABILITIES_DB_PATH={db_path!r} does not end with 'assets/abilities.sqlite'"
    )


def test_find_tools_module_level_abilities_db_path():
    """_ABILITIES_DB_PATH is also accessible at module level via the class (ClassVar)."""
    from abilities.find_tools import FindToolsAbility
    from pathlib import Path
    assert isinstance(FindToolsAbility._ABILITIES_DB_PATH, Path), (
        "_ABILITIES_DB_PATH must be a pathlib.Path"
    )


# ===========================================================================
# subagent
# ===========================================================================


def test_subagent_contract():
    """SubagentAbility satisfies the ABC contract."""
    from abilities.subagent import SubagentAbility
    # TIMEOUT=305 = wait=true cap (300s) + 5s buffer for envelope render+record.
    # See backend/abilities/subagent.py for the rationale.
    _assert_contract(SubagentAbility, name="subagent", timeout=305)


def test_subagent_registered():
    """AbilityRegistry.get('subagent') returns a SubagentAbility."""
    from abilities.subagent import SubagentAbility
    _assert_registered("subagent", SubagentAbility)


# ===========================================================================
# list
# ===========================================================================


def test_list_contract():
    """ListAbility satisfies the ABC contract."""
    from abilities.list import ListAbility
    _assert_contract(ListAbility, name="list", timeout=10)


def test_list_registered():
    """AbilityRegistry.get('list') returns a ListAbility."""
    from abilities.list import ListAbility
    _assert_registered("list", ListAbility)


# ===========================================================================
# memory
# ===========================================================================


def test_memory_contract():
    """MemoryAbility satisfies the ABC contract."""
    from abilities.memory import MemoryAbility
    _assert_contract(MemoryAbility, name="memory", timeout=10)


def test_memory_registered():
    """AbilityRegistry.get('memory') returns a MemoryAbility."""
    from abilities.memory import MemoryAbility
    _assert_registered("memory", MemoryAbility)


def test_memory_all_8_radius_constants_present():
    """MemoryAbility carries all 8 radius ClassVars required by the meta-harness."""
    from abilities.memory import MemoryAbility

    required = [
        "RECALL_RADIUS_BASELINE",
        "SEED_RADIUS_BASELINE",
        "NARROW_MIN_DIST",
        "NARROW_MAX_DIST",
        "NARROW_FACTOR_FLOOR",
        "EXPAND_MIN_DIST",
        "EXPAND_MAX_DIST",
        "EXPAND_FACTOR_CEILING",
    ]
    for attr in required:
        val = getattr(MemoryAbility, attr, None)
        assert val is not None, f"MemoryAbility.{attr} is missing"
        assert isinstance(val, float), f"MemoryAbility.{attr} must be a float, got {type(val)}"


def test_memory_module_level_recall_episodes_callable():
    """recall_episodes is a module-level callable used by pre-turn seed path."""
    import abilities.memory as mem_mod
    assert callable(getattr(mem_mod, "recall_episodes", None)), (
        "abilities.memory.recall_episodes must be a module-level callable"
    )


# ===========================================================================
# news
# ===========================================================================


def test_news_contract():
    """NewsAbility satisfies the ABC contract."""
    from abilities.news import NewsAbility
    _assert_contract(NewsAbility, name="news", timeout=10)


def test_news_registered():
    """AbilityRegistry.get('news') returns a NewsAbility."""
    from abilities.news import NewsAbility
    _assert_registered("news", NewsAbility)


# ===========================================================================
# programming_docs_search
# ===========================================================================


def test_programming_docs_search_contract():
    """ProgrammingDocsSearchAbility satisfies the ABC contract."""
    from abilities.programming_docs_search import ProgrammingDocsSearchAbility
    _assert_contract(
        ProgrammingDocsSearchAbility,
        name="programming_docs_search",
        timeout=15,
    )


def test_programming_docs_search_registered():
    """AbilityRegistry.get('programming_docs_search') returns a ProgrammingDocsSearchAbility."""
    from abilities.programming_docs_search import ProgrammingDocsSearchAbility
    _assert_registered("programming_docs_search", ProgrammingDocsSearchAbility)


# ===========================================================================
# read
# ===========================================================================
# Note: ReadAbility.TIMEOUT is the dispatch timeout (10);
# _URL_FETCH_TIMEOUT ClassVar (15) is a separate HTTP-fetch budget.


def test_read_contract():
    """ReadAbility satisfies the ABC contract."""
    from abilities.read import ReadAbility
    _assert_contract(ReadAbility, name="read", timeout=10)


def test_read_registered():
    """AbilityRegistry.get('read') returns a ReadAbility."""
    from abilities.read import ReadAbility
    _assert_registered("read", ReadAbility)


def test_read_url_fetch_timeout_is_15():
    """ReadAbility._URL_FETCH_TIMEOUT equals 15."""
    from abilities.read import ReadAbility
    assert ReadAbility._URL_FETCH_TIMEOUT == 15, (
        f"ReadAbility._URL_FETCH_TIMEOUT={ReadAbility._URL_FETCH_TIMEOUT}, expected 15"
    )


def test_read_blocked_nets_contains_loopback():
    """ReadAbility._BLOCKED_NETS contains the loopback network 127.0.0.0/8."""
    from abilities.read import ReadAbility
    blocked_strs = [str(net) for net in ReadAbility._BLOCKED_NETS]
    assert "127.0.0.0/8" in blocked_strs, (
        f"_BLOCKED_NETS does not contain '127.0.0.0/8'. Found: {blocked_strs}"
    )


# ===========================================================================
# review_tool_calls
# ===========================================================================


def test_review_tool_calls_contract():
    """ReviewToolCallsAbility satisfies the ABC contract."""
    from abilities.review_tool_calls import ReviewToolCallsAbility
    _assert_contract(
        ReviewToolCallsAbility,
        name="review_tool_calls",
        timeout=10,
    )


def test_review_tool_calls_registered():
    """AbilityRegistry.get('review_tool_calls') returns a ReviewToolCallsAbility."""
    from abilities.review_tool_calls import ReviewToolCallsAbility
    _assert_registered("review_tool_calls", ReviewToolCallsAbility)


# ===========================================================================
# schedule
# ===========================================================================


def test_schedule_contract():
    """ScheduleAbility satisfies the ABC contract."""
    from abilities.schedule import ScheduleAbility
    _assert_contract(ScheduleAbility, name="schedule", timeout=10)


def test_schedule_registered():
    """AbilityRegistry.get('schedule') returns a ScheduleAbility."""
    from abilities.schedule import ScheduleAbility
    _assert_registered("schedule", ScheduleAbility)


def test_schedule_past_due_grace_seconds_is_120():
    """ScheduleAbility._PAST_DUE_GRACE_SECONDS equals 120."""
    from abilities.schedule import ScheduleAbility
    assert ScheduleAbility._PAST_DUE_GRACE_SECONDS == 120, (
        f"_PAST_DUE_GRACE_SECONDS={ScheduleAbility._PAST_DUE_GRACE_SECONDS}, expected 120"
    )


# ===========================================================================
# search
# ===========================================================================


def test_search_contract():
    """SearchAbility satisfies the ABC contract."""
    from abilities.search import SearchAbility
    _assert_contract(SearchAbility, name="search", timeout=20)


def test_search_registered():
    """AbilityRegistry.get('search') returns a SearchAbility."""
    from abilities.search import SearchAbility
    _assert_registered("search", SearchAbility)

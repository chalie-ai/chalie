"""Feature tests for find_tools v2.

All six tests run against the REAL production stack:
- Real FindToolsAbility.run() entry point
- Real EmbeddingService (ONNX, local)
- Real abilities.sqlite (production DB, not a tmp fixture)
- Real MessageProcessor stub (flat object.__new__, same pattern as
  test_find_tools_discovery.py / P1 no-subclass rule)

These tests are written BEFORE the v2 implementation exists.
They MUST FAIL on current (v1) code — that is the required failing
baseline for the feature.

Empirically-pinned floor query
-------------------------------
Query: 'search the web for news'
Observed against real abilities.sqlite (2026-06-05):
  web_search   RRF=0.1213  (dual-signal: vec rank-1 + FTS rank-1)  GOOD
  news         RRF=0.0625  (single-signal: FTS rank-1 only)         JUNK
  read         RRF=0.0556  (single-signal)                          JUNK
  search       RRF=0.0526  (single-signal)                          JUNK
  web_browse   RRF=0.0500  (single-signal)                          JUNK

v2 floor MIN_RRF_SCORE=0.075 retains web_search (0.1213 >= 0.075)
and evicts news (0.0625 <= 0.075). On v1 (no floor), news IS injected
into active_tools. The floor eviction test catches this regression.
"""

import pytest

from abilities.find_tools import FindToolsAbility
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared stub-processor factory — mirrors test_find_tools_discovery.py pattern
# ---------------------------------------------------------------------------

def _stub_proc(discoverable: list[str]) -> MessageProcessor:
    """Flat MessageProcessor (no subclass, per P1 rule) carrying a config whose
    ``discoverable`` is the allow-list gate.  find_tools reads
    ``mp.config.discoverable`` / ``mp.config.blocked`` and appends
    matched names to _active_tools via _append_active()."""
    proc = object.__new__(MessageProcessor)
    proc.config = make_stub_config(discoverable=discoverable)
    proc._active_tools = []
    return proc


def _run(ability: FindToolsAbility, proc: MessageProcessor, params: dict) -> str:
    """Bind proc to ability and call run(); return the result string."""
    ability.mp = proc
    return ability.run(params)


# ---------------------------------------------------------------------------
# Test 1 — floor eviction (query path)
#
# EMPIRICALLY PINNED against real abilities.sqlite 2026-06-05:
#   query='search the web for news'
#   web_search  RRF=0.1213  (dual-signal: vec+FTS) → KEPT  by floor >=0.075
#   news        RRF=0.0625  (single-signal: FTS only) → EVICTED by floor <=0.075
#
# On v1 code: news IS in active_tools (no floor). Test FAILS → failing baseline.
# On v2 code: news is evicted, web_search stays.     Test PASSES.
# ---------------------------------------------------------------------------

class TestFloorEviction:

    def test_single_signal_junk_evicted_dual_signal_match_kept(self):
        """The MIN_RRF_SCORE=0.075 floor keeps the dual-signal match (web_search,
        RRF=0.1213) and evicts the single-signal junk (news, RRF=0.0625).

        Current v1 code has no floor — news IS injected → this test FAILS on v1.
        """
        ability = FindToolsAbility()
        # Include all the tools that appear in the top-5 for this query so
        # the gate does not filter them away before the floor would act.
        proc = _stub_proc(discoverable=[
            "web_search", "news", "read", "search", "web_browse",
        ])

        _run(ability, proc, {"query": "search the web for news"})

        assert "web_search" in proc.active_tools, (
            "web_search (RRF=0.1213, dual-signal) must survive the floor and "
            f"appear in active_tools. Got: {proc.active_tools}"
        )
        assert "news" not in proc.active_tools, (
            "news (RRF=0.0625, single-signal junk) must be evicted by the "
            f"MIN_RRF_SCORE=0.075 floor. Got: {proc.active_tools}"
        )


# ---------------------------------------------------------------------------
# Test 2 — select exact path
#
# v2 adds a 'select' param: array[str] of exact tool names.
# Matched names from effective_allow are appended to active_tools.
# On v1: 'select' is not read; run() falls through to query-required error.
# ---------------------------------------------------------------------------

class TestSelectExact:

    def test_select_exact_appends_both_named_tools(self):
        """run({'select': ['weather', 'email']}) appends both names to
        active_tools via the select path.

        v1 does not read 'select' → falls to query-required error, nothing
        appended. Test FAILS on v1.
        """
        ability = FindToolsAbility()
        proc = _stub_proc(discoverable=["weather", "email", "web_search"])

        result = _run(ability, proc, {"select": ["weather", "email"]})

        assert "weather" in proc.active_tools, (
            f"'weather' must be appended via select path. active_tools={proc.active_tools}"
        )
        assert "email" in proc.active_tools, (
            f"'email' must be appended via select path. active_tools={proc.active_tools}"
        )
        # Sanity: no error slug in the tag when select succeeds
        assert "error=" not in result, (
            f"select with valid names must not produce an error result. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — select not-found feedback
#
# When a name in 'select' does not resolve (not in effective_allow or DISCOVERABLE),
# the result must include a 'not found or unavailable' line listing the missing name.
# The valid name is still added.
# ---------------------------------------------------------------------------

class TestSelectNotFound:

    def test_select_valid_added_bogus_reported_as_not_found(self):
        """run({'select': ['weather', 'bogus_tool_xyz']}) → weather added to
        active_tools, result contains 'not found or unavailable' and
        'bogus_tool_xyz'.

        v1 returns query-required error → neither behaviour. Test FAILS on v1.
        """
        ability = FindToolsAbility()
        proc = _stub_proc(discoverable=["weather", "email"])

        result = _run(ability, proc, {"select": ["weather", "bogus_tool_xyz"]})

        assert "weather" in proc.active_tools, (
            f"Valid 'weather' must be appended. active_tools={proc.active_tools}"
        )
        assert "bogus_tool_xyz" not in proc.active_tools, (
            f"Non-existent 'bogus_tool_xyz' must NOT be added. "
            f"active_tools={proc.active_tools}"
        )
        # The result string must report the unresolved name
        result_lower = result.lower()
        assert "not found or unavailable" in result_lower, (
            f"Result must contain 'not found or unavailable' for missing names. "
            f"result={result!r}"
        )
        assert "bogus_tool_xyz" in result, (
            f"Result must name the unresolved tool 'bogus_tool_xyz'. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — both params → select wins, query is ignored
#
# When both 'select' and 'query' are given, v2 uses 'select' and ignores
# 'query'. The query path would discover different tools via semantic search.
# We can verify select wins by checking only the selected tool is added and
# no extra query-matched tools appear alongside it from the query path.
#
# On v1: 'select' is ignored, 'query' path runs. If the query matches
# something other than 'weather' at the top, the wrong set is appended.
# The test catches the precedence contract.
# ---------------------------------------------------------------------------

class TestBothParamsSelectWins:

    def test_both_params_only_selected_tool_added_not_query_results(self):
        """run({'select': ['weather'], 'query': 'search the web for news'})
        must add only 'weather' (the selected tool). The query 'search the web
        for news' would add web_search + junk if the query path ran.

        v1 ignores 'select', runs the query path → web_search (not weather) is
        appended. Test FAILS on v1.
        """
        ability = FindToolsAbility()
        # Allow all candidates so the gate doesn't interfere with what we're testing
        proc = _stub_proc(discoverable=[
            "weather", "web_search", "news", "read", "search", "web_browse",
        ])

        _run(ability, proc, {
            "select": ["weather"],
            "query": "search the web for news",
        })

        assert "weather" in proc.active_tools, (
            f"'weather' (select param) must be added. active_tools={proc.active_tools}"
        )
        # If the query path ran, web_search would be the top hit (RRF=0.1213).
        # Its presence proves the query path executed instead of select-wins.
        assert "web_search" not in proc.active_tools, (
            f"'web_search' must NOT be added — it comes from the query path which "
            f"should be ignored when 'select' is present. "
            f"active_tools={proc.active_tools}"
        )


# ---------------------------------------------------------------------------
# Test 5 — neither param → error, nothing appended
#
# v2 requires at least one of 'select' or 'query'. Neither → error result,
# zero names added to active_tools.
#
# On v1: run({}) already returns query-required error and nothing is added.
# BUT v2 must report a specific error about BOTH params being missing
# (not "query-required"). We assert on the behaviour: error in result +
# active_tools remains empty. If v2 changes the error slug, the intent test
# still passes — we do not lock on the exact slug string, only the direction.
#
# Additionally: the existing test_find_tools_discovery.py does not assert the
# v2 error format. This test ensures the neither-param case stays clean under v2.
# ---------------------------------------------------------------------------

class TestNeitherParam:

    def test_neither_select_nor_query_returns_error_nothing_appended(self):
        """run({}) with neither 'select' nor 'query' must return an error result
        and must NOT append anything to active_tools.

        This behaviour is identical in v1 and v2, but the test locks the contract
        for the v2 implementation path (which restructures the param-validation
        logic). The test will pass on v1 (error already returned), which is
        acceptable — it documents a must-not-regress invariant.

        To provide a v2-specific failing hook: also assert that the error message
        does NOT say 'query-required' alone (v2 must mention both missing params
        OR use a combined error slug). This assertion FAILS on v1 which emits
        exactly 'error=query-required'.
        """
        ability = FindToolsAbility()
        proc = _stub_proc(discoverable=["weather", "email"])

        result = _run(ability, proc, {})

        assert proc.active_tools == [], (
            f"Nothing must be appended when neither param is given. "
            f"active_tools={proc.active_tools}"
        )
        # v2 error must cover the combined missing-params case.
        # v1 emits 'error=query-required' which is a query-only error slug.
        # v2 must NOT emit this v1-only slug (it must use a combined-params slug
        # e.g. 'error=params-required' or similar).
        assert "query-required" not in result, (
            f"v2 must use a combined-params error slug, not the v1 'query-required' "
            f"slug that only mentions 'query'. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Test 6 — result format: added the following tools + input_schema, no
#           legacy 'relevance' or 'added_tools' tokens
# ---------------------------------------------------------------------------

class TestResultFormat:

    def test_successful_run_result_contains_required_v2_phrases_not_v1(self):
        """A successful find_tools call must produce a result string that:
          - Contains 'added the following tools'  (v2 universal result format)
          - Contains the added tool's 'input_schema' key  (v2 embeds full schema)
          - Does NOT contain 'relevance'  (v1 cosmetic field, dropped in v2)
          - Does NOT contain 'added_tools'  (v1 result envelope, dropped in v2)

        On v1 code: result contains 'added_tools' + 'relevance', no
        'added the following tools', no 'input_schema'. All four assertions fail.
        """
        ability = FindToolsAbility()
        # Use a single discoverable tool so the result is deterministic.
        # 'weather' reliably appears in the top-5 for 'weather forecast'.
        proc = _stub_proc(discoverable=["weather"])

        result = _run(ability, proc, {"query": "weather forecast"})

        assert "added the following tools" in result, (
            "v2 universal result must contain 'added the following tools'. "
            f"result={result!r}"
        )
        assert "input_schema" in result, (
            "v2 result must carry the tool's input_schema. "
            f"result={result!r}"
        )
        assert "relevance" not in result, (
            "v2 result must NOT contain the legacy 'relevance' field. "
            f"result={result!r}"
        )
        assert "added_tools" not in result, (
            "v2 result must NOT contain the legacy 'added_tools' envelope. "
            f"result={result!r}"
        )

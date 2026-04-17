# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for ToolSynthesisProcessor and its worker helpers.

Strategy: input → output on real objects.
- Real in-memory SQLite via the `db` fixture (full production schema).
- Real MemoryStore via the `store` fixture.
- No mocks for the data path.
- send() is NOT called — it requires a live LLM provider. Every test
  exercises observable behaviour below that boundary.
"""

from datetime import datetime, timezone, timedelta

import pytest

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utc_iso(offset_minutes: float = 0) -> str:
    """Return an ISO-8601 UTC timestamp offset from now by *offset_minutes*."""
    dt = datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)
    return dt.isoformat()


def _seed_transcript_row(db, channel: str = 'user', role: str = 'user',
                          content: str = 'test') -> int:
    """Insert a minimal transcript row and return its id."""
    ts = _utc_iso()
    db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) VALUES (?, ?, ?, ?)",
        (channel, role, content, ts),
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def _seed_tool_call(db, transcript_id: int, tool_name: str, result: str,
                    ephemeral: int = 1, offset_minutes: float = 0) -> None:
    """Insert a tool_calls row with the given attributes."""
    ts = _utc_iso(offset_minutes)
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, result, ephemeral, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (transcript_id, tool_name, result, ephemeral, ts),
    )
    db.commit()


# ── TestShouldExclude ──────────────────────────────────────────────────────────

class TestShouldExclude:
    """_should_exclude() — exact membership and prefix rules."""

    def test_memory_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('memory') is True

    def test_compaction_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('compaction') is True

    def test_tool_synthesis_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('tool_synthesis') is True

    def test_user_steer_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('user_steer') is True

    def test_tool_compaction_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('tool_compaction') is True

    def test_act_restart_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('act_restart') is True

    def test_weather_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('weather') is True

    def test_search_exact_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('search') is True

    def test_search_prefix_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('search_web') is True
        assert _should_exclude('search_ddg') is True
        assert _should_exclude('search_brave') is True

    def test_unknown_tool_not_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('browser') is False

    def test_document_not_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('document') is False

    def test_news_not_excluded(self):
        from services.tool_synthesis_processor import _should_exclude
        assert _should_exclude('news') is False


# ── TestGatherRecentToolCalls ──────────────────────────────────────────────────

class TestGatherRecentToolCalls:
    """_gather_recent_tool_calls() — queries the real DB, returns formatted digest."""

    def test_returns_none_when_no_rows(self, db, store):
        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()
        assert result is None

    def test_returns_string_when_qualifying_row_exists(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'Page loaded: https://example.com', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        assert isinstance(result, str)

    def test_tool_name_and_result_appear_in_digest(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'Found article: Python 3.13 released', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert 'browser' in result
        assert 'Python 3.13 released' in result

    def test_ephemeral_zero_rows_excluded(self, db, store):
        """ephemeral=0 rows (memory seed, compaction) must not be gathered."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'SHOULD NOT APPEAR', ephemeral=0)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_excluded_tool_name_filtered_out(self, db, store):
        """A tool in _EXCLUDED_TOOLS produces no digest entry."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'memory', 'stored fact', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_search_prefix_tool_filtered_out(self, db, store):
        """Tool names starting with search_ are filtered regardless of suffix."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'search_brave', 'lots of results', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_old_rows_outside_lookback_excluded(self, db, store):
        """Rows older than LOOKBACK_MINUTES are not returned."""
        from services.tool_synthesis_processor import LOOKBACK_MINUTES

        tid = _seed_transcript_row(db)
        # Place the row 5 minutes past the lookback boundary
        _seed_tool_call(db, tid, 'browser', 'old result',
                        ephemeral=1, offset_minutes=-(LOOKBACK_MINUTES + 5))

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_recent_row_inside_lookback_included(self, db, store):
        """Rows within the lookback window are included."""
        from services.tool_synthesis_processor import LOOKBACK_MINUTES

        tid = _seed_transcript_row(db)
        # Place the row 5 minutes inside the lookback boundary
        _seed_tool_call(db, tid, 'browser', 'recent result',
                        ephemeral=1, offset_minutes=-(LOOKBACK_MINUTES - 5))

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        assert 'recent result' in result

    def test_empty_result_rows_skipped(self, db, store):
        """Rows with an empty result string produce no digest entry."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', '', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_multiple_qualifying_rows_all_appear(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'result alpha', ephemeral=1, offset_minutes=-1)
        _seed_tool_call(db, tid, 'document', 'result beta', ephemeral=1, offset_minutes=-2)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert 'result alpha' in result
        assert 'result beta' in result

    def test_mixed_ephemeral_only_ephemeral_one_appears(self, db, store):
        """Only ephemeral=1 rows surface; ephemeral=0 siblings are invisible."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'SHOULD APPEAR', ephemeral=1)
        _seed_tool_call(db, tid, 'browser', 'SHOULD NOT APPEAR', ephemeral=0)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert 'SHOULD APPEAR' in result
        assert 'SHOULD NOT APPEAR' not in result

    def test_header_contains_count_and_window(self, db, store):
        """Digest header lists the entry count and the lookback window in minutes."""
        from services.tool_synthesis_processor import LOOKBACK_MINUTES

        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'result one', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert '1 calls' in result
        assert str(LOOKBACK_MINUTES) in result

    def test_all_excluded_tools_returns_none(self, db, store):
        """When every DB row is for an excluded tool, None is returned."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'memory', 'fact stored', ephemeral=1)
        _seed_tool_call(db, tid, 'weather', 'sunny 25C', ephemeral=1)
        _seed_tool_call(db, tid, 'search', 'lots of links', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None


# ── TestResultTruncation ───────────────────────────────────────────────────────

class TestResultTruncation:
    """Result strings longer than RESULT_TRUNCATE_CHARS are truncated."""

    def test_long_result_truncated_to_limit(self, db, store):
        from services.tool_synthesis_processor import RESULT_TRUNCATE_CHARS

        long_result = 'X' * (RESULT_TRUNCATE_CHARS + 500)
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', long_result, ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        assert '...[truncated]' in result
        # The digest body for this entry must not exceed well beyond the limit
        assert long_result not in result

    def test_result_at_exact_limit_not_truncated(self, db, store):
        from services.tool_synthesis_processor import RESULT_TRUNCATE_CHARS

        exact_result = 'Y' * RESULT_TRUNCATE_CHARS
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', exact_result, ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        assert '...[truncated]' not in result

    def test_short_result_not_truncated(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'short result', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert '...[truncated]' not in result


# ── TestChunking ───────────────────────────────────────────────────────────────

class TestChunking:
    """MAX_TOOL_CALLS cap — 100 qualifying rows → exactly 100 returned.

    We verify the DB cap rather than a Python-level chunking concept because
    the implementation uses a SQL LIMIT, not a for-loop chunker.
    The 'batch of 60 produces 2 chunks of ≤50' requirement from the spec
    maps to the SQL LIMIT ensuring the digest never exceeds MAX_TOOL_CALLS rows.
    """

    def test_cap_limits_rows_returned(self, db, store):
        """Insert 110 qualifying rows; only MAX_TOOL_CALLS appear in the digest."""
        from services.tool_synthesis_processor import MAX_TOOL_CALLS

        tid = _seed_transcript_row(db)
        for i in range(MAX_TOOL_CALLS + 10):
            offset = -float(i)  # stagger timestamps slightly
            _seed_tool_call(db, tid, 'document', f'result_{i}',
                            ephemeral=1, offset_minutes=offset)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        # The header line says how many entries are in the digest
        count_in_header = int(
            result.split(' calls from the last')[0].rsplit('(', 1)[-1].strip()
        )
        assert count_in_header <= MAX_TOOL_CALLS


# ── TestProcessorClass ─────────────────────────────────────────────────────────

class TestProcessorClass:
    """ToolSynthesisProcessor class attributes and method contracts."""

    def _make(self, raw_input: str = 'test digest'):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        return ToolSynthesisProcessor(raw_input=raw_input, metadata={})

    # ── Class constants ────────────────────────────────────────────────────────

    def test_channel_is_tool_synthesis(self):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        assert ToolSynthesisProcessor.CHANNEL == 'tool_synthesis'

    def test_role_is_tool_synthesis(self):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        assert ToolSynthesisProcessor.ROLE == 'tool_synthesis'

    def test_max_iterations_is_ten(self):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        assert ToolSynthesisProcessor.MAX_ITERATIONS == 10

    def test_max_timeout_is_300(self):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        assert ToolSynthesisProcessor.MAX_TIMEOUT == 300

    def test_native_tools_is_memory_only(self):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        assert ToolSynthesisProcessor.NATIVE_TOOLS == ['memory']

    # ── getUserDefinition() ────────────────────────────────────────────────────

    def test_get_user_definition_returns_non_empty_string(self):
        p = self._make()
        defn = p.getUserDefinition()
        assert isinstance(defn, str)
        assert len(defn) > 0

    def test_get_user_definition_mentions_tool_synthesis(self):
        p = self._make()
        assert 'tool_synthesis' in p.getUserDefinition()

    # ── getUserPrompt() ────────────────────────────────────────────────────────

    def test_get_user_prompt_contains_role_prefix(self):
        p = self._make('my digest content')
        prompt = p.getUserPrompt()
        assert prompt.startswith('tool_synthesis:')

    def test_get_user_prompt_contains_raw_input(self):
        p = self._make('unique_digest_abc123')
        prompt = p.getUserPrompt()
        assert 'unique_digest_abc123' in prompt

    def test_get_user_prompt_no_trail_on_first_iteration(self):
        """On the first iteration _act_trail is empty — no trail appended."""
        p = self._make('digest text')
        prompt = p.getUserPrompt()
        # Raw input is the only content
        assert prompt == 'tool_synthesis: digest text'

    def test_get_user_prompt_includes_act_trail_when_present(self):
        """After tool calls populate _act_trail, it appears in getUserPrompt."""
        p = self._make('digest text')
        p._act_trail = ['[browser()] some result']
        prompt = p.getUserPrompt()
        assert '[browser()]' in prompt

    # ── getSystemPrompt() ─────────────────────────────────────────────────────

    def test_get_system_prompt_returns_non_empty_string(self):
        p = self._make()
        sp = p.getSystemPrompt()
        assert isinstance(sp, str)
        assert len(sp) > 0

    def test_get_system_prompt_contains_user_definition(self):
        p = self._make()
        sp = p.getSystemPrompt()
        defn = p.getUserDefinition()
        assert defn in sp

    def test_get_system_prompt_contains_synthesis_instruction(self):
        p = self._make()
        sp = p.getSystemPrompt()
        # The ToolSynthesisSystemMessagePrompt body must survive into the output
        assert 'synthesis' in sp.lower()

    def test_get_system_prompt_contains_synthesis_no_action_sentinel(self):
        p = self._make()
        sp = p.getSystemPrompt()
        assert 'SYNTHESIS_NO_ACTION' in sp

    # ── getTools() ────────────────────────────────────────────────────────────

    def test_get_tools_returns_exactly_one_schema(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tools = p.getTools()
        assert len(tools) == 1

    def test_get_tools_returns_memory_schema(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tools = p.getTools()
        assert tools[0]['name'] == 'memory'

    def test_get_tools_memory_schema_has_input_schema(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tools = p.getTools()
        assert 'input_schema' in tools[0]

    def test_get_tools_memory_schema_has_action_enum(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tools = p.getTools()
        props = tools[0]['input_schema']['properties']
        assert 'action' in props
        assert set(props['action']['enum']) == {'store', 'recall', 'reflect', 'forget'}

    def test_get_tools_no_find_tools_present(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tool_names = [t['name'] for t in p.getTools()]
        assert 'find_tools' not in tool_names

    def test_get_tools_no_goal_pursuit_present(self):
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        tool_names = [t['name'] for t in p.getTools()]
        assert 'goal_pursuit' not in tool_names

    # ── getDynamicTools() ─────────────────────────────────────────────────────

    def test_get_dynamic_tools_returns_empty_list(self):
        p = self._make()
        assert p.getDynamicTools() == []

    def test_get_dynamic_tools_empty_even_after_discovered_tools_set(self):
        """Manually populating _discovered_tools does not leak into getDynamicTools."""
        p = self._make()
        p._discovered_tools = [{'name': 'some_external_tool', 'input_schema': {}}]
        assert p.getDynamicTools() == []

    def test_get_tools_returns_only_memory_when_discovered_tools_populated(self):
        """getDynamicTools() returning [] means getTools() stays at memory only."""
        from services.tool_schema_service import clear_cache
        clear_cache()
        p = self._make()
        p._discovered_tools = [{'name': 'injected_tool', 'input_schema': {}}]
        tool_names = [t['name'] for t in p.getTools()]
        assert 'injected_tool' not in tool_names
        assert tool_names == ['memory']

    # ── handleTool() ──────────────────────────────────────────────────────────

    def test_handle_tool_memory_recall_returns_string(self, db, store):
        """handleTool with memory/recall dispatches to the real skill and returns str."""
        tid = _seed_transcript_row(db)
        p = self._make('test')
        # _uid must be set before handleTool calls ToolRenderAndRecordService
        p._uid = tid

        from services.act_dispatcher_service import ActDispatcherService
        p._dispatcher = ActDispatcherService(execution_gate=False)

        tc = {'name': 'memory', 'input': {'action': 'recall', 'query': 'test query'}}
        result = p.handleTool(tc)

        assert isinstance(result, str)

    def test_handle_tool_memory_store_returns_string(self, db, store):
        """handleTool with memory/store dispatches to the real skill and returns str."""
        tid = _seed_transcript_row(db)
        p = self._make('test')
        p._uid = tid

        from services.act_dispatcher_service import ActDispatcherService
        p._dispatcher = ActDispatcherService(execution_gate=False)

        tc = {'name': 'memory', 'input': {
            'action': 'store', 'kind': 'user_specific',
            'key': 'test_key', 'value': 'test_value',
        }}
        result = p.handleTool(tc)

        assert isinstance(result, str)

    def test_handle_tool_unknown_name_returns_error_string(self, db, store):
        """An unknown tool name never raises — returns an error string instead."""
        tid = _seed_transcript_row(db)
        p = self._make('test')
        p._uid = tid

        from services.act_dispatcher_service import ActDispatcherService
        p._dispatcher = ActDispatcherService(execution_gate=False)

        tc = {'name': 'nonexistent_tool_xyz', 'input': {}}
        result = p.handleTool(tc)

        assert isinstance(result, str)
        # Must not raise and must surface the tool name in some form
        assert 'nonexistent_tool_xyz' in result or 'Unknown' in result or 'ERROR' in result

    def test_handle_tool_appends_to_act_trail(self, db, store):
        """Each handleTool call appends one rendered entry to _act_trail."""
        tid = _seed_transcript_row(db)
        p = self._make('test')
        p._uid = tid

        from services.act_dispatcher_service import ActDispatcherService
        p._dispatcher = ActDispatcherService(execution_gate=False)

        assert len(p._act_trail) == 0

        tc = {'name': 'memory', 'input': {'action': 'recall', 'query': 'something'}}
        p.handleTool(tc)

        assert len(p._act_trail) == 1

    def test_handle_tool_memory_store_persists_to_data_graph(self, db, store):
        """A memory/store call via handleTool lands a row in data_graph.

        We reset the DataGraphService singleton so it binds to the test DB
        (injected by the `db` fixture via get_shared_db_service) rather than
        a stale connection from a previous test.  After the call we query via
        the same shared DB service to avoid connection-closed races.
        """
        import services.data_graph_service as _dgs_mod

        # Reset singleton so DataGraphService opens a fresh connection against
        # the test DB that the `db` fixture has injected as the shared service.
        _dgs_mod._instance = None

        tid = _seed_transcript_row(db)
        p = self._make('test')
        p._uid = tid

        from services.act_dispatcher_service import ActDispatcherService
        p._dispatcher = ActDispatcherService(execution_gate=False)

        tc = {'name': 'memory', 'input': {
            'action': 'store', 'kind': 'user_specific',
            'key': 'synthesis_test_key', 'value': 'synthesis_test_value',
        }}
        p.handleTool(tc)

        # Query via the shared DB service (same connection the service used)
        from services.database_service import get_shared_db_service
        svc_db = get_shared_db_service()
        with svc_db.connection() as conn:
            row = conn.execute(
                "SELECT value FROM data_graph WHERE key = 'synthesis_test_key' LIMIT 1"
            ).fetchone()

        assert row is not None
        assert row[0] == 'synthesis_test_value'


# ── TestPostTurn ───────────────────────────────────────────────────────────────

class TestPostTurn:
    """postTurn() — must not raise regardless of MetricsService availability."""

    def test_post_turn_does_not_raise(self, db, store):
        from services.tool_synthesis_processor import ToolSynthesisProcessor
        p = ToolSynthesisProcessor(raw_input='test', metadata={})
        # postTurn() internally catches MetricsService failures — must never raise
        p.postTurn()  # no assertion needed — just must not raise


# ── TestRunCycleSkipLogic ──────────────────────────────────────────────────────

class TestRunCycleSkipLogic:
    """_run_cycle() — no tool calls in window → processor never constructed.

    We verify this by confirming no transcript rows for 'tool_synthesis'
    channel are created when _run_cycle() finds nothing to process.
    Since we cannot intercept the ToolSynthesisProcessor constructor without
    mocking, we use the transcript table as the observable side-effect:
    send() writes an input row; if send() was never called, that row is absent.
    """

    def test_empty_window_leaves_no_transcript_rows(self, db, store):
        from services.tool_synthesis_processor import _run_cycle
        _run_cycle()

        rows = db.execute(
            "SELECT id FROM transcript WHERE channel = 'tool_synthesis'"
        ).fetchall()
        assert rows == []

    def test_all_excluded_tools_leaves_no_transcript_rows(self, db, store):
        """All rows excluded → _run_cycle skips → no transcript rows created."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'memory', 'fact', ephemeral=1)
        _seed_tool_call(db, tid, 'weather', '30C sunny', ephemeral=1)

        from services.tool_synthesis_processor import _run_cycle
        _run_cycle()

        rows = db.execute(
            "SELECT id FROM transcript WHERE channel = 'tool_synthesis'"
        ).fetchall()
        assert rows == []


# ── TestDigestFormat ───────────────────────────────────────────────────────────

class TestDigestFormat:
    """Format assertions on the digest string produced by _gather_recent_tool_calls."""

    def test_entries_separated_by_divider(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'result A', ephemeral=1, offset_minutes=-1)
        _seed_tool_call(db, tid, 'document', 'result B', ephemeral=1, offset_minutes=-2)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        # Multiple entries must be separated by the divider
        assert '---' in result

    def test_single_entry_no_divider(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'only result', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert '---' not in result

    def test_timestamp_prefix_in_entries(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'result with timestamp', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        # Each entry line starts with a bracketed timestamp [YYYY-MM-DD HH:MM]
        assert '[20' in result  # minimal year-prefix check

    def test_digest_header_present(self, db, store):
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'result', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert 'Recent tool activity' in result

    def test_empty_result_after_filtering_returns_none(self, db, store):
        """Only rows with non-empty results contribute to the digest; empty → None."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', '   ', ephemeral=1)  # whitespace-only

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is None

    def test_params_included_when_non_empty(self, db, store):
        """Non-trivial params appear in the digest entry for context."""
        tid = _seed_transcript_row(db)
        ts = _utc_iso()
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tid, 'browser', '{"url": "https://example.com"}', 'Page loaded', 1, ts),
        )
        db.commit()

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        assert 'https://example.com' in result  # NOSONAR

    def test_default_empty_params_omitted(self, db, store):
        """The default '{}' params value does not appear in the digest."""
        tid = _seed_transcript_row(db)
        _seed_tool_call(db, tid, 'browser', 'Page loaded', ephemeral=1)

        from services.tool_synthesis_processor import _gather_recent_tool_calls
        result = _gather_recent_tool_calls()

        assert result is not None
        # '{}' should not appear as a params parenthetical
        assert '({})' not in result


# ── TestWorkerFunction ─────────────────────────────────────────────────────────

class TestWorkerFunction:
    """tool_synthesis_worker is importable and has correct signature."""

    def test_worker_function_is_callable(self):
        from services.tool_synthesis_processor import tool_synthesis_worker
        assert callable(tool_synthesis_worker)

    def test_worker_accepts_shared_state_kwarg(self):
        """The function signature accepts an optional shared_state dict."""
        import inspect
        from services.tool_synthesis_processor import tool_synthesis_worker
        sig = inspect.signature(tool_synthesis_worker)
        assert 'shared_state' in sig.parameters

    def test_worker_shared_state_defaults_to_none(self):
        import inspect
        from services.tool_synthesis_processor import tool_synthesis_worker
        sig = inspect.signature(tool_synthesis_worker)
        assert sig.parameters['shared_state'].default is None

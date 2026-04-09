"""Integration tests for context_window_service and related ToolCallService updates.

All data operations use a real SQLite database (via the shared `db` fixture from
conftest.py).  The only things mocked are:
  - The LLM provider (compaction calls we cannot make in tests)
  - _embed_entry (transcript_service side-effect — not part of what is under test)

Test layout
-----------
TestBuildMessages      — build_messages() happy paths and edge cases
TestEstimateTokens     — estimate_messages_tokens() arithmetic
TestCheckAndCompact    — check_and_compact() trigger conditions
TestToolCallServiceUpdates — store/store_batch tool_call_id, get_by_transcript_ids,
                             get_find_tools_results with list input
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_transcript(db, channel, role, content, tool_call_id=None, tool_name=None):
    """Insert a transcript row directly and return its rowid."""
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO transcript (channel, role, content, tool_call_id, tool_name)
        VALUES (?, ?, ?, ?, ?)
        """,
        (channel, role, content, tool_call_id, tool_name),
    )
    db.commit()
    return cursor.lastrowid


def _insert_tool_call(db, transcript_id, tool_name, params, result,
                      invoked_by='llm', ephemeral=0, tool_call_id=None):
    """Insert a tool_calls row directly and return its rowid."""
    cursor = db.cursor()
    params_str = json.dumps(params) if isinstance(params, dict) else params
    cursor.execute(
        """
        INSERT INTO tool_calls
            (transcript_id, tool_name, params, result, invoked_by, ephemeral, tool_call_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (transcript_id, tool_name, params_str, result, invoked_by, ephemeral, tool_call_id),
    )
    db.commit()
    return cursor.lastrowid


def _insert_compaction(db, channel, compacted_text, watermark_id,
                       token_count=10, overflow_content=None):
    """Insert (or replace) a compaction record."""
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO compactions
            (channel, compacted_text, compacted_up_to_id, token_count, overflow_content)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(channel) DO UPDATE SET
            compacted_text     = excluded.compacted_text,
            compacted_up_to_id = excluded.compacted_up_to_id,
            token_count        = excluded.token_count,
            overflow_content   = excluded.overflow_content
        """,
        (channel, compacted_text, watermark_id, token_count, overflow_content),
    )
    db.commit()


# ── Tests: build_messages() ───────────────────────────────────────────────────

class TestBuildMessages:
    def test_empty_channel_no_compaction_returns_empty_list(self, db):
        """A channel with no transcript entries and no compaction yields []."""
        from services.context_window_service import build_messages

        result = build_messages('chan-empty')
        assert result == []

    def test_user_and_assistant_entries_format(self, db):
        """User and assistant entries are mapped to correct role/content dicts."""
        from services.context_window_service import build_messages

        with patch('services.transcript_service._embed_entry'):
            _insert_transcript(db, 'chan-basic', 'user', 'Hello from user')
            _insert_transcript(db, 'chan-basic', 'assistant', 'Hello from assistant')

        result = build_messages('chan-basic')

        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello from user'}
        assert result[1] == {'role': 'assistant', 'content': 'Hello from assistant'}

    def test_tool_entries_include_tool_call_id_and_name(self, db):
        """Tool entries are mapped with role=tool, tool_call_id, name, and content."""
        from services.context_window_service import build_messages

        _insert_transcript(db, 'chan-tool', 'user', 'Run a search')
        asst_id = _insert_transcript(db, 'chan-tool', 'assistant', '')
        _insert_transcript(
            db, 'chan-tool', 'tool', 'Search results here',
            tool_call_id='tc_abc123', tool_name='search',
        )
        # The assistant entry needs a matching tool_call row with tool_call_id so
        # build_messages can reconstruct tool_calls on the assistant message.
        _insert_tool_call(db, asst_id, 'search', {'q': 'test'}, 'ok',
                          tool_call_id='tc_abc123')

        result = build_messages('chan-tool')

        # user, assistant, tool
        assert len(result) == 3
        tool_msg = result[2]
        assert tool_msg['role'] == 'tool'
        assert tool_msg['tool_call_id'] == 'tc_abc123'
        assert tool_msg['name'] == 'search'
        assert tool_msg['content'] == 'Search results here'

    def test_compaction_prepended_as_user_message(self, db):
        """When a compaction exists, its text is the first message."""
        from services.context_window_service import build_messages

        watermark = _insert_transcript(db, 'chan-compact', 'user', 'First message')
        _insert_compaction(db, 'chan-compact', 'Context summary here', watermark)
        _insert_transcript(db, 'chan-compact', 'user', 'Second message')

        result = build_messages('chan-compact')

        assert result[0] == {'role': 'user', 'content': 'Context summary here'}
        # Second message is after the watermark, so it should follow
        assert result[1] == {'role': 'user', 'content': 'Second message'}

    def test_overflow_content_appears_before_compacted_text(self, db):
        """Overflow content is the very first message, before the compacted text."""
        from services.context_window_service import build_messages

        watermark = _insert_transcript(db, 'chan-overflow', 'user', 'Earlier turn')
        _insert_compaction(
            db, 'chan-overflow',
            compacted_text='Compacted context',
            watermark_id=watermark,
            overflow_content='Large tool result that caused overflow',
        )
        _insert_transcript(db, 'chan-overflow', 'user', 'After overflow')

        result = build_messages('chan-overflow')

        # Order must be: overflow → compacted → entries after watermark
        assert result[0] == {'role': 'user', 'content': 'Large tool result that caused overflow'}
        assert result[1] == {'role': 'user', 'content': 'Compacted context'}
        assert result[2] == {'role': 'user', 'content': 'After overflow'}

    def test_overflow_cleared_after_build_messages(self, db):
        """build_messages() clears overflow_content from DB after consuming it."""
        from services.context_window_service import build_messages

        watermark = _insert_transcript(db, 'chan-clear-overflow', 'user', 'Turn')
        _insert_compaction(
            db, 'chan-clear-overflow',
            compacted_text='Summary',
            watermark_id=watermark,
            overflow_content='Overflow to be cleared',
        )

        build_messages('chan-clear-overflow')

        # Verify overflow was cleared in the DB
        cursor = db.cursor()
        cursor.execute(
            "SELECT overflow_content FROM compactions WHERE channel = ?",
            ('chan-clear-overflow',),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None

    def test_assistant_with_tool_calls_reconstructed(self, db):
        """Assistant entries get tool_calls list if matching tool_calls rows exist."""
        from services.context_window_service import build_messages

        asst_id = _insert_transcript(db, 'chan-tc-recon', 'assistant', 'Calling tools')
        _insert_tool_call(
            db, asst_id, 'web_search',
            params={'query': 'latest news'},
            result='...',
            tool_call_id='tc_99',
        )

        result = build_messages('chan-tc-recon')

        assert len(result) == 1
        msg = result[0]
        assert msg['role'] == 'assistant'
        assert 'tool_calls' in msg
        assert len(msg['tool_calls']) == 1
        tc = msg['tool_calls'][0]
        assert tc['id'] == 'tc_99'
        assert tc['name'] == 'web_search'
        assert tc['input'] == {'query': 'latest news'}

    def test_assistant_without_tool_call_id_has_no_tool_calls(self, db):
        """Old assistant entries whose tool_calls rows have NULL tool_call_id are skipped."""
        from services.context_window_service import build_messages

        asst_id = _insert_transcript(db, 'chan-old-asst', 'assistant', 'Old entry')
        # Insert tool_call row with NO tool_call_id (legacy row)
        _insert_tool_call(
            db, asst_id, 'some_tool',
            params={},
            result='result',
            tool_call_id=None,  # No ID — must not appear in tool_calls
        )

        result = build_messages('chan-old-asst')

        assert len(result) == 1
        msg = result[0]
        assert 'tool_calls' not in msg

    def test_entries_ordered_by_transcript_id(self, db):
        """Entries are returned in ascending transcript id order (oldest first)."""
        from services.context_window_service import build_messages

        _insert_transcript(db, 'chan-order', 'user', 'First')
        _insert_transcript(db, 'chan-order', 'assistant', 'Second')
        _insert_transcript(db, 'chan-order', 'user', 'Third')

        result = build_messages('chan-order')

        assert [m['content'] for m in result] == ['First', 'Second', 'Third']

    def test_entries_before_compaction_watermark_excluded(self, db):
        """Entries with id <= watermark are excluded; only post-watermark entries appear."""
        from services.context_window_service import build_messages

        id1 = _insert_transcript(db, 'chan-wm', 'user', 'Before watermark')
        id2 = _insert_transcript(db, 'chan-wm', 'user', 'Also before watermark')
        _insert_compaction(db, 'chan-wm', 'Summary', watermark_id=id2)
        _insert_transcript(db, 'chan-wm', 'user', 'After watermark')

        result = build_messages('chan-wm')

        contents = [m['content'] for m in result]
        assert 'Before watermark' not in contents
        assert 'Also before watermark' not in contents
        assert 'Summary' in contents
        assert 'After watermark' in contents

    def test_multiple_tool_calls_on_one_assistant_entry(self, db):
        """Multiple tool calls on one assistant entry are all reconstructed."""
        from services.context_window_service import build_messages

        asst_id = _insert_transcript(db, 'chan-multi-tc', 'assistant', '')
        _insert_tool_call(db, asst_id, 'tool_a', {'k': 'v1'}, 'r1', tool_call_id='tc_1')
        _insert_tool_call(db, asst_id, 'tool_b', {'k': 'v2'}, 'r2', tool_call_id='tc_2')

        result = build_messages('chan-multi-tc')

        assert len(result) == 1
        assert len(result[0]['tool_calls']) == 2
        ids = {tc['id'] for tc in result[0]['tool_calls']}
        assert ids == {'tc_1', 'tc_2'}


# ── Tests: estimate_messages_tokens() ────────────────────────────────────────

class TestEstimateMessagesTokens:
    def test_empty_messages_returns_zero(self, db):
        from services.context_window_service import estimate_messages_tokens
        assert estimate_messages_tokens([]) == 0

    def test_single_message_returns_nonzero_estimate(self, db):
        from services.context_window_service import estimate_messages_tokens

        messages = [{'role': 'user', 'content': 'Hello world this is a test sentence'}]
        result = estimate_messages_tokens(messages)
        assert result > 0

    def test_multiple_messages_sum_content_tokens(self, db):
        from services.context_window_service import estimate_messages_tokens
        from services.llm_service import estimate_tokens

        messages = [
            {'role': 'user', 'content': 'first message content'},
            {'role': 'assistant', 'content': 'second message content'},
        ]
        result = estimate_messages_tokens(messages)
        expected = (
            estimate_tokens('first message content') +
            estimate_tokens('second message content')
        )
        assert result == expected

    def test_tool_calls_metadata_contributes_tokens(self, db):
        from services.context_window_service import estimate_messages_tokens

        messages_with_tc = [
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'id': 'tc_1', 'name': 'web_search', 'input': {'query': 'breaking news'}},
                ],
            }
        ]
        messages_without_tc = [{'role': 'assistant', 'content': ''}]

        with_tc = estimate_messages_tokens(messages_with_tc)
        without_tc = estimate_messages_tokens(messages_without_tc)
        assert with_tc > without_tc

    def test_none_content_treated_as_empty(self, db):
        from services.context_window_service import estimate_messages_tokens

        # content=None must not raise
        messages = [{'role': 'assistant', 'content': None}]
        result = estimate_messages_tokens(messages)
        assert result == 0

    def test_message_without_content_key(self, db):
        from services.context_window_service import estimate_messages_tokens

        messages = [{'role': 'tool', 'tool_call_id': 'tc_1', 'name': 'x'}]
        result = estimate_messages_tokens(messages)
        assert result == 0


# ── Tests: check_and_compact() ───────────────────────────────────────────────

class TestCheckAndCompact:
    def test_returns_false_when_context_limit_is_zero(self, db):
        from services.context_window_service import check_and_compact
        assert check_and_compact('chan-zero-limit', context_limit=0) is False

    def test_returns_false_when_channel_is_empty(self, db):
        from services.context_window_service import check_and_compact
        assert check_and_compact('', context_limit=32000) is False

    def test_returns_false_when_below_80_percent_threshold(self, db):
        """Small context well below 80% threshold should not trigger compaction."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            _insert_transcript(db, 'chan-below', 'user', 'Short message')
            _insert_transcript(db, 'chan-below', 'assistant', 'Short reply')

        # 2 short messages << 80% of 32000 (25600 tokens)
        result = check_and_compact('chan-below', context_limit=32000)
        assert result is False

    def test_triggers_compaction_above_80_percent_threshold(self, db):
        """Context exceeding 80% of limit triggers compaction."""
        from services.context_window_service import check_and_compact

        # Insert enough content to exceed 80% of a tiny context_limit
        with patch('services.transcript_service._embed_entry'):
            # ~300 tokens worth of content
            long_text = 'word ' * 100
            _insert_transcript(db, 'chan-above', 'user', long_text)
            _insert_transcript(db, 'chan-above', 'assistant', long_text)
            _insert_transcript(db, 'chan-above', 'user', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted summary text')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            # context_limit=200 → threshold=160 tokens; ~390 tokens of content
            result = check_and_compact('chan-above', context_limit=200)

        assert result is True

    def test_compaction_stored_in_db_with_correct_watermark(self, db):
        """After compaction, the compactions table has the correct watermark id."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 100
            _insert_transcript(db, 'chan-wm-store', 'user', long_text)
            id2 = _insert_transcript(db, 'chan-wm-store', 'assistant', long_text)
            id3 = _insert_transcript(db, 'chan-wm-store', 'user', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Summary stored')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            check_and_compact('chan-wm-store', context_limit=200)

        cursor = db.cursor()
        cursor.execute(
            "SELECT compacted_text, compacted_up_to_id FROM compactions WHERE channel = ?",
            ('chan-wm-store',),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'Summary stored'
        # Watermark should be the highest transcript id for the channel
        assert row[1] == id3

    def test_overflow_pending_content_exceeds_100_percent_triggers_compaction(self, db):
        """pending_content that would push context over 100% triggers overflow compaction."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 50  # ~65 tokens
            _insert_transcript(db, 'chan-overflow-trig', 'user', long_text)
            _insert_transcript(db, 'chan-overflow-trig', 'assistant', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Overflow compaction result')

        # context_limit=150; current ~130 tokens + pending 100 tokens > 150 → overflow
        pending = 'word ' * 80  # ~104 tokens
        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = check_and_compact(
                'chan-overflow-trig', context_limit=150,
                pending_content=pending, is_tool_triggered=True,
            )

        assert result is True

    def test_overflow_content_stored_in_compaction_record(self, db):
        """When overflow compaction fires, overflow_content is stored in DB."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 50
            _insert_transcript(db, 'chan-overflow-store', 'user', long_text)
            _insert_transcript(db, 'chan-overflow-store', 'assistant', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Overflow summary')

        pending = 'word ' * 80
        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            check_and_compact(
                'chan-overflow-store', context_limit=150,
                pending_content=pending,
            )

        cursor = db.cursor()
        cursor.execute(
            "SELECT overflow_content FROM compactions WHERE channel = ?",
            ('chan-overflow-store',),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == pending

    def test_returns_false_when_llm_returns_empty_response(self, db):
        """Compaction returns False if LLM gives back empty text."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 100
            _insert_transcript(db, 'chan-llm-empty', 'user', long_text)
            _insert_transcript(db, 'chan-llm-empty', 'assistant', long_text)
            _insert_transcript(db, 'chan-llm-empty', 'user', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='   ')  # whitespace-only

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = check_and_compact('chan-llm-empty', context_limit=200)

        assert result is False

    def test_returns_false_when_llm_raises_exception(self, db):
        """Compaction returns False when the LLM call raises."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 100
            _insert_transcript(db, 'chan-llm-err', 'user', long_text)
            _insert_transcript(db, 'chan-llm-err', 'assistant', long_text)
            _insert_transcript(db, 'chan-llm-err', 'user', long_text)

        with patch('services.llm_service.create_refreshable_llm_service',
                   side_effect=Exception('LLM unavailable')):
            result = check_and_compact('chan-llm-err', context_limit=200)

        assert result is False

    def test_no_compaction_when_pending_content_within_limit(self, db):
        """pending_content that fits within 100% of limit does not force overflow compaction."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            # Very short entries — well below both 80% and 100% thresholds
            _insert_transcript(db, 'chan-no-overflow', 'user', 'Short')

        # pending is 1 token; limit=32000 — no compaction needed
        result = check_and_compact(
            'chan-no-overflow', context_limit=32000,
            pending_content='tiny',
        )
        assert result is False

    def test_upsert_updates_existing_compaction(self, db):
        """A second compaction call upserts, overwriting the first record."""
        from services.context_window_service import check_and_compact

        with patch('services.transcript_service._embed_entry'):
            long_text = 'word ' * 100
            _insert_transcript(db, 'chan-upsert', 'user', long_text)
            _insert_transcript(db, 'chan-upsert', 'assistant', long_text)
            _insert_transcript(db, 'chan-upsert', 'user', long_text)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='First summary')
        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            check_and_compact('chan-upsert', context_limit=200)

        # Add more content
        with patch('services.transcript_service._embed_entry'):
            _insert_transcript(db, 'chan-upsert', 'user', long_text)

        mock_llm.send_message.return_value = MagicMock(text='Second summary')
        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            check_and_compact('chan-upsert', context_limit=200)

        cursor = db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM compactions WHERE channel = ?",
            ('chan-upsert',),
        )
        assert cursor.fetchone()[0] == 1  # Only one row (upserted)


# ── Tests: ToolCallService — store() with tool_call_id ────────────────────────

class TestToolCallServiceWithToolCallId:
    """Tests that target the tool_call_id column added to ToolCallService."""

    def _make_transcript(self, db, channel='tc-svc-test'):
        return _insert_transcript(db, channel, 'user', 'prompt')

    def test_store_with_tool_call_id_persisted(self, db):
        """store() saves tool_call_id to the tool_calls row."""
        from services.tool_call_service import ToolCallService

        tid = self._make_transcript(db)
        svc = ToolCallService()
        svc.store(tid, 'search', {'q': 'test'}, 'results', tool_call_id='tc_xyz')

        cursor = db.cursor()
        cursor.execute("SELECT tool_call_id FROM tool_calls WHERE transcript_id = ?", (tid,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'tc_xyz'

    def test_store_without_tool_call_id_is_null(self, db):
        """store() without tool_call_id leaves the column NULL."""
        from services.tool_call_service import ToolCallService

        tid = self._make_transcript(db, 'tc-svc-null')
        svc = ToolCallService()
        svc.store(tid, 'memory', {}, 'ok')

        cursor = db.cursor()
        cursor.execute("SELECT tool_call_id FROM tool_calls WHERE transcript_id = ?", (tid,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None

    def test_store_batch_extracts_tc_id_from_input(self, db):
        """store_batch() reads tc['id'] from each tool call and stores it as tool_call_id."""
        from services.tool_call_service import ToolCallService

        tid = self._make_transcript(db, 'tc-batch-id')
        svc = ToolCallService()

        tool_calls = [
            {'id': 'batch_tc_1', 'name': 'search', 'input': {'q': 'news'}},
            {'id': 'batch_tc_2', 'name': 'memory', 'input': {}},
        ]
        results = [
            {'result': 'news results', 'status': 'ok'},
            {'result': 'memory ok', 'status': 'ok'},
        ]
        svc.store_batch(tid, tool_calls, results, ephemeral=False)

        cursor = db.cursor()
        cursor.execute(
            "SELECT tool_name, tool_call_id FROM tool_calls WHERE transcript_id = ? ORDER BY id",
            (tid,),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][0] == 'search'
        assert rows[0][1] == 'batch_tc_1'
        assert rows[1][0] == 'memory'
        assert rows[1][1] == 'batch_tc_2'

    def test_store_batch_without_id_field_stores_null(self, db):
        """store_batch() handles missing 'id' key gracefully — stores NULL."""
        from services.tool_call_service import ToolCallService

        tid = self._make_transcript(db, 'tc-batch-noid')
        svc = ToolCallService()

        tool_calls = [{'name': 'some_tool', 'input': {}}]  # no 'id' key
        results = [{'result': 'ok', 'status': 'ok'}]
        svc.store_batch(tid, tool_calls, results, ephemeral=False)

        cursor = db.cursor()
        cursor.execute(
            "SELECT tool_call_id FROM tool_calls WHERE transcript_id = ?",
            (tid,),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None


# ── Tests: ToolCallService.get_by_transcript_ids() ────────────────────────────

class TestGetByTranscriptIds:
    def test_returns_empty_dict_for_empty_list(self, db):
        from services.tool_call_service import ToolCallService
        result = ToolCallService().get_by_transcript_ids([])
        assert result == {}

    def test_groups_records_by_transcript_id(self, db):
        """Results are keyed by transcript_id with correct lists."""
        from services.tool_call_service import ToolCallService

        tid1 = _insert_transcript(db, 'chan-grp', 'user', 'msg1')
        tid2 = _insert_transcript(db, 'chan-grp', 'assistant', 'msg2')

        _insert_tool_call(db, tid1, 'tool_a', {}, 'r1', tool_call_id='tc_1')
        _insert_tool_call(db, tid2, 'tool_b', {}, 'r2', tool_call_id='tc_2')
        _insert_tool_call(db, tid2, 'tool_c', {}, 'r3', tool_call_id='tc_3')

        svc = ToolCallService()
        result = svc.get_by_transcript_ids([tid1, tid2])

        assert set(result.keys()) == {tid1, tid2}
        assert len(result[tid1]) == 1
        assert result[tid1][0]['tool_name'] == 'tool_a'
        assert len(result[tid2]) == 2
        tool_names = {r['tool_name'] for r in result[tid2]}
        assert tool_names == {'tool_b', 'tool_c'}

    def test_excludes_ephemeral_when_flag_false(self, db):
        """include_ephemeral=False filters out ephemeral records."""
        from services.tool_call_service import ToolCallService

        tid = _insert_transcript(db, 'chan-eph-filter', 'user', 'x')
        _insert_tool_call(db, tid, 'ephemeral_tool', {}, 'e_result', ephemeral=1)
        _insert_tool_call(db, tid, 'real_tool', {}, 'real_result', ephemeral=0)

        svc = ToolCallService()
        result = svc.get_by_transcript_ids([tid], include_ephemeral=False)

        assert tid in result
        assert len(result[tid]) == 1
        assert result[tid][0]['tool_name'] == 'real_tool'

    def test_includes_tool_call_id_field_in_returned_dicts(self, db):
        """Each returned dict contains the tool_call_id field."""
        from services.tool_call_service import ToolCallService

        tid = _insert_transcript(db, 'chan-tc-field', 'assistant', 'calling')
        _insert_tool_call(db, tid, 'search', {}, 'results', tool_call_id='tc_fieldtest')

        svc = ToolCallService()
        result = svc.get_by_transcript_ids([tid])

        assert tid in result
        assert result[tid][0]['tool_call_id'] == 'tc_fieldtest'

    def test_single_id_not_in_list_returns_empty(self, db):
        """A transcript_id with no matching tool_calls is absent from the dict."""
        from services.tool_call_service import ToolCallService

        tid = _insert_transcript(db, 'chan-notc', 'user', 'nothing')
        result = ToolCallService().get_by_transcript_ids([tid])
        assert result == {}

    def test_unknown_transcript_ids_ignored(self, db):
        """IDs not present in tool_calls don't appear in output — no crash."""
        from services.tool_call_service import ToolCallService
        result = ToolCallService().get_by_transcript_ids([99999, 88888])
        assert result == {}


# ── Tests: ToolCallService.get_find_tools_results() with list input ────────────

class TestGetFindToolsResultsList:
    def test_accepts_list_of_ids(self, db):
        """get_find_tools_results() accepts a list and returns discovered tools."""
        from services.tool_call_service import ToolCallService

        tid1 = _insert_transcript(db, 'chan-ft-list', 'user', 'find me tools')
        tid2 = _insert_transcript(db, 'chan-ft-list', 'user', 'find more')

        _insert_tool_call(
            db, tid1, 'find_tools',
            params={'discovered': ['tool_alpha', 'tool_beta']},
            result='Found 2',
        )
        _insert_tool_call(
            db, tid2, 'find_tools',
            params={'discovered': ['tool_gamma']},
            result='Found 1',
        )

        svc = ToolCallService()
        result = svc.get_find_tools_results([tid1, tid2])

        assert set(result) == {'tool_alpha', 'tool_beta', 'tool_gamma'}

    def test_deduplicates_across_multiple_ids(self, db):
        """Duplicate tool names across multiple transcript_ids appear only once."""
        from services.tool_call_service import ToolCallService

        tid1 = _insert_transcript(db, 'chan-ft-dedup', 'user', 'first call')
        tid2 = _insert_transcript(db, 'chan-ft-dedup', 'user', 'second call')

        _insert_tool_call(
            db, tid1, 'find_tools',
            params={'discovered': ['shared_tool', 'unique_a']},
            result='',
        )
        _insert_tool_call(
            db, tid2, 'find_tools',
            params={'discovered': ['shared_tool', 'unique_b']},
            result='',
        )

        svc = ToolCallService()
        result = svc.get_find_tools_results([tid1, tid2])

        # shared_tool must appear exactly once
        assert result.count('shared_tool') == 1
        assert 'unique_a' in result
        assert 'unique_b' in result

    def test_empty_list_returns_empty(self, db):
        from services.tool_call_service import ToolCallService
        result = ToolCallService().get_find_tools_results([])
        assert result == []

    def test_single_int_still_accepted(self, db):
        """Backward compat: passing a single int (not a list) still works."""
        from services.tool_call_service import ToolCallService

        tid = _insert_transcript(db, 'chan-ft-int', 'user', 'find tools')
        _insert_tool_call(
            db, tid, 'find_tools',
            params={'discovered': ['single_tool']},
            result='',
        )

        svc = ToolCallService()
        result = svc.get_find_tools_results(tid)  # int, not list
        assert result == ['single_tool']

    def test_ignores_non_find_tools_rows_across_ids(self, db):
        """Rows for other tool names are ignored even when queried by list of ids."""
        from services.tool_call_service import ToolCallService

        tid = _insert_transcript(db, 'chan-ft-ignore', 'user', 'run')
        _insert_tool_call(
            db, tid, 'memory',
            params={'discovered': ['should_not_appear']},
            result='',
        )
        _insert_tool_call(
            db, tid, 'find_tools',
            params={'discovered': ['should_appear']},
            result='',
        )

        svc = ToolCallService()
        result = svc.get_find_tools_results([tid])
        assert result == ['should_appear']


# ── Tests: build_messages() integration with real tool pipeline data ──────────

class TestBuildMessagesIntegration:
    """End-to-end data-flow tests: insert via transcript_service/tool_call_service,
    then verify build_messages() produces the correct structure."""

    def test_full_turn_user_assistant_tool_round_trip(self, db):
        """Insert a complete user→assistant+tool_call→tool_result turn via services,
        then verify build_messages returns the right structure."""
        from services.context_window_service import build_messages
        from services.tool_call_service import ToolCallService

        channel = 'chan-integration-full'

        # 1. User turn
        with patch('services.transcript_service._embed_entry'):
            from services.transcript_service import append as ts_append
            ts_append(channel, 'user', 'What is the weather in Malta?')

        # 2. Assistant turn (calls weather tool)
        asst_id = _insert_transcript(channel=channel, role='assistant',
                                      content='Let me check the weather.', db=db)

        # 3. Store the tool call (linked to assistant entry)
        svc = ToolCallService()
        svc.store(asst_id, 'weather_tool', {'location': 'Malta'}, 'Sunny, 25°C',
                  tool_call_id='tc_weather_01')

        # 4. Tool result in transcript
        _insert_transcript(db, channel, 'tool', 'Sunny, 25°C',
                           tool_call_id='tc_weather_01', tool_name='weather_tool')

        messages = build_messages(channel)

        roles = [m['role'] for m in messages]
        assert roles == ['user', 'assistant', 'tool']

        asst_msg = messages[1]
        assert 'tool_calls' in asst_msg
        assert asst_msg['tool_calls'][0]['id'] == 'tc_weather_01'
        assert asst_msg['tool_calls'][0]['name'] == 'weather_tool'

        tool_msg = messages[2]
        assert tool_msg['tool_call_id'] == 'tc_weather_01'
        assert tool_msg['name'] == 'weather_tool'

    def test_build_messages_channels_are_isolated(self, db):
        """Entries from channel A never appear in channel B's messages."""
        from services.context_window_service import build_messages

        _insert_transcript(db, 'chan-iso-a', 'user', 'Channel A message')
        _insert_transcript(db, 'chan-iso-b', 'user', 'Channel B message')

        result_a = build_messages('chan-iso-a')
        result_b = build_messages('chan-iso-b')

        assert len(result_a) == 1
        assert result_a[0]['content'] == 'Channel A message'

        assert len(result_b) == 1
        assert result_b[0]['content'] == 'Channel B message'

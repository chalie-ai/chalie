"""Tests for Phase 4D: Compaction Service.

Tests cover:
- check_and_compact() trigger logic
- get_compaction() retrieval
- get_entries_since() delegation
- _run_compaction() LLM call and storage
"""

from unittest.mock import patch, MagicMock


def _insert_entries(conn, topic, count, content_template='Message {}'):
    """Insert transcript entries and return their IDs."""
    ids = []
    for i in range(count):
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            (topic, 'user' if i % 2 == 0 else 'assistant', content_template.format(i)),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    return ids


class TestGetCompaction:
    def test_returns_none_when_no_compaction(self, db):
        from services.compaction_service import get_compaction
        assert get_compaction('nonexistent-topic') is None

    def test_returns_stored_compaction(self, db):
        from services.compaction_service import get_compaction

        # Seed transcript entries so FK constraint is satisfied
        ids = _insert_entries(db, 'test-topic', 10)
        watermark = ids[-1]

        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Summary of conversation', watermark, 50, '2026-03-20T10:00:00+00:00'),
        )
        db.commit()

        result = get_compaction('test-topic')
        assert result is not None
        assert result['compacted_text'] == 'Summary of conversation'
        assert result['compacted_up_to_id'] == watermark
        assert result['token_count'] == 50


class TestCheckAndCompact:
    def test_returns_false_for_empty_topic(self, db):
        from services.compaction_service import check_and_compact
        assert check_and_compact('', 32000) is False

    def test_returns_false_when_too_few_entries(self, db):
        from services.compaction_service import check_and_compact

        # Only 2 entries — below _MIN_ENTRIES_TO_COMPACT (4)
        with patch('services.compaction_service.get_entries_since', return_value=[
            {'id': 1, 'content': 'Hello'},
            {'id': 2, 'content': 'Hi there'},
        ]):
            assert check_and_compact('test', 32000) is False

    def test_returns_false_when_below_threshold(self, db):
        from services.compaction_service import check_and_compact

        entries = [{'id': i, 'content': f'Short msg {i}'} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries):
            # 5 short entries, well below 85% of 32000
            assert check_and_compact('test', 32000) is False

    def test_large_budget_fraction_threshold_fires_when_exceeded(self, db):
        """Large model context budget uses fraction-based threshold (85% of 200K = 170K)."""
        from services.compaction_service import check_and_compact

        # ~175K tokens of content — above 85% of 200K budget (170K)
        long_content = 'word ' * 7000  # ~8750 tokens each
        entries = [{'id': i, 'content': long_content} for i in range(20)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            result = check_and_compact('test', 200_000)

        assert result is True
        mock_run.assert_called_once()

    def test_small_budget_uses_fraction_threshold(self, db):
        """Small context budget should use fraction-based threshold (below absolute ceiling)."""
        from services.compaction_service import check_and_compact

        # Budget=10000, fraction threshold=8500, absolute=24000 → min=8500
        # ~10K tokens of content — above 8500 fraction threshold
        long_content = 'word ' * 2000  # ~2500 tokens each
        entries = [{'id': i, 'content': long_content} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            result = check_and_compact('test', 10_000)

        assert result is True
        mock_run.assert_called_once()

class TestRunCompaction:
    def test_stores_compaction_result(self, db):
        from services.compaction_service import _run_compaction

        # Seed transcript entries so FK constraint is satisfied
        ids = _insert_entries(db, 'test-topic', 4, 'Weather msg {}')
        entries = [
            {'id': ids[0], 'role': 'user', 'content': 'What is the weather?'},
            {'id': ids[1], 'role': 'assistant', 'content': 'It is sunny in Malta today.'},
            {'id': ids[2], 'role': 'user', 'content': 'Thanks!'},
            {'id': ids[3], 'role': 'assistant', 'content': 'You are welcome.'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='- Weather in Malta: sunny')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', '', entries)

        assert result is True

        # Verify stored in database
        cursor = db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM compactions WHERE channel = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == '- Weather in Malta: sunny'
        assert row[1] == ids[3]  # highest entry id

    def test_updates_existing_compaction(self, db):
        from services.compaction_service import _run_compaction

        # Seed transcript entries — old batch + new batch
        old_ids = _insert_entries(db, 'test-topic', 5, 'Old msg {}')
        new_ids = _insert_entries(db, 'test-topic', 2, 'New msg {}')

        # Insert initial compaction covering old entries
        db.execute(
            "INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Old summary', old_ids[-1], 20, '2026-03-20T09:00:00'),
        )
        db.commit()

        entries = [
            {'id': new_ids[0], 'role': 'user', 'content': 'New information here'},
            {'id': new_ids[1], 'role': 'assistant', 'content': 'Noted and processed'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Updated summary with new info')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', 'Old summary', entries)

        assert result is True

        cursor = db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM compactions WHERE channel = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row[0] == 'Updated summary with new info'
        assert row[1] == new_ids[1]

    def test_includes_previous_text_in_prompt(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            _run_compaction('test', 'Previous context here', entries)

        # Verify the user message includes the previous summary
        call_args = mock_llm.send_message.call_args
        user_message = call_args[0][1]
        assert '## Previous Summary' in user_message
        assert 'Previous context here' in user_message
        assert '## New Conversation Turns' in user_message

    def test_includes_tool_name_in_formatted_entries(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [
            {'id': ids[0], 'role': 'tool', 'content': 'Search results', 'tool_name': 'search'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            _run_compaction('test', '', entries)

        user_message = mock_llm.send_message.call_args[0][1]
        assert '[tool — search]' in user_message

    def test_returns_false_on_empty_llm_response(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='   ')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test', '', entries)

        assert result is False

    def test_returns_false_on_llm_error(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        with patch('services.llm_service.create_refreshable_llm_service', side_effect=Exception('LLM down')):
            result = _run_compaction('test', '', entries)

        assert result is False


class TestCompactionTopicContext:
    def test_backward_compat_without_context(self, db):
        """get_compaction still works without _context."""
        from services.compaction_service import get_compaction
        result = get_compaction('nonexistent')
        assert result is None


import pytest


@pytest.mark.unit
class TestStripEntry:
    """Unit tests for _strip_entry — the pure linguistic-fluff stripper."""

    def test_internal_passthrough_returns_content_unchanged(self):
        """Internal role bypasses all processing — HTML, newlines, filler all preserved."""
        from services.compaction_service import _strip_entry
        content = '<p>I think a lot</p>\n\n\n\nSure, just do it.'
        assert _strip_entry(content, 'internal') == content

    def test_tool_strips_html_tags(self):
        """Tool role removes HTML tags, leaving only text content."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('<p>text</p>', 'tool') == 'text'

    def test_tool_strips_nested_html(self):
        """Nested HTML tags are all removed from tool output."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('<div><span>Result</span></div>', 'tool') == 'Result'

    def test_tool_collapses_three_or_more_newlines_to_two(self):
        """Three or more consecutive newlines collapse to exactly two."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('line1\n\n\n\nline2', 'tool') == 'line1\n\nline2'
        assert _strip_entry('a\n\n\nb', 'tool') == 'a\n\nb'

    def test_tool_preserves_two_newlines(self):
        """Two consecutive newlines are not touched by tool processing."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('a\n\nb', 'tool') == 'a\n\nb'

    def test_tool_strips_html_and_collapses_newlines_together(self):
        """HTML stripping and newline collapsing both apply in a single pass."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('<div>Result</div>\n\n\n\n<span>More</span>', 'tool') == 'Result\n\nMore'

    def test_tool_preserves_json_structured_data(self):
        """JSON/structured data in tool output passes through intact."""
        from services.compaction_service import _strip_entry
        import json
        data = json.dumps({'key': 'value', 'count': 42})
        assert _strip_entry(data, 'tool') == data

    def test_user_filler_sure_removed_at_start(self):
        """'Sure,' at sentence start is stripped; remaining content preserved."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Sure, I will check that.', 'user') == 'I will check that.'

    def test_user_filler_ill_removed_at_start(self):
        """\"I'll\" filler at sentence start is stripped."""
        from services.compaction_service import _strip_entry
        assert _strip_entry("I'll do it.", 'user') == 'do it.'

    def test_assistant_filler_let_me_removed_at_start(self):
        """'Let me' filler at sentence start is stripped, articles also removed."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Let me look at the logs.', 'assistant') == 'look at logs.'

    def test_assistant_filler_certainly_removed_at_start(self):
        """'Certainly,' at start of assistant message is stripped."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Certainly, the config is broken.', 'assistant') == 'config is broken.'

    def test_user_hedge_i_think_removed(self):
        """'I think' hedge phrase is removed; article and remaining text stripped normally."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('I think the file is broken.', 'user') == 'file is broken.'

    def test_user_hedge_it_seems_like_removed(self):
        """'It seems like' hedge is removed along with trailing article."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('It seems like a problem.', 'user') == 'problem.'

    def test_assistant_hedge_basically_removed(self):
        """'basically' hedge word is removed mid-sentence."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('basically the answer is yes', 'assistant') == 'answer is yes'

    def test_user_article_the_removed(self):
        """'the' article is stripped from natural language text."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('the config file is missing', 'user') == 'config file is missing'

    def test_user_articles_a_and_an_removed(self):
        """'a' and 'an' articles are both stripped."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('open a terminal and run an update', 'user') == 'open terminal and run update'

    def test_user_code_block_content_survives_stripping(self):
        """Content inside fenced code blocks is not modified by any stripping rule."""
        from services.compaction_service import _strip_entry
        result = _strip_entry(
            'Check ```python\nthe_var = a + 1\n``` for errors.',
            'user',
        )
        assert result == 'Check ```python\nthe_var = a + 1\n``` for errors.'

    def test_user_inline_code_content_survives_stripping(self):
        """Content inside backtick inline code is not modified."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Call `the_method(a)` here.', 'user') == 'Call `the_method(a)` here.'

    def test_user_url_survives_article_stripping(self):
        """URLs with 'the' or 'a' in their path components are not altered."""
        from services.compaction_service import _strip_entry
        result = _strip_entry('See https://example.com/the/path for the docs.', 'user')
        assert result == 'See https://example.com/the/path for docs.'

    def test_user_absolute_file_path_survives_article_stripping(self):
        """Absolute paths containing 'the' components are preserved."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Edit /etc/the/config now.', 'user') == 'Edit /etc/the/config now.'

    def test_user_relative_file_path_survives_article_stripping(self):
        """Relative paths starting with ./ are preserved."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Load ./the/settings.json file.', 'user') == 'Load ./the/settings.json file.'

    def test_empty_content_user(self):
        """Empty string input returns empty string for user role."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('', 'user') == ''

    def test_empty_content_tool(self):
        """Empty string input returns empty string for tool role."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('', 'tool') == ''

    def test_empty_content_internal(self):
        """Empty string input returns empty string for internal role."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('', 'internal') == ''

    def test_content_that_is_only_a_code_block(self):
        """A message consisting solely of a fenced code block is returned intact."""
        from services.compaction_service import _strip_entry
        code = '```python\nprint("hello")\n```'
        assert _strip_entry(code, 'user') == code

    def test_assistant_code_block_with_articles_inside_preserved(self):
        """Articles inside fenced code blocks are not stripped even for assistant role."""
        from services.compaction_service import _strip_entry
        code_block = '```\nthe_var = a_value\n```'
        result = _strip_entry(code_block, 'assistant')
        assert 'the_var' in result
        assert 'a_value' in result

    def test_user_whitespace_collapsed_to_single_space(self):
        """Multiple spaces within a line are collapsed to a single space."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('hello   world', 'user') == 'hello world'

    def test_sure_thing_filler_fully_stripped(self):
        """'Sure thing,' should be stripped as a whole phrase, not just 'Sure'."""
        from services.compaction_service import _strip_entry
        assert _strip_entry('Sure thing, I will handle it.', 'assistant') == 'I will handle it.'

    def test_null_bytes_in_content_do_not_crash(self):
        """Content with null bytes is sanitised before placeholder logic runs."""
        from services.compaction_service import _strip_entry
        result = _strip_entry('hello\x00world', 'user')
        assert result == 'helloworld'

    def test_hedge_at_line_boundary_preserves_newline(self):
        """Hedge removal does not absorb newlines between lines."""
        from services.compaction_service import _strip_entry
        result = _strip_entry('I think\nline two', 'user')
        assert '\n' in result
        assert 'line two' in result

    def test_unknown_role_returns_content_unchanged(self):
        """Unknown roles pass through without any stripping."""
        from services.compaction_service import _strip_entry
        content = 'Sure, I think the answer is basically yes.'
        assert _strip_entry(content, 'custom_role') == content

    def test_please_not_stripped_from_user_directive(self):
        """'Please' is an instruction marker and should not be removed."""
        from services.compaction_service import _strip_entry
        result = _strip_entry('Please use port 8080.', 'user')
        assert 'Please' in result

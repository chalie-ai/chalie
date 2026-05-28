"""Unit tests for the _extract_compaction_summary() module-level parser.

Tests the regex-based extractor that parses <summary>...</summary> from
raw LLM compaction output. No DB or LLM calls.
"""

import pytest

pytestmark = pytest.mark.unit


class TestExtractCompactionSummary:
    def test_extracts_summary_from_full_output(self):
        from services.message_processor import _extract_compaction_summary
        raw = '<analysis>notes here</analysis><summary>Person: Alex\nNow: coding</summary>'
        result = _extract_compaction_summary(raw)
        assert result == 'Person: Alex\nNow: coding'

    def test_extracts_summary_only_tag(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary('<summary>just the summary</summary>')
        assert result == 'just the summary'

    def test_returns_none_when_no_summary_tags(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary('plain text no tags here')
        assert result is None

    def test_returns_none_on_none_input(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary(None)
        assert result is None

    def test_strips_leading_trailing_whitespace_inside_tags(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary('<summary>   trimmed content   </summary>')
        assert result == 'trimmed content'

    def test_case_insensitive_tags(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary('<SUMMARY>upper case</SUMMARY>')
        assert result == 'upper case'

    def test_multiline_summary_preserved(self):
        from services.message_processor import _extract_compaction_summary
        raw = '<summary>\nPerson: Alice\nNow: planning\nOpen: fix the bug\n</summary>'
        result = _extract_compaction_summary(raw)
        assert 'Person: Alice' in result
        assert 'Now: planning' in result
        assert 'Open: fix the bug' in result

    def test_analysis_block_before_summary_does_not_bleed(self):
        from services.message_processor import _extract_compaction_summary
        raw = (
            '<analysis>internal reasoning\n'
            'should not appear in result</analysis>\n'
            '<summary>clean summary only</summary>'
        )
        result = _extract_compaction_summary(raw)
        assert result == 'clean summary only'
        assert 'internal reasoning' not in result

    def test_returns_first_summary_tag_when_multiple(self):
        from services.message_processor import _extract_compaction_summary
        raw = '<summary>first</summary><summary>second</summary>'
        result = _extract_compaction_summary(raw)
        assert result == 'first'

    def test_analysis_only_returns_none(self):
        from services.message_processor import _extract_compaction_summary
        result = _extract_compaction_summary('<analysis>only analysis, no summary</analysis>')
        assert result is None

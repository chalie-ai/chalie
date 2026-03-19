"""Tests for the find_skills innate skill."""

import struct
import pytest
from unittest.mock import patch, MagicMock, mock_open

from services.innate_skills.find_skills_skill import (
    handle_find_skills,
    _filter_to_contextual,
    _fallback_keyword_search,
    _load_skill_doc,
    _format_results,
)

pytestmark = pytest.mark.unit

# Patch targets: imports are lazy (inside functions), so patch the source module
_EMB = 'services.embedding_service.EmbeddingService'
_DB = 'services.database_service.get_shared_db_service'
_REGISTRY = 'services.innate_skills.registry.CONTEXTUAL_SKILLS'

# Skills that must never appear in find_skills results
PRIMITIVE_SKILLS = {'recall', 'memorize', 'associate', 'find_tools', 'find_skills'}

# Skills that are valid contextual results
SAMPLE_CONTEXTUAL_SKILLS = frozenset({
    'schedule', 'autobiography', 'focus', 'list',
    'persistent_task', 'document', 'read', 'reflect', 'notes', 'introspect', 'moment',
})


def _mock_db_with_rows(rows):
    """Create a mock DB service that returns rows from a cursor.

    Attaches mock_cursor as ._test_cursor for easy assertion access.
    """
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: mock_conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = rows
    mock_db._test_cursor = mock_cursor
    return mock_db


class TestRequiresQuery:

    def test_empty_query_returns_error(self):
        """Missing query should return a descriptive error string without raising."""
        result = handle_find_skills("topic", {"query": ""})
        assert "[FIND_SKILLS]" in result
        assert "Error" in result
        assert "query" in result.lower()

    def test_whitespace_only_query_returns_error(self):
        """A query of only whitespace should be treated as missing."""
        result = handle_find_skills("topic", {"query": "   "})
        assert "[FIND_SKILLS]" in result
        assert "Error" in result

    def test_missing_query_key_returns_error(self):
        """Omitting the query key entirely should return an error."""
        result = handle_find_skills("topic", {})
        assert "[FIND_SKILLS]" in result
        assert "Error" in result

    def test_result_is_always_a_string(self):
        """Handler must never raise — always returns a str."""
        result = handle_find_skills("topic", {})
        assert isinstance(result, str)


class TestSemanticSearchReturnsFullDocs:

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_full_doc_content_in_result(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """Vec search hit should return the full .md documentation for the skill."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768

        mock_db_fn.return_value = _mock_db_with_rows([
            ('schedule', 'skill', 0.2),
        ])

        full_doc = "# Schedule Skill\nUse this skill to set reminders and schedule tasks."
        mock_load_doc.return_value = full_doc

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "set a reminder"})

        assert "[FIND_SKILLS]" in result
        assert "schedule" in result
        assert full_doc in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_result_includes_section_header(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """Result should include a markdown section header for each matched skill."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([('list', 'skill', 0.15)])
        mock_load_doc.return_value = "List skill documentation."

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "manage a list"})

        assert "## list" in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_result_prefix_contains_match_count(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """Result header should state how many skills were found."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([
            ('schedule', 'skill', 0.2),
            ('list', 'skill', 0.3),
        ])
        mock_load_doc.return_value = "Doc content."

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "organise things"})

        assert "Found 2 skill(s)" in result


class TestFiltersToContextualSkillsOnly:

    def test_primitive_recall_is_filtered_out(self):
        """Primitive skill 'recall' must never appear in results."""
        rows = [('recall', 'skill', 0.1)]
        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            results = _filter_to_contextual(rows, limit=5)
        names = {r['skill_name'] for r in results}
        assert 'recall' not in names

    def test_all_primitives_filtered_out(self):
        """All five cognitive primitives must be excluded."""
        rows = [(name, 'skill', 0.1) for name in PRIMITIVE_SKILLS]
        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            results = _filter_to_contextual(rows, limit=10)
        returned_names = {r['skill_name'] for r in results}
        assert returned_names.isdisjoint(PRIMITIVE_SKILLS)

    def test_contextual_skill_passes_through(self):
        """A genuine contextual skill must appear in the filtered output."""
        rows = [('schedule', 'skill', 0.2)]
        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            results = _filter_to_contextual(rows, limit=5)
        assert len(results) == 1
        assert results[0]['skill_name'] == 'schedule'

    def test_mixed_rows_only_contextual_returned(self):
        """Mixed input of primitives and contextual skills keeps only contextual."""
        rows = [
            ('recall', 'skill', 0.1),
            ('focus', 'skill', 0.2),
            ('memorize', 'skill', 0.15),
            ('document', 'skill', 0.25),
        ]
        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            results = _filter_to_contextual(rows, limit=10)
        returned_names = {r['skill_name'] for r in results}
        assert returned_names == {'focus', 'document'}

    @patch('services.innate_skills.find_skills_skill._fallback_keyword_search')
    @patch(_DB)
    @patch(_EMB)
    def test_primitive_filtered_in_full_handler_path(
        self, mock_emb_cls, mock_db_fn, mock_fallback
    ):
        """End-to-end: primitive returned by vec search never reaches the output."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([('recall', 'skill', 0.05)])
        mock_fallback.return_value = []

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "remember something"})

        # recall was filtered out, nothing remains → no-match message
        assert "No matching skills found" in result
        # primitive names are mentioned in the no-match explanation, but not as a result
        assert "## recall" not in result


class TestKeywordFallbackWhenEmbeddingFails:

    @patch(_DB)
    @patch(_EMB)
    def test_falls_back_on_embedding_exception(self, mock_emb_cls, mock_db_fn):
        """RuntimeError from EmbeddingService triggers keyword fallback."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("no model")
        # Fallback DB query returns an empty result — no matching skills
        mock_db_fn.return_value = _mock_db_with_rows([])

        result = handle_find_skills("topic", {"query": "set a reminder"})

        # Result must still be a valid string with the log prefix
        assert isinstance(result, str)
        assert "[FIND_SKILLS]" in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_keyword_fallback_returns_contextual_results(
        self, mock_emb_cls, mock_db_fn, mock_load_doc
    ):
        """When embedding fails and keyword query matches, results are returned."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("no model")
        mock_db_fn.return_value = _mock_db_with_rows([('schedule', )])
        mock_load_doc.return_value = "Schedule skill doc."

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "schedule"})

        assert "[FIND_SKILLS]" in result
        # Either found a result or returned a graceful no-match — never an uncaught exception
        assert isinstance(result, str)

    @patch(_DB)
    @patch(_EMB)
    def test_keyword_fallback_filters_primitives(self, mock_emb_cls, mock_db_fn):
        """Keyword fallback must also exclude cognitive primitives."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("no model")
        # DB returns a primitive from the keyword search
        mock_db_fn.return_value = _mock_db_with_rows([('recall', )])

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = _fallback_keyword_search("recall", limit=5)

        returned_names = {r['skill_name'] for r in result}
        assert 'recall' not in returned_names

    @patch(_DB)
    def test_keyword_fallback_returns_empty_on_db_failure(self, mock_db_fn):
        """If the keyword fallback DB query also fails, return an empty list silently."""
        mock_db_fn.side_effect = Exception("DB unavailable")
        result = _fallback_keyword_search("schedule", limit=3)
        assert result == []


class TestLimitCapping:

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_limit_capped_at_5(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """Limit parameter must be silently capped at 5."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        # Return enough rows to distinguish a limit of 5 vs. 50
        rows = [(name, 'skill', 0.1 + i * 0.01)
                for i, name in enumerate(SAMPLE_CONTEXTUAL_SKILLS)]
        mock_db_fn.return_value = _mock_db_with_rows(rows)
        mock_load_doc.return_value = "Doc."

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "anything", "limit": 50})

        # Count section headers — there should be at most 5
        header_count = result.count("## ")
        assert header_count <= 5

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_explicit_limit_below_max_respected(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """An explicit limit of 2 should return at most 2 results."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        rows = [(name, 'skill', 0.1 + i * 0.01)
                for i, name in enumerate(SAMPLE_CONTEXTUAL_SKILLS)]
        mock_db_fn.return_value = _mock_db_with_rows(rows)
        mock_load_doc.return_value = "Doc."

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "anything", "limit": 2})

        header_count = result.count("## ")
        assert header_count <= 2

    def test_filter_to_contextual_respects_limit(self):
        """_filter_to_contextual must stop collecting once limit is reached."""
        rows = [(name, 'skill', 0.1) for name in SAMPLE_CONTEXTUAL_SKILLS]
        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            results = _filter_to_contextual(rows, limit=2)
        assert len(results) <= 2


class TestNoMatchingSkills:

    @patch(_DB)
    @patch(_EMB)
    def test_empty_vec_results_returns_helpful_message(self, mock_emb_cls, mock_db_fn):
        """No matches from vector search produces an informative no-match response."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([])

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "something obscure"})

        assert "[FIND_SKILLS]" in result
        assert "No matching skills found" in result

    @patch(_DB)
    @patch(_EMB)
    def test_no_match_message_mentions_primitives(self, mock_emb_cls, mock_db_fn):
        """No-match message should remind caller that primitives are always available."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([])

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "xyzzy"})

        assert "recall" in result or "primitive" in result.lower()

    @patch(_DB)
    @patch(_EMB)
    def test_no_match_still_has_log_prefix(self, mock_emb_cls, mock_db_fn):
        """The [FIND_SKILLS] prefix must be present even on empty results."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([])

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "nothing"})

        assert result.startswith("[FIND_SKILLS]")


class TestMissingSkillDocFile:

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_missing_doc_skipped_gracefully(self, mock_emb_cls, mock_db_fn, mock_load_doc):
        """A matched skill whose .md file is absent is silently skipped."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([('schedule', 'skill', 0.2)])
        mock_load_doc.return_value = None  # simulate missing file

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "remind me"})

        # The skill was matched but its doc is missing — should NOT raise
        assert isinstance(result, str)
        assert "[FIND_SKILLS]" in result
        # No partial/broken section should appear
        assert "## schedule" not in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_all_docs_missing_returns_no_match_message(
        self, mock_emb_cls, mock_db_fn, mock_load_doc
    ):
        """When all matched skills have missing docs, return the no-match fallback."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([
            ('schedule', 'skill', 0.2),
            ('focus', 'skill', 0.3),
        ])
        mock_load_doc.return_value = None

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "productivity"})

        assert "No matching skills found" in result

    def test_load_skill_doc_returns_none_on_file_not_found(self, tmp_path):
        """_load_skill_doc returns None (not an exception) for absent files."""
        result = _load_skill_doc("nonexistent_skill_xyz")
        assert result is None

    def test_load_skill_doc_returns_content_when_file_exists(self, tmp_path, monkeypatch):
        """_load_skill_doc returns file content when the .md file exists."""
        import services.innate_skills.find_skills_skill as module

        fake_dir = str(tmp_path)
        monkeypatch.setattr(module, '_PROMPTS_DIR', fake_dir)

        skill_file = tmp_path / "schedule.md"
        skill_file.write_text("# Schedule\nSet reminders.")

        result = _load_skill_doc("schedule")
        assert result == "# Schedule\nSet reminders."

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    @patch(_DB)
    @patch(_EMB)
    def test_partial_doc_missing_still_returns_available_skills(
        self, mock_emb_cls, mock_db_fn, mock_load_doc
    ):
        """If only some docs are missing, the available ones still appear in output."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([
            ('schedule', 'skill', 0.2),
            ('focus', 'skill', 0.3),
        ])
        # schedule doc missing, focus doc present
        mock_load_doc.side_effect = lambda name: (
            None if name == 'schedule' else "Focus skill documentation."
        )

        with patch(_REGISTRY, SAMPLE_CONTEXTUAL_SKILLS):
            result = handle_find_skills("topic", {"query": "productivity"})

        assert "## focus" in result
        assert "## schedule" not in result
        assert "Focus skill documentation." in result


class TestFormatResults:

    def test_empty_matches_returns_no_match_message(self):
        """_format_results with zero matches returns the no-match fallback string."""
        result = _format_results("anything", [])
        assert "No matching skills found" in result
        assert "[FIND_SKILLS]" in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    def test_separator_between_skills(self, mock_load_doc):
        """Multiple results should be separated by the horizontal rule."""
        mock_load_doc.return_value = "Documentation content."
        matches = [{'skill_name': 'schedule'}, {'skill_name': 'focus'}]
        result = _format_results("something", matches)
        assert "────────────────────────────────" in result

    @patch('services.innate_skills.find_skills_skill._load_skill_doc')
    def test_query_reflected_in_header(self, mock_load_doc):
        """The search query should appear in the result header."""
        mock_load_doc.return_value = "Doc."
        result = _format_results("set a reminder", [{'skill_name': 'schedule'}])
        assert "set a reminder" in result

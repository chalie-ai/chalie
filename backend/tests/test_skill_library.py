"""Feature tests for the Self-Refining Skill Library pure functions (TKT-579).

Three behaviours under test:

1. _rrf_merge() — reciprocal rank fusion of vec + FTS result rows.
   Verifies: correct merging, dedup, FTS-only path, vec-only path,
   empty inputs, and score-ordering invariant.

2. _parse_associations() — LLM JSON response parser.
   Verifies: valid JSON, fenced code blocks, empty array, non-list,
   empty string, and malformed JSON all handled correctly.

3. _safe_parse_summary() — extracts summary from behavioural pattern JSON.
   Verifies: happy path, missing summary key, invalid JSON, None input.

All functions are pure: no IO, no collaborators, no network. Zero mocks.
"""

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. _rrf_merge — pure RRF fusion, no IO
# ---------------------------------------------------------------------------

# The module-level constant matches the source: _RRF_K = 15
_RRF_K = 15


class TestRrfMerge:
    """_rrf_merge(vec_rows, fts_rows, k) returns a merged ranked list."""

    def test_vec_and_fts_rows_both_appear_in_result(self):
        """Two different skills, one from vec and one from FTS, both surface."""
        from abilities.find_skills import _rrf_merge

        vec_rows = [(1, "Research Framework", 0.12)]
        fts_rows = [(2, "Project Planning", -3.5)]

        result = _rrf_merge(vec_rows, fts_rows, k=5)

        skill_ids = {r["skill_id"] for r in result}
        assert skill_ids == {1, 2}, (
            f"Both skills must appear in merged result, got ids: {skill_ids}"
        )

    def test_skill_in_both_retrievers_appears_exactly_once(self):
        """A skill ranked by both vec and FTS must appear exactly once in output."""
        from abilities.find_skills import _rrf_merge

        vec_rows = [(1, "Research Framework", 0.10)]
        fts_rows = [(1, "Research Framework", -4.2)]

        result = _rrf_merge(vec_rows, fts_rows, k=5)

        skill_ids = [r["skill_id"] for r in result]
        assert skill_ids.count(1) == 1, (
            f"Skill 1 duplicated in output: {skill_ids}"
        )

    def test_skill_in_both_retrievers_has_higher_score_than_single_retriever(self):
        """A skill ranked in both vec and FTS accumulates a higher RRF score
        than a skill ranked only in vec at the same position.

        Skill A: vec rank 1 only            → score = 1/(RRF_K+1)
        Skill B: vec rank 2 + FTS rank 1    → score = 1/(RRF_K+2) + 1/(RRF_K+1)
        Skill B must rank first.
        """
        from abilities.find_skills import _rrf_merge

        vec_rows = [(1, "Skill A", 0.05), (2, "Skill B", 0.15)]
        fts_rows = [(2, "Skill B", -5.0)]

        result = _rrf_merge(vec_rows, fts_rows, k=5)

        assert result[0]["skill_id"] == 2, (
            f"Skill B (dual-retriever) must rank above Skill A (vec-only). "
            f"Got order: {[r['skill_id'] for r in result]}"
        )

    def test_cap_at_k_limits_output_size(self):
        """Result is capped at k even when more rows are available."""
        from abilities.find_skills import _rrf_merge

        vec_rows = [(i, f"Skill {i}", float(i) * 0.01) for i in range(10)]
        fts_rows = []

        result = _rrf_merge(vec_rows, fts_rows, k=3)

        assert len(result) == 3, (
            f"Result must be capped at k=3, got {len(result)} items"
        )

    def test_empty_inputs_return_empty_list(self):
        """Both empty → empty result, no exception."""
        from abilities.find_skills import _rrf_merge

        result = _rrf_merge([], [], k=5)
        assert result == []

    def test_fts_only_path_returns_results(self):
        """vec_rows empty, fts_rows populated → FTS skills surface."""
        from abilities.find_skills import _rrf_merge

        fts_rows = [(7, "Meal Planning", -2.1), (8, "Fitness Routine", -3.8)]
        result = _rrf_merge([], fts_rows, k=5)

        skill_ids = {r["skill_id"] for r in result}
        assert skill_ids == {7, 8}

    def test_result_items_contain_required_keys(self):
        """Each item in the result has skill_id, title, and score keys."""
        from abilities.find_skills import _rrf_merge

        vec_rows = [(1, "Research Framework", 0.10)]
        fts_rows = [(2, "Project Planning", -4.0)]

        result = _rrf_merge(vec_rows, fts_rows, k=5)

        for item in result:
            assert "skill_id" in item, f"Missing 'skill_id' in: {item}"
            assert "title" in item, f"Missing 'title' in: {item}"
            assert "score" in item, f"Missing 'score' in: {item}"

    def test_best_distance_wins_when_skill_appears_multiple_times_in_vec(self):
        """If a skill appears twice in vec_rows, only the lowest distance counts."""
        from abilities.find_skills import _rrf_merge

        # Skill 1 with two distances — only the smaller (0.05) should be used for ranking.
        vec_rows = [(1, "Skill A", 0.50), (1, "Skill A", 0.05)]
        fts_rows = []

        result = _rrf_merge(vec_rows, fts_rows, k=5)

        skill_ids = [r["skill_id"] for r in result]
        assert skill_ids.count(1) == 1, (
            f"Dedup by best vec distance must collapse to one row, got: {skill_ids}"
        )
        # Rank should be 1 (best distance = 0.05, sorts first)
        assert result[0]["skill_id"] == 1


# ---------------------------------------------------------------------------
# 2. _parse_associations — pure JSON parser, no IO
# ---------------------------------------------------------------------------


class TestParseAssociations:
    """_parse_associations(text) converts LLM JSON text into a list of dicts."""

    def test_valid_json_array_returns_list_of_dicts(self):
        """Plain JSON array is parsed correctly."""
        from services.skill_association_service import _parse_associations

        payload = json.dumps([
            {"skill_id": 1, "pattern_name": "dawn_meditation", "rule": "Schedule mindfulness tasks early morning."},
            {"skill_id": 3, "pattern_name": "evening_walk", "rule": "Plan physical activities post-work."},
        ])
        result = _parse_associations(payload)

        assert result is not None
        assert len(result) == 2
        assert result[0]["skill_id"] == 1
        assert result[1]["pattern_name"] == "evening_walk"

    def test_fenced_code_block_is_stripped_before_parse(self):
        """LLM sometimes wraps output in ```json ... ``` — should parse cleanly."""
        from services.skill_association_service import _parse_associations

        payload = '```json\n[{"skill_id": 2, "pattern_name": "morning_run", "rule": "Prioritise physical tasks."}]\n```'
        result = _parse_associations(payload)

        assert result is not None
        assert len(result) == 1
        assert result[0]["skill_id"] == 2

    def test_empty_array_returns_empty_list(self):
        """LLM returning [] is valid — means no patterns match any skills."""
        from services.skill_association_service import _parse_associations

        result = _parse_associations("[]")

        assert result is not None
        assert result == []

    def test_non_list_json_returns_none(self):
        """If the LLM returns a dict instead of a list, return None."""
        from services.skill_association_service import _parse_associations

        result = _parse_associations('{"error": "unexpected"}')

        assert result is None

    def test_malformed_json_returns_none(self):
        """Unparseable text must return None without raising."""
        from services.skill_association_service import _parse_associations

        result = _parse_associations("this is not json at all")

        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string input must return None."""
        from services.skill_association_service import _parse_associations

        result = _parse_associations("")

        assert result is None


# ---------------------------------------------------------------------------
# 3. _safe_parse_summary — pure summary extractor, no IO
# ---------------------------------------------------------------------------


class TestSafeParseSummary:
    """_safe_parse_summary(value) extracts 'summary' from pattern JSON."""

    def test_valid_json_with_summary_returns_summary_text(self):
        """Happy path: JSON with 'summary' key returns the summary string."""
        from services.skill_association_service import _safe_parse_summary

        value = json.dumps({
            "name": "dawn_meditation",
            "frequency": "daily",
            "summary": "Practises 20 minutes of meditation before breakfast",
            "confidence": 8.0,
        })
        result = _safe_parse_summary(value)

        assert result == "Practises 20 minutes of meditation before breakfast"

    def test_json_missing_summary_key_falls_back_to_truncated_raw(self):
        """JSON without 'summary' key falls back to the raw value (up to 200 chars)."""
        from services.skill_association_service import _safe_parse_summary

        value = json.dumps({"name": "evening_walk", "frequency": "weekdays"})
        result = _safe_parse_summary(value)

        # Must not raise; result is a string
        assert isinstance(result, str)
        assert len(result) <= 200

    def test_invalid_json_returns_raw_string_up_to_200_chars(self):
        """Non-JSON string falls back to the raw value, truncated at 200 chars."""
        from services.skill_association_service import _safe_parse_summary

        long_value = "x" * 300
        result = _safe_parse_summary(long_value)

        assert isinstance(result, str)
        assert len(result) == 200

    def test_none_input_returns_string_without_crash(self):
        """None input must not raise; returns a string."""
        from services.skill_association_service import _safe_parse_summary

        result = _safe_parse_summary(None)

        assert isinstance(result, str)

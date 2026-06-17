

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. rrf_merge — pure RRF fusion, no IO
# ---------------------------------------------------------------------------


class TestRrfMerge:
    """rrf_merge(vec_rows, fts_rows, k) returns a merged ranked list."""

    def test_vec_and_fts_rows_both_appear_in_result(self):
        from abilities._search import SearchableAbility

        vec_rows = [(1, "Research Framework", 0.12)]
        fts_rows = [(2, "Project Planning", -3.5)]

        result = SearchableAbility.rrf_merge(vec_rows, fts_rows, k=5)

        keys = {r["key"] for r in result}
        assert keys == {1, 2}

    def test_item_in_both_retrievers_appears_exactly_once(self):
        from abilities._search import SearchableAbility

        vec_rows = [(1, "Research Framework", 0.10)]
        fts_rows = [(1, "Research Framework", -4.2)]

        result = SearchableAbility.rrf_merge(vec_rows, fts_rows, k=5)

        keys = [r["key"] for r in result]
        assert keys.count(1) == 1

    def test_item_in_both_retrievers_has_higher_score_than_single_retriever(self):
        from abilities._search import SearchableAbility

        vec_rows = [(1, "Skill A", 0.05), (2, "Skill B", 0.15)]
        fts_rows = [(2, "Skill B", -5.0)]

        result = SearchableAbility.rrf_merge(vec_rows, fts_rows, k=5)

        assert result[0]["key"] == 2

    def test_cap_at_k_limits_output_size(self):
        from abilities._search import SearchableAbility

        vec_rows = [(i, f"Skill {i}", float(i) * 0.01) for i in range(10)]

        result = SearchableAbility.rrf_merge(vec_rows, [], k=3)

        assert len(result) == 3

    def test_empty_inputs_return_empty_list(self):
        from abilities._search import SearchableAbility

        assert SearchableAbility.rrf_merge([], [], k=5) == []

    def test_fts_only_path_returns_results(self):
        from abilities._search import SearchableAbility

        fts_rows = [(7, "Meal Planning", -2.1), (8, "Fitness Routine", -3.8)]
        result = SearchableAbility.rrf_merge([], fts_rows, k=5)

        assert {r["key"] for r in result} == {7, 8}

    def test_result_items_contain_required_keys(self):
        from abilities._search import SearchableAbility

        result = SearchableAbility.rrf_merge([(1, "A", 0.1)], [(2, "B", -4.0)], k=5)

        for item in result:
            assert "key" in item
            assert "label" in item
            assert "score" in item

    def test_best_distance_wins_when_item_appears_multiple_times_in_vec(self):
        from abilities._search import SearchableAbility

        vec_rows = [(1, "Skill A", 0.50), (1, "Skill A", 0.05)]

        result = SearchableAbility.rrf_merge(vec_rows, [], k=5)

        assert [r["key"] for r in result].count(1) == 1
        assert result[0]["key"] == 1


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

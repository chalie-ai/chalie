"""
Tests for recall_skill — user_traits layer, transcript layer, and helpers.

Note: UserTraitService and get_shared_db_service are imported inside the function
body, so we patch them at their source modules (not in recall_skill).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFormatTraitHit:
    """Unit tests for _format_trait_hit helper.

    source column removed in migration 006 (Stream 1 — memory chunker killed).
    _format_trait_hit now takes 4 args: key, value, category, confidence.
    """

    def test_high_confidence_label(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("name", "Dylan", "core", 0.95)
        assert hit["layer"] == "user_traits"
        assert hit["content"] == "name: Dylan"
        assert hit["confidence"] == 0.95
        assert hit["freshness"] == "well established"
        assert hit["meta"]["confidence_label"] == "well established"
        assert hit["meta"]["category"] == "core"

    def test_medium_confidence_label(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("food_preference", "ramen", "preference", 0.55)
        assert hit["freshness"] == "likely"
        assert hit["meta"]["confidence_label"] == "likely"

    def test_low_confidence_label(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("hobby", "hiking", "general", 0.25)
        assert hit["freshness"] == "uncertain"
        assert hit["meta"]["confidence_label"] == "uncertain"


def _make_traits():
    # source column removed in migration 006 (Stream 1 — memory chunker killed)
    return [
        {"trait_key": "name", "trait_value": "Dylan", "category": "core", "confidence": 0.95},
        {"trait_key": "food_preference", "trait_value": "ramen", "category": "preference", "confidence": 0.6},
        {"trait_key": "hobby", "trait_value": "coding", "category": "general", "confidence": 0.35},
        {"trait_key": "low_conf_thing", "trait_value": "yoga", "category": "general", "confidence": 0.15},
    ]


@pytest.mark.unit
class TestSearchUserTraits:
    """Unit tests for _search_user_traits."""

    def test_broad_query_returns_all_above_threshold(self):
        from services.innate_skills.recall_skill import _search_user_traits
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = _make_traits()
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, status = _search_user_traits("topic", "user profile", 3)

        # 3 traits have confidence >= 0.3 (yoga=0.15 excluded)
        assert len(hits) == 3
        assert "3 matches" in status

    def test_specific_query_keyword_matches_key(self):
        from services.innate_skills.recall_skill import _search_user_traits
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = _make_traits()
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, status = _search_user_traits("topic", "food", 3)

        assert len(hits) == 1
        assert hits[0]["content"] == "food_preference: ramen"

    def test_empty_traits_returns_empty(self):
        from services.innate_skills.recall_skill import _search_user_traits
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = []
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, status = _search_user_traits("topic", "user profile", 3)

        assert hits == []
        assert "0 traits" in status

    def test_broad_query_sorted_by_confidence(self):
        from services.innate_skills.recall_skill import _search_user_traits
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = _make_traits()
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, _ = _search_user_traits("topic", "me", 3)

        confidences = [h["confidence"] for h in hits]
        assert confidences == sorted(confidences, reverse=True)

    def test_broad_cap_triggers_more_available_message(self):
        from services.innate_skills.recall_skill import _search_user_traits, BROAD_TRAIT_DISPLAY_CAP
        many_traits = [
            {"trait_key": f"key_{i}", "trait_value": f"val_{i}",
             "category": "general", "confidence": 0.5}
            for i in range(BROAD_TRAIT_DISPLAY_CAP + 5)
        ]
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = many_traits
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, status = _search_user_traits("topic", "user profile", 3)

        assert len(hits) == BROAD_TRAIT_DISPLAY_CAP
        assert "more available" in status

    def test_meta_fields_present_in_hits(self):
        from services.innate_skills.recall_skill import _search_user_traits
        mock_svc = MagicMock()
        mock_svc.get_all_traits.return_value = _make_traits()
        mock_db = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            hits, _ = _search_user_traits("topic", "name", 3)

        assert len(hits) == 1
        assert "meta" in hits[0]
        # source removed from meta in migration 006 (Stream 1 — memory chunker killed)
        assert "confidence_label" in hits[0]["meta"]
        assert "category" in hits[0]["meta"]

    def test_error_returns_empty_tuple(self):
        from services.innate_skills.recall_skill import _search_user_traits
        with patch('services.database_service.get_shared_db_service', side_effect=Exception("db down")):
            hits, status = _search_user_traits("topic", "user profile", 3)
        assert hits == []
        assert "error" in status


@pytest.mark.unit
class TestBroadQuerySet:
    """Ensure all expected broad queries are recognized."""

    def test_broad_queries_recognized(self):
        from services.innate_skills.recall_skill import BROAD_QUERIES
        expected = {"me", "myself", "user profile", "about me", "what do you know", "what do you remember", "profile"}
        for q in expected:
            assert q in BROAD_QUERIES, f"Expected '{q}' in BROAD_QUERIES"

    def test_specific_query_not_broad(self):
        from services.innate_skills.recall_skill import BROAD_QUERIES
        assert "food preferences" not in BROAD_QUERIES
        assert "my name" not in BROAD_QUERIES


@pytest.mark.unit
class TestAllLayersConstant:
    """Ensure user_traits is in ALL_LAYERS."""

    def test_user_traits_in_all_layers(self):
        from services.innate_skills.recall_skill import ALL_LAYERS
        assert "user_traits" in ALL_LAYERS

    def test_all_layers_order(self):
        from services.innate_skills.recall_skill import ALL_LAYERS
        # user_traits should be last — doesn't pollute non-self-knowledge queries
        assert ALL_LAYERS[-1] == "user_traits"

    def test_transcript_not_in_default_layers(self):
        from services.innate_skills.recall_skill import ALL_LAYERS
        assert "transcript" not in ALL_LAYERS


@pytest.mark.unit
class TestSearchTranscript:
    """Tests for _search_transcript — opt-in transcript layer."""

    def test_default_uses_current_topic(self):
        from services.innate_skills.recall_skill import _search_transcript

        mock_results = [
            {'id': 1, 'role': 'user', 'content': 'Hello world',
             'tool_name': None, 'created_at': '2026-03-20', 'similarity': 0.9, 'topic': 'greetings'},
        ]
        with patch('services.transcript_service.search', return_value=mock_results) as mock_search:
            hits, status = _search_transcript('greetings', 'hello', 3, {})

        mock_search.assert_called_once_with('greetings', 'hello', limit=3, date_from=None, date_to=None)
        assert len(hits) == 1
        assert hits[0]['layer'] == 'transcript'
        assert '1 matches' in status

    def test_global_passes_none_topic(self):
        from services.innate_skills.recall_skill import _search_transcript

        with patch('services.transcript_service.search', return_value=[]) as mock_search:
            _search_transcript('current', 'query', 3, {'transcript_topic': 'global'})

        mock_search.assert_called_once_with(None, 'query', limit=3, date_from=None, date_to=None)

    def test_specific_topic_override(self):
        from services.innate_skills.recall_skill import _search_transcript

        with patch('services.transcript_service.search', return_value=[]) as mock_search:
            _search_transcript('current', 'query', 3, {'transcript_topic': 'other-topic'})

        mock_search.assert_called_once_with('other-topic', 'query', limit=3, date_from=None, date_to=None)

    def test_date_range_passed_through(self):
        from services.innate_skills.recall_skill import _search_transcript

        params = {
            'date_range': {'from': '2026-03-01', 'to': '2026-03-20'},
        }
        with patch('services.transcript_service.search', return_value=[]) as mock_search:
            _search_transcript('topic', 'query', 3, params)

        mock_search.assert_called_once_with(
            'topic', 'query', limit=3,
            date_from='2026-03-01', date_to='2026-03-20',
        )

    def test_empty_results_includes_scope(self):
        from services.innate_skills.recall_skill import _search_transcript

        with patch('services.transcript_service.search', return_value=[]):
            hits, status = _search_transcript('topic', 'query', 3, {'transcript_topic': 'global'})

        assert hits == []
        assert 'global' in status

    def test_content_truncated_at_300(self):
        from services.innate_skills.recall_skill import _search_transcript

        long_content = 'x' * 500
        mock_results = [
            {'id': 1, 'role': 'user', 'content': long_content,
             'tool_name': None, 'created_at': '2026-03-20', 'similarity': 0.8, 'topic': 't'},
        ]
        with patch('services.transcript_service.search', return_value=mock_results):
            hits, _ = _search_transcript('t', 'query', 3, {})

        # [user] prefix + 300 chars + ...
        assert hits[0]['content'].endswith('...')
        assert len(hits[0]['content']) < 320

    def test_error_returns_empty(self):
        from services.innate_skills.recall_skill import _search_transcript

        with patch('services.transcript_service.search', side_effect=Exception('db down')):
            hits, status = _search_transcript('topic', 'query', 3, {})

        assert hits == []
        assert 'error' in status


@pytest.mark.unit
class TestHandleRecallTranscriptIntegration:
    """Test that include_transcript flag works in handle_recall."""

    def test_include_transcript_false_by_default(self):
        from services.innate_skills.recall_skill import handle_recall

        with patch('services.innate_skills.recall_skill._search_working_memory', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_episodes', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_concepts', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_user_traits', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_transcript') as mock_transcript, \
             patch('services.innate_skills.recall_skill._store_fok_signal'):

            handle_recall('topic', {'query': 'test'})

        mock_transcript.assert_not_called()

    def test_include_transcript_true_calls_search(self):
        from services.innate_skills.recall_skill import handle_recall

        with patch('services.innate_skills.recall_skill._search_working_memory', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_episodes', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_concepts', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_user_traits', return_value=([], 'empty')), \
             patch('services.innate_skills.recall_skill._search_transcript', return_value=([], 'empty')) as mock_transcript, \
             patch('services.innate_skills.recall_skill._store_fok_signal'):

            handle_recall('topic', {'query': 'test', 'include_transcript': True})

        mock_transcript.assert_called_once()

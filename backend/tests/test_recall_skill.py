"""
Tests for recall_skill — focusing on the user_traits layer added in plan-06.

Existing layers (working_memory, gists, facts, episodes, concepts) are tested
indirectly through integration; these tests cover the new user_traits layer and
the _format_trait_hit helper.

Note: UserTraitService and get_shared_db_service are imported inside the function
body, so we patch them at their source modules (not in recall_skill).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFormatTraitHit:
    """Unit tests for _format_trait_hit helper."""

    def test_high_confidence_explicit_source(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("name", "Dylan", "core", 0.95, "explicit")
        assert hit["layer"] == "user_traits"
        assert hit["content"] == "name: Dylan"
        assert hit["confidence"] == 0.95
        assert hit["freshness"] == "well established"
        assert hit["meta"]["source"] == "explicit"
        assert hit["meta"]["confidence_label"] == "well established"
        assert hit["meta"]["category"] == "core"

    def test_medium_confidence_inferred(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("food_preference", "ramen", "preference", 0.55, "inferred")
        assert hit["freshness"] == "likely"
        assert hit["meta"]["source"] == "inferred"
        assert hit["meta"]["confidence_label"] == "likely"

    def test_low_confidence_uncertain(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("hobby", "hiking", "general", 0.25, "inferred")
        assert hit["freshness"] == "uncertain"
        assert hit["meta"]["confidence_label"] == "uncertain"

    def test_default_source_is_inferred(self):
        from services.innate_skills.recall_skill import _format_trait_hit
        hit = _format_trait_hit("key", "val", "general", 0.5)
        assert hit["meta"]["source"] == "inferred"


def _make_traits():
    return [
        {"trait_key": "name", "trait_value": "Dylan", "category": "core", "confidence": 0.95, "source": "explicit"},
        {"trait_key": "food_preference", "trait_value": "ramen", "category": "preference", "confidence": 0.6, "source": "inferred"},
        {"trait_key": "hobby", "trait_value": "coding", "category": "general", "confidence": 0.35, "source": "inferred"},
        {"trait_key": "low_conf_thing", "trait_value": "yoga", "category": "general", "confidence": 0.15, "source": "inferred"},
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
             "category": "general", "confidence": 0.5, "source": "inferred"}
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
        assert "source" in hits[0]["meta"]
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

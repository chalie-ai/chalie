"""Tests for news_sources — static RSS registry."""

import pytest
from services.news_sources import (
    SOURCES, CATEGORIES, Source,
    get_source_by_id, get_sources_by_category, search_sources,
)


@pytest.mark.unit
class TestNewsSources:

    def test_total_source_count(self):
        assert len(SOURCES) == 56

    def test_category_count(self):
        assert len(CATEGORIES) == 8

    def test_all_sources_have_valid_fields(self):
        for s in SOURCES:
            assert isinstance(s, Source)
            assert len(s.id) >= 2
            assert len(s.name) >= 2
            assert s.category in CATEGORIES
            assert s.feed_url.startswith("https://")
            assert len(s.country) == 2

    def test_no_duplicate_ids(self):
        ids = [s.id for s in SOURCES]
        assert len(ids) == len(set(ids))

    def test_category_distribution(self):
        expected = {
            "international": 10, "us": 10, "uk": 6, "tech": 8,
            "business": 6, "science": 5, "sports": 6, "entertainment": 5,
        }
        for cat, count in expected.items():
            assert len(get_sources_by_category(cat)) == count, f"{cat} expected {count}"

    def test_all_source_categories_in_categories(self):
        for s in SOURCES:
            assert s.category in CATEGORIES

    def test_get_source_by_id_found(self):
        src = get_source_by_id("bbc_world")
        assert src is not None
        assert src.name == "BBC World News"
        assert src.category == "international"

    def test_get_source_by_id_not_found(self):
        assert get_source_by_id("nonexistent_source") is None

    def test_get_sources_by_category_unknown(self):
        assert get_sources_by_category("fictional") == []

    def test_search_sources_empty_query(self):
        result = search_sources("")
        assert len(result) == 56

    def test_search_sources_partial_match(self):
        result = search_sources("BBC")
        assert len(result) >= 3  # bbc_world, bbc_uk, bbc_sport
        for s in result:
            assert "BBC" in s.name

    def test_search_sources_case_insensitive(self):
        result = search_sources("bbc")
        assert len(result) >= 3

    def test_search_sources_no_match(self):
        result = search_sources("zzzznonexistent")
        assert result == []

    def test_search_sources_single_match(self):
        result = search_sources("TechCrunch")
        assert len(result) == 1
        assert result[0].id == "techcrunch"

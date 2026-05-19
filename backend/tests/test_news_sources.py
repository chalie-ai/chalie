"""Tests for news_sources — static RSS registry."""

import pytest
from services.news_sources import (
    SOURCES, Source,
    get_source_by_id,
)


@pytest.mark.unit
class TestNewsSources:

    def test_total_source_count(self):
        assert len(SOURCES) == 56

    def test_all_sources_have_valid_fields(self):
        for s in SOURCES:
            assert isinstance(s, Source)
            assert len(s.id) >= 2
            assert len(s.name) >= 2
            assert s.category in {"international", "us", "uk", "tech", "business", "science", "sports", "entertainment"}
            assert s.feed_url.startswith("https://")
            assert len(s.country) == 2

    def test_no_duplicate_ids(self):
        ids = [s.id for s in SOURCES]
        assert len(ids) == len(set(ids))

    def test_get_source_by_id_found(self):
        src = get_source_by_id("bbc_world")
        assert src is not None
        assert src.name == "BBC World News"
        assert src.category == "international"

    def test_get_source_by_id_not_found(self):
        assert get_source_by_id("nonexistent_source") is None



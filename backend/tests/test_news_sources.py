import pytest
from services.news_sources import (
    SOURCES,
    get_source_by_id,
)


@pytest.mark.unit
class TestNewsSources:

    def test_no_duplicate_ids(self):
        ids = [s.id for s in SOURCES]
        assert len(ids) == len(set(ids))

    def test_get_source_by_id_found(self):
        src = get_source_by_id("bbc_world")
        assert src is not None
        assert src.name == "BBC World News"
        assert src.category == "international"


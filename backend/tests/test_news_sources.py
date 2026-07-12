import pytest

from services.news_sources import SOURCES


@pytest.mark.unit
class TestNewsSources:

    def test_no_duplicate_ids(self) -> None:
        ids = [s.id for s in SOURCES]
        assert len(ids) == len(set(ids))


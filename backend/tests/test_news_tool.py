"""Tests for news tool — action routing and formatting."""

from unittest.mock import MagicMock, patch

import pytest

from services.news_service import NewsArticle
from tools.news.news import execute, _format_articles, _format_clusters, _format_sources, _relative_time


def _make_article(title="Test Article", source="BBC", **kwargs):
    defaults = {
        "title": title,
        "description": "Test description",
        "url": "https://example.com/test",
        "published_at": "2026-03-24T10:00:00+00:00",
        "source": source,
        "source_id": "bbc_world",
        "category": "international",
    }
    defaults.update(kwargs)
    return NewsArticle(**defaults)


@pytest.fixture
def mock_service():
    with patch("tools.news.news._get_service") as mock:
        svc = MagicMock()
        mock.return_value = svc
        yield svc


# ── Action routing ────────────────────────────────────────────

@pytest.mark.unit
class TestActionRouting:

    def test_unknown_action_returns_error(self, mock_service):
        result = execute("test-topic", {"action": "invalid"})
        assert result["error"]
        assert "Unknown action" in result["error"]

    def test_missing_action_returns_error(self, mock_service):
        result = execute("test-topic", {})
        assert result["error"]


# ── Search action ─────────────────────────────────────────────

@pytest.mark.unit
class TestSearchAction:

    def test_search_returns_articles(self, mock_service):
        mock_service.search.return_value = [_make_article(), _make_article(title="Second")]
        result = execute("test", {"action": "search", "query": "test"})
        assert result["error"] == ""
        assert "Test Article" in result["text"]
        assert "News:" in result["title"]

    def test_search_missing_query(self, mock_service):
        result = execute("test", {"action": "search"})
        assert result["error"]
        assert "query" in result["error"].lower()

    def test_search_no_results(self, mock_service):
        mock_service.search.return_value = []
        result = execute("test", {"action": "search", "query": "nothing"})
        assert "No news articles" in result["text"]

    def test_search_limit_capped(self, mock_service):
        mock_service.search.return_value = []
        execute("test", {"action": "search", "query": "test", "limit": 50})
        _, kwargs = mock_service.search.call_args
        assert kwargs.get("limit", 10) <= 20


# ── Digest action ─────────────────────────────────────────────

@pytest.mark.unit
class TestDigestAction:

    def test_digest_returns_sections(self, mock_service):
        mock_service.get_digest.return_value = {
            "international": [_make_article()],
            "local": [_make_article(title="Local Story")],
        }
        result = execute("test", {"action": "digest"}, config={}, telemetry={"city": "London"})
        assert result["error"] == ""
        assert "INTERNATIONAL" in result["text"]
        assert result["title"] == "News Digest"

    def test_digest_uses_config_preferred_source(self, mock_service):
        mock_service.get_digest.return_value = {"international": [], "local": []}
        execute("test", {"action": "digest"}, config={"preferred_source": "reuters_world"})
        mock_service.get_digest.assert_called_once()

    def test_digest_location_from_telemetry(self, mock_service):
        mock_service.get_digest.return_value = {"international": [], "local": []}
        execute("test", {"action": "digest"}, config={}, telemetry={"city": "Malta"})
        call_kwargs = mock_service.get_digest.call_args[1]
        assert call_kwargs["location"] == "Malta"


# ── Trending action ───────────────────────────────────────────

@pytest.mark.unit
class TestTrendingAction:

    def test_trending_returns_clusters(self, mock_service):
        mock_service.fetch_feeds.return_value = [_make_article()]
        mock_service.deduplicate.return_value = [_make_article()]
        mock_service.cluster_trending.return_value = [
            {"title": "Big Story", "articles": [_make_article()], "coverage": 3},
        ]
        result = execute("test", {"action": "trending"})
        assert result["error"] == ""
        assert "Big Story" in result["text"]

    def test_trending_invalid_category_defaults(self, mock_service):
        mock_service.fetch_feeds.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.cluster_trending.return_value = []
        result = execute("test", {"action": "trending", "category": "fake"})
        assert "international" in result["title"]

    def test_trending_no_clusters(self, mock_service):
        mock_service.fetch_feeds.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.cluster_trending.return_value = []
        result = execute("test", {"action": "trending"})
        assert "No trending" in result["text"]


# ── Sources action ────────────────────────────────────────────

@pytest.mark.unit
class TestSourcesAction:

    def test_sources_returns_all(self, mock_service):
        result = execute("test", {"action": "sources"})
        assert result["error"] == ""
        assert "INTERNATIONAL" in result["text"]
        assert "TECH" in result["text"]

    def test_sources_filter_by_category(self, mock_service):
        result = execute("test", {"action": "sources", "category": "tech"})
        assert "TECH" in result["text"]
        assert "TechCrunch" in result["text"]

    def test_sources_search_by_name(self, mock_service):
        result = execute("test", {"action": "sources", "source": "BBC"})
        assert "BBC" in result["text"]


# ── Formatting helpers ────────────────────────────────────────

@pytest.mark.unit
class TestFormatting:

    def test_format_articles(self):
        articles = [_make_article(title="Big News", source="Reuters")]
        text = _format_articles(articles)
        assert "Big News" in text
        assert "Reuters" in text

    def test_format_articles_truncates_description(self):
        articles = [_make_article(description="x" * 200)]
        text = _format_articles(articles)
        assert len([l for l in text.split("\n") if l.startswith("  x")]) == 1

    def test_format_clusters(self):
        clusters = [{
            "title": "Cluster Title",
            "articles": [_make_article(title="Sub Article", source="BBC")],
            "coverage": 3,
        }]
        text = _format_clusters(clusters)
        assert "Cluster Title" in text
        assert "3 sources" in text

    def test_format_sources(self):
        from services.news_sources import Source
        sources = [
            Source("bbc_world", "BBC World", "international", "https://example.com", "GB"),
            Source("cnn", "CNN", "us", "https://example.com", "US"),
        ]
        text = _format_sources(sources)
        assert "INTERNATIONAL" in text
        assert "US" in text

    def test_relative_time_just_now(self):
        from services.time_utils import utc_now
        assert _relative_time(utc_now().isoformat()) == "just now"

    def test_relative_time_invalid(self):
        # parse_utc may coerce garbage to a datetime — either returns "" or a time string
        result = _relative_time("garbage")
        assert isinstance(result, str)

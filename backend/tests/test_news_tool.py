"""Tests for news tool — 2-param interface (query + optional category)."""

from unittest.mock import MagicMock, patch

import pytest

from services.news_service import NewsArticle
from tools.news.news import execute, _format_articles, _resolve_country_code
from services.time_formatter_service import TimeFormatterService


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


# ── Core routing ──────────────────────────────────────────────

@pytest.mark.unit
class TestRouting:

    def test_missing_query_returns_error(self, mock_service):
        result = execute("test-topic", {})
        assert result["error"]
        assert "query" in result["error"].lower()

    def test_empty_query_returns_error(self, mock_service):
        result = execute("test-topic", {"query": "   "})
        assert result["error"]

    def test_query_only_calls_google_news(self, mock_service):
        mock_service.fetch_google_news.return_value = [_make_article()]
        result = execute("test", {"query": "AI news"})
        mock_service.fetch_google_news.assert_called_once()
        mock_service.fetch_feeds.assert_not_called()
        assert not result.get("error")

    def test_query_with_category_calls_feeds_and_google(self, mock_service):
        mock_service.fetch_feeds.return_value = [_make_article()]
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = [_make_article()]
        mock_service.rank_by_relevance.return_value = [_make_article()]
        result = execute("test", {"query": "AI news", "category": "tech"})
        mock_service.fetch_feeds.assert_called_once()
        mock_service.fetch_google_news.assert_called_once()
        assert not result.get("error")


# ── No category path (Google News) ────────────────────────────

@pytest.mark.unit
class TestQueryOnlyPath:

    def test_returns_articles(self, mock_service):
        mock_service.fetch_google_news.return_value = [_make_article(), _make_article(title="Second")]
        result = execute("test", {"query": "climate change"})
        assert not result.get("error")
        assert "Test Article" in result["text"]
        assert "News:" in result["title"]

    def test_no_results_returns_message(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        result = execute("test", {"query": "nothing found"})
        assert "No news found" in result["text"]

    def test_geo_from_telemetry_country(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "local news"}, telemetry={"country": "United Kingdom"})
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "GB"

    def test_no_telemetry_defaults_us(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "news"})
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "US"

    def test_unknown_country_defaults_us(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "news"}, telemetry={"country": "Narnia"})
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "US"

    def test_capped_at_ten_articles(self, mock_service):
        mock_service.fetch_google_news.return_value = [
            _make_article(title=f"Article {i}") for i in range(20)
        ]
        result = execute("test", {"query": "news"})
        bullet_count = result["text"].count("\u2022")
        assert bullet_count <= 10


# ── Category path (curated RSS + Google) ──────────────────────

@pytest.mark.unit
class TestCategoryPath:

    def test_category_tech_fetches_feeds(self, mock_service):
        mock_service.fetch_feeds.return_value = [_make_article()]
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = [_make_article()]
        mock_service.rank_by_relevance.return_value = [_make_article()]
        execute("test", {"query": "AI", "category": "tech"})
        source_ids = mock_service.fetch_feeds.call_args[0][0]
        assert len(source_ids) > 0

    def test_category_no_results(self, mock_service):
        mock_service.fetch_feeds.return_value = []
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.rank_by_relevance.return_value = []
        result = execute("test", {"query": "sports", "category": "sports"})
        assert "No news found" in result["text"]

    def test_category_deduplicates_and_ranks(self, mock_service):
        articles = [_make_article()]
        mock_service.fetch_feeds.return_value = articles
        mock_service.fetch_google_news.return_value = articles
        mock_service.deduplicate.return_value = articles
        mock_service.rank_by_relevance.return_value = articles
        execute("test", {"query": "business", "category": "business"})
        mock_service.deduplicate.assert_called_once()
        mock_service.rank_by_relevance.assert_called_once()


# ── Country code resolver ─────────────────────────────────────

@pytest.mark.unit
class TestResolveCountryCode:

    def test_known_country(self):
        assert _resolve_country_code("Malta") == "MT"

    def test_known_country_case_insensitive(self):
        assert _resolve_country_code("UNITED KINGDOM") == "GB"

    def test_known_country_with_whitespace(self):
        assert _resolve_country_code("  germany  ") == "DE"

    def test_unknown_country_defaults_us(self):
        assert _resolve_country_code("Narnia") == "US"

    def test_none_defaults_us(self):
        assert _resolve_country_code(None) == "US"

    def test_empty_string_defaults_us(self):
        assert _resolve_country_code("") == "US"

    def test_united_states(self):
        assert _resolve_country_code("United States") == "US"

    def test_japan(self):
        assert _resolve_country_code("japan") == "JP"


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
        assert len([line for line in text.split("\n") if line.startswith("  x")]) == 1

    def test_relative_time_just_now(self):
        from services.time_utils import utc_now
        result = TimeFormatterService.ago(utc_now().isoformat())
        assert result == "0s ago" or result.endswith("ago")

    def test_relative_time_invalid(self):
        result = TimeFormatterService.ago("garbage")
        assert isinstance(result, str)

    def test_relative_time_minutes_ago(self):
        from datetime import timedelta
        from services.time_utils import utc_now
        past = (utc_now() - timedelta(minutes=30)).isoformat()
        assert TimeFormatterService.ago(past) == "30m ago"

    def test_relative_time_hours_ago(self):
        from datetime import timedelta
        from services.time_utils import utc_now
        past = (utc_now() - timedelta(hours=5)).isoformat()
        assert TimeFormatterService.ago(past) == "5h 0m ago"

    def test_relative_time_days_ago(self):
        from datetime import timedelta
        from services.time_utils import utc_now
        past = (utc_now() - timedelta(days=3)).isoformat()
        assert TimeFormatterService.ago(past) == "3d 0h ago"

    def test_format_articles_empty_description(self):
        articles = [_make_article(description="")]
        text = _format_articles(articles)
        assert "Test Article" in text
        # No description line — only bullet and source·time lines
        assert text.count("  ") <= 2  # source line + possible blank

    def test_format_articles_multiple_items(self):
        articles = [_make_article(title=f"Article {i}") for i in range(5)]
        text = _format_articles(articles)
        assert text.count("\u2022") == 5


# ── Unknown / invalid category ────────────────────────────────

@pytest.mark.unit
class TestUnknownCategory:

    def test_unknown_category_produces_no_sources(self, mock_service):
        # get_sources_by_category("bogus") returns [] → fetch_feeds([]) → []
        mock_service.fetch_feeds.return_value = []
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.rank_by_relevance.return_value = []
        result = execute("test", {"query": "AI", "category": "bogus"})
        assert "No news found" in result["text"]

    def test_unknown_category_still_calls_google_news(self, mock_service):
        # Even with no RSS sources, the code calls fetch_google_news for the query
        mock_service.fetch_feeds.return_value = []
        mock_service.fetch_google_news.return_value = [_make_article()]
        mock_service.deduplicate.return_value = [_make_article()]
        mock_service.rank_by_relevance.return_value = [_make_article()]
        result = execute("test", {"query": "AI", "category": "bogus"})
        mock_service.fetch_google_news.assert_called_once()
        assert not result.get("error")


# ── Telemetry edge cases ──────────────────────────────────────

@pytest.mark.unit
class TestTelemetryEdgeCases:

    def test_telemetry_none_defaults_us(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "news"}, telemetry=None)
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "US"

    def test_telemetry_missing_country_key_defaults_us(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "news"}, telemetry={"city": "London"})
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "US"

    def test_telemetry_none_country_value_defaults_us(self, mock_service):
        mock_service.fetch_google_news.return_value = []
        execute("test", {"query": "news"}, telemetry={"country": None})
        _, kwargs = mock_service.fetch_google_news.call_args
        assert kwargs.get("country_code") == "US"


# ── Service exception handling ────────────────────────────────

@pytest.mark.unit
class TestErrorHandling:

    def test_service_exception_returns_error_dict(self, mock_service):
        mock_service.fetch_google_news.side_effect = RuntimeError("network down")
        result = execute("test", {"query": "AI news"})
        assert result.get("error")
        assert "network down" in result["error"]
        assert result["text"] == ""

    def test_service_exception_in_category_path_returns_error_dict(self, mock_service):
        mock_service.fetch_feeds.side_effect = RuntimeError("db error")
        result = execute("test", {"query": "sports", "category": "sports"})
        assert result.get("error")


# ── Category path country code behavior ──────────────────────

@pytest.mark.unit
class TestCategoryPathCountryCode:

    def test_category_path_passes_country_code_from_telemetry(self, mock_service):
        mock_service.fetch_feeds.return_value = []
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.rank_by_relevance.return_value = []
        execute("test", {"query": "AI", "category": "tech"},
                telemetry={"country": "Germany"})
        _, call_kwargs = mock_service.fetch_google_news.call_args
        assert call_kwargs.get("country_code") == "DE"

    def test_category_path_defaults_country_code_us(self, mock_service):
        mock_service.fetch_feeds.return_value = []
        mock_service.fetch_google_news.return_value = []
        mock_service.deduplicate.return_value = []
        mock_service.rank_by_relevance.return_value = []
        execute("test", {"query": "AI", "category": "tech"})
        _, call_kwargs = mock_service.fetch_google_news.call_args
        assert call_kwargs.get("country_code") == "US"


# ── _resolve_country_code additional edge cases ───────────────

@pytest.mark.unit
class TestResolveCountryCodeEdgeCases:

    def test_whitespace_only_defaults_us(self):
        assert _resolve_country_code("   ") == "US"

    def test_all_categories_in_map_resolve(self):
        # Spot-check a handful of entries in the map
        assert _resolve_country_code("France") == "FR"
        assert _resolve_country_code("Canada") == "CA"
        assert _resolve_country_code("Australia") == "AU"
        assert _resolve_country_code("South Korea") == "KR"
        assert _resolve_country_code("United Arab Emirates") == "AE"

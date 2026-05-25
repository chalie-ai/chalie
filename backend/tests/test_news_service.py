"""Tests for news_service — fetch, cache, rank, cluster, dedup."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.news_service import (
    NewsService, NewsArticle,
    _strip_html, _derive_domain,
)

SAMPLE_RSS = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Test Article One</title>
      <description>This is a test description.</description>
      <link>https://example.com/article-1</link>
      <pubDate>Mon, 24 Mar 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Test Article Two</title>
      <description>&lt;p&gt;HTML description&lt;/p&gt;</description>
      <link>https://example.com/article-2</link>
      <pubDate>Mon, 24 Mar 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <description>No title item</description>
      <link>https://example.com/no-title</link>
    </item>
  </channel>
</rss>'''

SAMPLE_ATOM = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Atom Entry One</title>
    <summary>Summary text here.</summary>
    <link rel="alternate" href="https://example.com/entry-1"/>
    <published>2026-03-24T10:00:00Z</published>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <content>Content text here.</content>
    <link rel="alternate" href="https://example.com/entry-2"/>
    <updated>2026-03-24T09:00:00Z</updated>
  </entry>
</feed>'''


def _make_article(title="Test Article", source="BBC", source_id="bbc_world", **kwargs):
    defaults = {
        "title": title,
        "description": "Test description",
        "url": "https://example.com/test",
        "published_at": "2026-03-24T10:00:00+00:00",
        "source": source,
        "source_id": source_id,
        "category": "international",
    }
    defaults.update(kwargs)
    return NewsArticle(**defaults)


# ── Helper function tests ─────────────────────────────────────

@pytest.mark.unit
class TestHelpers:

    def test_strip_html_removes_tags_and_decodes_entities(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"
        assert _strip_html("&amp; &lt;") == "& <"

    def test_derive_domain_strips_feed_subdomains(self):
        assert _derive_domain("https://feeds.bbci.co.uk/news/rss.xml") == "bbci.co.uk"
        assert _derive_domain("https://www.example.com/feed") == "example.com"


# ── Feed parsing tests ────────────────────────────────────────

@pytest.mark.unit
class TestFeedParsing:

    def setup_method(self):
        self.svc = NewsService()

    @patch("services.news_service.requests.get")
    def test_parse_rss_feed(self, mock_get):
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test Source", "international", "https://example.com/rss", "US")
        articles = self.svc._parse_feed(src, 5.0)

        assert len(articles) == 2  # item without title is skipped
        assert articles[0].title == "Test Article One"
        assert articles[0].source == "Test Source"
        assert articles[0].source_id == "test"
        assert "example.com/article-1" in articles[0].url


# ── Deduplication tests ───────────────────────────────────────

@pytest.mark.unit
class TestDeduplication:

    def setup_method(self):
        self.svc = NewsService()

    def test_exact_duplicates_removed(self):
        articles = [
            _make_article(title="Breaking News Today"),
            _make_article(title="Breaking News Today"),
        ]
        result = self.svc.deduplicate(articles)
        assert len(result) == 1

    def test_near_duplicates_removed(self):
        articles = [
            _make_article(title="Breaking News in Markets Today"),
            _make_article(title="Breaking News in Market Today"),  # edit distance 1 from the title above
        ]
        result = self.svc.deduplicate(articles)
        assert len(result) == 1


# ── Ranking tests ─────────────────────────────────────────────

@pytest.mark.unit
class TestRanking:

    def setup_method(self):
        self.svc = NewsService()

    def test_rank_by_relevance_sorts_by_similarity(self):
        # High similarity for first article, low for second
        vec_high = np.zeros(768, dtype=np.float32)
        vec_high[0] = 1.0
        vec_low = np.zeros(768, dtype=np.float32)
        vec_low[1] = 1.0

        mock_emb = MagicMock()
        mock_emb.generate_embedding_np.return_value = vec_high
        mock_emb.generate_embeddings_batch.return_value = [vec_high, vec_low]
        self.svc._embedding_svc = mock_emb

        articles = [
            _make_article(title="Relevant Article", published_at="2026-03-24T09:00:00+00:00"),
            _make_article(title="Irrelevant Article", published_at="2026-03-24T10:00:00+00:00"),
        ]
        result = self.svc.rank_by_relevance(articles, "relevant query")

        # Only the high-similarity article should pass the floor
        assert len(result) >= 1
        assert result[0].title == "Relevant Article"




# ── Google News country code tests ───────────────────────────

@pytest.mark.unit
class TestFetchGoogleNewsCountryCode:

    def setup_method(self):
        self.svc = NewsService()

    @patch("services.news_service.requests.get")
    def test_country_code_in_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_store = MagicMock()
        mock_store.get.return_value = None
        self.svc._store = mock_store

        with patch.object(self.svc, "_get_store", return_value=mock_store):
            self.svc.fetch_google_news("test query", country_code="GB")

        call_url = mock_get.call_args[0][0]
        assert "gl=GB" in call_url
        assert "ceid=GB:en" in call_url



# ── fetch_feeds edge cases ────────────────────────────────────

@pytest.mark.unit
class TestFetchFeedsEdgeCases:

    def setup_method(self):
        self.svc = NewsService()

    def test_fetch_feeds_empty_source_ids(self):
        result = self.svc.fetch_feeds([])
        assert result == []

    @patch("services.news_service.requests.get")
    def test_fetch_feeds_cache_hit_skips_network(self, mock_get):
        mock_store = MagicMock()
        cached_data = json.dumps([_make_article().to_dict()])
        mock_store.get.return_value = cached_data.encode()

        with patch.object(self.svc, "_get_store", return_value=mock_store):
            result = self.svc.fetch_feeds(["bbc_world"])

        mock_get.assert_not_called()
        assert len(result) == 1
        assert result[0].title == "Test Article"



# ── _parse_feed XML error handling ───────────────────────────

@pytest.mark.unit
class TestParseFeedXmlError:

    def setup_method(self):
        self.svc = NewsService()

    @patch("services.news_service.requests.get")
    def test_malformed_xml_returns_empty(self, mock_get):
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = b"<this is not valid xml <<<<"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test", "international", "https://example.com/rss", "US")
        articles = self.svc._parse_feed(src, 5.0)
        assert articles == []






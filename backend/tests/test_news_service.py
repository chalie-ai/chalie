"""Tests for news_service — fetch, cache, rank, cluster, dedup."""

import json
from unittest.mock import MagicMock, patch

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

# RSS with a media:thumbnail element — exercises the _feedparser_image priority path.
# The `href` key is used by feedparser for enclosures; `url` is used for media_thumbnail.
SAMPLE_RSS_WITH_THUMBNAIL = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <item>
      <title>Image Article</title>
      <link>https://example.com/image-article</link>
      <pubDate>Mon, 24 Mar 2026 10:00:00 GMT</pubDate>
      <media:thumbnail url="https://example.com/thumbnail.jpg"/>
    </item>
  </channel>
</rss>'''


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

    @pytest.mark.parametrize("feed_bytes,expected_titles,expected_urls", [
        (
            SAMPLE_RSS,
            ["Test Article One", "Test Article Two"],
            ["https://example.com/article-1", "https://example.com/article-2"],
        ),
        (
            SAMPLE_ATOM,
            ["Atom Entry One", "Atom Entry Two"],
            ["https://example.com/entry-1", "https://example.com/entry-2"],
        ),
    ])
    @patch("services.news_service.requests.get")
    def test_parse_feed_rss_and_atom(self, mock_get, feed_bytes, expected_titles, expected_urls):
        """_parse_feed handles both RSS and Atom via the unified feedparser path.

        Parametrized over SAMPLE_RSS and SAMPLE_ATOM: asserts article count, titles,
        source metadata, URLs, and that published_at is a valid ISO-8601 UTC string.
        Exercises the feedparser unification for both date formats (RFC 2822 vs ISO 8601).
        """
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = feed_bytes
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test Source", "international", "https://example.com/feed", "US")
        articles = self.svc._parse_feed(src, 5.0)

        assert len(articles) == 2
        for i, article in enumerate(articles):
            assert article.title == expected_titles[i]
            assert article.source == "Test Source"
            assert article.source_id == "test"
            assert article.url == expected_urls[i]
            # published_at must be a valid ISO-8601 UTC string (parse_utc accepts it)
            from services.time_utils import parse_utc
            dt = parse_utc(article.published_at)
            assert dt.tzinfo is not None
            assert "2026-03-24" in article.published_at

    @patch("services.news_service.requests.get")
    def test_media_thumbnail_populates_image_url(self, mock_get):
        """_parse_feed extracts media:thumbnail URL into NewsArticle.image_url.

        Uses SAMPLE_RSS_WITH_THUMBNAIL which carries a media:thumbnail element.
        Exercises the _feedparser_image priority path (thumbnail > media_content > enclosures).
        feedparser maps media:thumbnail to entry.media_thumbnail[0]['url'] (not 'href').
        This is the code path that feeds the image_candidates in the news ability (scenario 054).
        """
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS_WITH_THUMBNAIL
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test Source", "international", "https://example.com/feed", "US")
        articles = self.svc._parse_feed(src, 5.0)

        assert len(articles) == 1
        assert articles[0].image_url == "https://example.com/thumbnail.jpg"


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

    def test_rank_by_relevance_falls_back_to_date_sort_on_embedding_failure(self):
        """rank_by_relevance returns articles newest-first when embedding service fails.

        This tests the deterministic fallback branch — no mock of production code needed
        because there is no embedding service configured on a fresh NewsService instance
        (get_embedding_service raises on import failure in a test environment, triggering
        the except branch that date-sorts). The fallback is the behavior users depend on
        when the embedding model is unavailable.
        """
        articles = [
            _make_article(title="Older Article", published_at="2026-03-24T08:00:00+00:00"),
            _make_article(title="Newer Article", published_at="2026-03-24T12:00:00+00:00"),
            _make_article(title="Middle Article", published_at="2026-03-24T10:00:00+00:00"),
        ]
        # Force the embedding path to fail by providing a service that raises,
        # exercising the except branch in rank_by_relevance that falls back to date sort.
        mock_emb = MagicMock()
        mock_emb.generate_embedding_np.side_effect = RuntimeError("model unavailable")
        self.svc._embedding_svc = mock_emb

        result = self.svc.rank_by_relevance(articles, "query")

        assert result[0].title == "Newer Article"
        assert result[-1].title == "Older Article"

    def test_rank_by_relevance_breaks_score_ties_newest_first(self):
        """Articles with identical cosine scores rank newest-first.

        The two title embeddings are constructed to produce identical dot products
        against the query embedding (both 0.8, above RELEVANCE_FLOOR), so ordering is
        decided purely by published_at — which must be newest-first per the documented
        'score desc, then date desc' contract.
        """
        import numpy as np
        older = _make_article(title="Older Equal", published_at="2026-03-24T08:00:00+00:00")
        newer = _make_article(title="Newer Equal", published_at="2026-03-24T12:00:00+00:00")

        mock_emb = MagicMock()
        mock_emb.generate_embedding_np.return_value = np.array([1.0, 0.0])
        mock_emb.generate_embeddings_batch.return_value = [
            np.array([0.8, 0.6]),   # older → dot product 0.8
            np.array([0.8, -0.6]),  # newer → dot product 0.8 (tie)
        ]
        self.svc._embedding_svc = mock_emb

        result = self.svc.rank_by_relevance([older, newer], "query")

        assert [a.title for a in result] == ["Newer Equal", "Older Equal"]


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






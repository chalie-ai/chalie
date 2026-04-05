"""Tests for news_service — fetch, cache, rank, cluster, dedup."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.news_service import (
    NewsService, NewsArticle,
    _strip_html, _parse_date, _normalize_title, _levenshtein,
    _tokenize_title, _jaccard, _derive_domain, FEED_CACHE_TTL,
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

    def test_strip_html_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_strip_html_decodes_entities(self):
        assert _strip_html("&amp; &lt; &gt; &quot;") == "& < > \""

    def test_strip_html_collapses_whitespace(self):
        assert _strip_html("  hello   world  ") == "hello world"

    def test_parse_date_rfc2822(self):
        result = _parse_date("Mon, 24 Mar 2026 10:00:00 GMT")
        assert "2026-03-24" in result
        assert "10:00:00" in result

    def test_parse_date_iso8601(self):
        result = _parse_date("2026-03-24T10:00:00Z")
        assert "2026-03-24" in result

    def test_parse_date_none_returns_now(self):
        result = _parse_date(None)
        assert "T" in result  # ISO format

    def test_parse_date_garbage_returns_now(self):
        result = _parse_date("not a date")
        assert "T" in result

    def test_normalize_title(self):
        assert _normalize_title("Hello, World! 123") == "hello world 123"

    def test_normalize_title_collapses_spaces(self):
        assert _normalize_title("  hello   world  ") == "hello world"

    def test_levenshtein_identical(self):
        assert _levenshtein("hello", "hello") == 0

    def test_levenshtein_empty(self):
        assert _levenshtein("", "hello") == 5
        assert _levenshtein("hello", "") == 5
        assert _levenshtein("", "") == 0

    def test_levenshtein_one_edit(self):
        assert _levenshtein("hello", "helo") == 1

    def test_levenshtein_different(self):
        assert _levenshtein("cat", "dog") == 3

    def test_tokenize_title_removes_stopwords(self):
        tokens = _tokenize_title("The cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_tokenize_title_lowercase(self):
        tokens = _tokenize_title("HELLO WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 0.01  # 2/4

    def test_jaccard_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_jaccard_one_empty(self):
        assert _jaccard({"a"}, set()) == 0.0

    def test_derive_domain_strips_rss_prefix(self):
        assert _derive_domain("https://feeds.bbci.co.uk/news/rss.xml") == "bbci.co.uk"

    def test_derive_domain_strips_www(self):
        assert _derive_domain("https://www.example.com/feed") == "example.com"

    def test_derive_domain_proxy_returns_none(self):
        assert _derive_domain("https://rsshub.app/some/feed") is None
        assert _derive_domain("https://hnrss.org/frontpage") is None

    def test_derive_domain_simple_url(self):
        assert _derive_domain("https://example.com/feed") == "example.com"


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

    @patch("services.news_service.requests.get")
    def test_parse_atom_feed(self, mock_get):
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_ATOM
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test Source", "tech", "https://example.com/atom", "US")
        articles = self.svc._parse_feed(src, 5.0)

        assert len(articles) == 2
        assert articles[0].title == "Atom Entry One"
        assert "entry-1" in articles[0].url

    @patch("services.news_service.requests.get")
    def test_parse_feed_strips_html_from_description(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from services.news_sources import Source
        src = Source("test", "Test", "international", "https://example.com/rss", "US")
        articles = self.svc._parse_feed(src, 5.0)

        # Second article has HTML description
        assert "<p>" not in articles[1].description
        assert "HTML description" in articles[1].description

    @patch("services.news_service.requests.get")
    def test_parse_feed_network_error(self, mock_get):
        mock_get.side_effect = Exception("Connection failed")
        from services.news_sources import Source
        src = Source("test", "Test", "international", "https://example.com/rss", "US")
        articles = self.svc._parse_feed(src, 5.0)
        assert articles == []


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
            _make_article(title="Breaking News in Market Today"),  # distance = 1
        ]
        result = self.svc.deduplicate(articles)
        assert len(result) == 1

    def test_different_articles_kept(self):
        articles = [
            _make_article(title="Completely Different Article"),
            _make_article(title="Another Unrelated Story Here"),
        ]
        result = self.svc.deduplicate(articles)
        assert len(result) == 2

    def test_empty_list(self):
        assert self.svc.deduplicate([]) == []

    def test_single_article(self):
        articles = [_make_article()]
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

    def test_rank_by_relevance_empty_query_returns_date_sorted(self):
        articles = [
            _make_article(title="Old", published_at="2026-03-23T10:00:00+00:00"),
            _make_article(title="New", published_at="2026-03-24T10:00:00+00:00"),
        ]
        result = self.svc.rank_by_relevance(articles, "")
        assert result[0].title == "New"

    def test_rank_by_relevance_embedding_failure_falls_back(self):
        mock_emb = MagicMock()
        mock_emb.generate_embedding_np.side_effect = Exception("Model not loaded")
        self.svc._embedding_svc = mock_emb

        articles = [
            _make_article(title="A", published_at="2026-03-23T10:00:00+00:00"),
            _make_article(title="B", published_at="2026-03-24T10:00:00+00:00"),
        ]
        result = self.svc.rank_by_relevance(articles, "query")
        assert len(result) == 2  # All returned, date-sorted


# ── Clustering tests ──────────────────────────────────────────

@pytest.mark.unit
class TestClustering:

    def setup_method(self):
        self.svc = NewsService()

    def test_similar_titles_clustered(self):
        articles = [
            _make_article(title="AI breakthrough in healthcare announced", source="BBC",
                         url="https://bbc.com/article-1"),
            _make_article(title="AI breakthrough in healthcare reported", source="CNN",
                         url="https://cnn.com/article-2"),
            _make_article(title="Completely unrelated sports story", source="ESPN",
                         url="https://espn.com/article-3"),
        ]
        clusters = self.svc.cluster_trending(articles, min_sources=2, limit=5)
        assert len(clusters) >= 1
        assert clusters[0]["coverage"] == 2

    def test_clusters_below_min_sources_filtered(self):
        articles = [
            _make_article(title="Unique story one", source="BBC"),
            _make_article(title="Unique story two", source="CNN"),
        ]
        clusters = self.svc.cluster_trending(articles, min_sources=3, limit=5)
        assert len(clusters) == 0

    def test_representative_title_is_longest(self):
        articles = [
            _make_article(title="Short title about AI", source="BBC",
                         url="https://bbc.com/1"),
            _make_article(title="A much longer and more detailed title about AI breakthroughs in medicine",
                         source="CNN", url="https://cnn.com/2"),
        ]
        clusters = self.svc.cluster_trending(articles, min_sources=2, limit=5)
        if clusters:
            assert "longer" in clusters[0]["title"] or "detailed" in clusters[0]["title"]

    def test_empty_articles(self):
        assert self.svc.cluster_trending([], min_sources=2, limit=5) == []

    def test_clusters_sorted_by_coverage(self):
        articles = [
            _make_article(title="Big story about climate change", source="BBC",
                         url="https://bbc.com/1"),
            _make_article(title="Big story about climate change today", source="CNN",
                         url="https://cnn.com/2"),
            _make_article(title="Big story about climate change reported", source="AP",
                         url="https://ap.com/3"),
            _make_article(title="Small tech news item", source="TechCrunch",
                         url="https://tc.com/4"),
            _make_article(title="Small tech news item today", source="Wired",
                         url="https://wired.com/5"),
        ]
        clusters = self.svc.cluster_trending(articles, min_sources=2, limit=5)
        if len(clusters) >= 2:
            assert clusters[0]["coverage"] >= clusters[1]["coverage"]


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

    @patch("services.news_service.requests.get")
    def test_default_country_code_is_us(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_store = MagicMock()
        mock_store.get.return_value = None
        with patch.object(self.svc, "_get_store", return_value=mock_store):
            self.svc.fetch_google_news("test query")

        call_url = mock_get.call_args[0][0]
        assert "gl=US" in call_url

    @patch("services.news_service.requests.get")
    def test_cache_key_includes_country_code(self, mock_get):
        import hashlib
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_store = MagicMock()
        mock_store.get.return_value = None
        with patch.object(self.svc, "_get_store", return_value=mock_store):
            self.svc.fetch_google_news("query", country_code="MT")

        set_key = mock_store.setex.call_args[0][0]
        expected_hash = hashlib.sha256(("queryMT").encode()).hexdigest()[:16]
        assert expected_hash in set_key

    @patch("services.news_service.requests.get")
    def test_different_country_codes_produce_different_cache_keys(self, mock_get):
        import hashlib
        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_RSS
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch.object(self.svc, "_get_store", return_value=mock_store):
            self.svc.fetch_google_news("query", country_code="US")
            key_us = mock_store.setex.call_args[0][0]
            mock_store.reset_mock()
            self.svc.fetch_google_news("query", country_code="GB")
            key_gb = mock_store.setex.call_args[0][0]

        assert key_us != key_gb


# ── Integration-level tests ───────────────────────────────────

@pytest.mark.unit
class TestSearchIntegration:

    def setup_method(self):
        self.svc = NewsService()

    @patch.object(NewsService, "fetch_feeds")
    @patch.object(NewsService, "fetch_google_news")
    def test_search_with_source_ids_uses_feeds_and_google(self, mock_google, mock_feeds):
        mock_feeds.return_value = [_make_article(title="Feed Article")]
        mock_google.return_value = [_make_article(title="Google Article")]

        mock_emb = MagicMock()
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb.generate_embedding_np.return_value = vec
        mock_emb.generate_embeddings_batch.return_value = [vec, vec]
        self.svc._embedding_svc = mock_emb

        result = self.svc.search("test query", source_ids=["bbc_world"], limit=10)
        assert len(result) >= 1
        mock_feeds.assert_called_once()
        mock_google.assert_called_once()

    @patch.object(NewsService, "fetch_google_news")
    def test_search_without_source_ids_uses_google_only(self, mock_google):
        mock_google.return_value = [_make_article(title="Google Article")]

        mock_emb = MagicMock()
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb.generate_embedding_np.return_value = vec
        mock_emb.generate_embeddings_batch.return_value = [vec]
        self.svc._embedding_svc = mock_emb

        with patch.object(NewsService, "fetch_feeds") as mock_feeds:
            result = self.svc.search("test query", limit=10)
            mock_feeds.assert_not_called()
        mock_google.assert_called_once()
        assert len(result) >= 1

    @patch.object(NewsService, "fetch_google_news")
    def test_search_passes_country_code(self, mock_google):
        mock_google.return_value = []
        with patch.object(self.svc, "deduplicate", return_value=[]):
            with patch.object(self.svc, "rank_by_relevance", return_value=[]):
                self.svc.search("query", country_code="GB")
        _, kwargs = mock_google.call_args
        assert kwargs.get("country_code") == "GB"

    @patch.object(NewsService, "fetch_google_news")
    def test_search_default_country_code_is_us(self, mock_google):
        mock_google.return_value = []
        with patch.object(self.svc, "deduplicate", return_value=[]):
            with patch.object(self.svc, "rank_by_relevance", return_value=[]):
                self.svc.search("query")
        _, kwargs = mock_google.call_args
        assert kwargs.get("country_code") == "US"

    @patch.object(NewsService, "fetch_google_news")
    def test_search_world_awareness_signature_compatible(self, mock_google):
        # world_awareness_service calls: news_svc.search(query=..., limit=...)
        # This verifies that signature works (no source_ids, positional keyword args)
        mock_google.return_value = [_make_article()]
        mock_emb = MagicMock()
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb.generate_embedding_np.return_value = vec
        mock_emb.generate_embeddings_batch.return_value = [vec]
        self.svc._embedding_svc = mock_emb

        result = self.svc.search(query="climate change", limit=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    @patch.object(NewsService, "fetch_google_news")
    def test_search_respects_limit(self, mock_google):
        mock_google.return_value = [_make_article(title=f"Article {i}", url=f"https://example.com/{i}")
                                    for i in range(20)]
        mock_emb = MagicMock()
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb.generate_embedding_np.return_value = vec
        mock_emb.generate_embeddings_batch.return_value = [vec] * 20
        self.svc._embedding_svc = mock_emb

        result = self.svc.search("query", limit=3)
        assert len(result) <= 3


# ── fetch_feeds edge cases ────────────────────────────────────

@pytest.mark.unit
class TestFetchFeedsEdgeCases:

    def setup_method(self):
        self.svc = NewsService()

    def test_fetch_feeds_empty_source_ids(self):
        result = self.svc.fetch_feeds([])
        assert result == []

    def test_fetch_feeds_unknown_source_id(self):
        # Source IDs that don't exist in the registry are silently skipped
        result = self.svc.fetch_feeds(["nonexistent_feed_id"])
        # Should return empty (no sources resolved)
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


# ── fetch_google_news cache hit ───────────────────────────────

@pytest.mark.unit
class TestFetchGoogleNewsCacheHit:

    def setup_method(self):
        self.svc = NewsService()

    @patch("services.news_service.requests.get")
    def test_cache_hit_skips_network_call(self, mock_get):
        cached_data = json.dumps([_make_article(title="Cached Article").to_dict()])
        mock_store = MagicMock()
        mock_store.get.return_value = cached_data.encode()

        with patch.object(self.svc, "_get_store", return_value=mock_store):
            result = self.svc.fetch_google_news("query", country_code="US")

        mock_get.assert_not_called()
        assert len(result) == 1
        assert result[0].title == "Cached Article"


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

    @patch("services.news_service.requests.get")
    def test_xml_with_no_items_or_entries_returns_empty(self, mock_get):
        from services.news_sources import Source
        mock_resp = MagicMock()
        mock_resp.content = b"<rss><channel><title>Empty feed</title></channel></rss>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        src = Source("test", "Test", "international", "https://example.com/rss", "US")
        articles = self.svc._parse_feed(src, 5.0)
        assert articles == []


# ── _derive_domain comprehensive ─────────────────────────────

@pytest.mark.unit
class TestDeriveDomainComprehensive:

    def test_feeds_subdomain_stripped(self):
        from services.news_service import _derive_domain
        assert _derive_domain("https://feeds.reuters.com/rss/topNews") == "reuters.com"

    def test_rss_subdomain_stripped(self):
        from services.news_service import _derive_domain
        assert _derive_domain("https://rss.cnn.com/rss/cnn_topstories.rss") == "cnn.com"

    def test_moxie_subdomain_stripped(self):
        from services.news_service import _derive_domain
        # moxie. prefix should be stripped
        result = _derive_domain("https://moxie.foxnews.com/google-publisher/latest.xml")
        assert result == "foxnews.com"

    def test_empty_url_returns_none(self):
        from services.news_service import _derive_domain
        assert _derive_domain("") is None

    def test_url_without_hostname_returns_none(self):
        from services.news_service import _derive_domain
        assert _derive_domain("not-a-url") is None

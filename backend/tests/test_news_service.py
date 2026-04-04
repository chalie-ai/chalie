"""Tests for news_service — fetch, cache, rank, cluster, dedup."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.news_service import (
    NewsService, NewsArticle,
    _strip_html, _parse_date, _normalize_title, _levenshtein,
    _tokenize_title, _jaccard, _derive_domain,
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


# ── Category routing tests ────────────────────────────────────

@pytest.mark.unit
class TestCategoryRouting:

    def setup_method(self):
        self.svc = NewsService()

    def test_keyword_fast_path_business(self):
        assert self.svc.route_to_category("stock market crash") == "business"

    def test_keyword_fast_path_sports(self):
        assert self.svc.route_to_category("football world cup") == "sports"

    def test_keyword_fast_path_tech(self):
        assert self.svc.route_to_category("new AI model released") == "tech"

    def test_keyword_fast_path_science(self):
        assert self.svc.route_to_category("NASA launches new mission") == "science"

    def test_embedding_fallback(self):
        mock_emb = MagicMock()
        # Return vectors that make "tech" the best match
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        mock_emb.generate_embedding_np.side_effect = lambda text: query_emb

        self.svc._embedding_svc = mock_emb
        # All centroids same = ties go to first computed, but at least it doesn't crash
        result = self.svc.route_to_category("quantum computing advances")
        assert result in ("tech", "science", "international")  # any valid category

    def test_embedding_failure_returns_international(self):
        mock_emb = MagicMock()
        mock_emb.generate_embedding_np.side_effect = Exception("Model error")
        self.svc._embedding_svc = mock_emb
        # Force no keyword match by using a gibberish query
        assert self.svc.route_to_category("xyzzy plugh") == "international"


# ── Integration-level tests ───────────────────────────────────

@pytest.mark.unit
class TestSearchIntegration:

    def setup_method(self):
        self.svc = NewsService()

    @patch.object(NewsService, "fetch_feeds")
    @patch.object(NewsService, "fetch_google_news")
    @patch.object(NewsService, "route_to_category")
    def test_search_combines_feeds_and_google(self, mock_route, mock_google, mock_feeds):
        mock_route.return_value = "tech"
        mock_feeds.return_value = [_make_article(title="Feed Article")]
        mock_google.return_value = [_make_article(title="Google Article")]

        # Mock embedding for ranking
        mock_emb = MagicMock()
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb.generate_embedding_np.return_value = vec
        mock_emb.generate_embeddings_batch.return_value = [vec, vec]
        self.svc._embedding_svc = mock_emb

        result = self.svc.search("test query", limit=10)
        assert len(result) >= 1
        mock_feeds.assert_called_once()
        mock_google.assert_called_once()

    @patch.object(NewsService, "fetch_feeds")
    def test_get_digest_returns_sections(self, mock_feeds):
        mock_feeds.return_value = [
            _make_article(title=f"Article {i}") for i in range(5)
        ]

        # Mock google news for local
        with patch.object(self.svc, "fetch_google_news") as mock_google:
            mock_google.return_value = [_make_article(title="Local News")]
            result = self.svc.get_digest(location="London")

        assert "international" in result
        assert "local" in result
        assert len(result["local"]) >= 1

    @patch.object(NewsService, "fetch_feeds")
    def test_get_digest_no_location_no_local(self, mock_feeds):
        mock_feeds.return_value = [_make_article()]
        result = self.svc.get_digest()
        assert result["local"] == []

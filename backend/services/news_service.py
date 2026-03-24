"""
News Service — RSS fetching, caching, ranking, clustering, and deduplication.
"""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import numpy as np
import requests

from services.time_utils import utc_now, parse_utc
from services import news_sources

logger = logging.getLogger(__name__)

LOG_PREFIX = "[news]"

# ── Constants ─────────────────────────────────────────────────
USER_AGENT = "Chalie-NewsAggregator/1.0 (RSS reader)"
FEED_CACHE_TTL = 600  # 10 minutes
PER_FEED_TIMEOUT = 5.0
TOTAL_BUDGET = 8.0
RELEVANCE_FLOOR = 0.3
LEVENSHTEIN_THRESHOLD = 5
JACCARD_THRESHOLD = 0.5

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "in", "of", "to", "and", "for", "with",
    "on", "at", "by", "it", "its", "as", "are", "was", "be", "has",
    "have", "had", "that", "this", "from", "or", "but", "not", "s", "us",
})

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#039;": "'", "&nbsp;": " "}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Keyword → category fast path
_CATEGORY_KEYWORDS = {
    "tech": ("technology", "software", "programming", "ai", "artificial intelligence",
             "computer", "startup", "app", "cyber", "silicon valley", "coding", "developer"),
    "business": ("stock", "market", "economy", "finance", "investment", "bank",
                 "trade", "gdp", "inflation", "corporate", "ceo", "ipo"),
    "sports": ("football", "soccer", "basketball", "tennis", "cricket", "nba",
               "nfl", "fifa", "olympic", "championship", "league", "match"),
    "science": ("science", "research", "physics", "biology", "chemistry", "space",
                "nasa", "climate", "genome", "quantum", "discovery"),
    "entertainment": ("movie", "film", "music", "celebrity", "netflix", "hollywood",
                      "tv show", "album", "concert", "oscar", "grammy"),
    "us": ("congress", "senate", "white house", "republican", "democrat", "biden",
           "trump", "washington dc", "federal", "supreme court"),
    "uk": ("parliament", "westminster", "boris", "labour", "tory", "nhs",
           "downing", "british", "brexit", "commons"),
}

# Category centroid texts for embedding-based fallback
_CATEGORY_CENTROID_TEXTS = {
    "international": "world news global international diplomacy conflict geopolitics",
    "us": "united states american politics congress washington government federal",
    "uk": "united kingdom british politics parliament london government",
    "tech": "technology software programming artificial intelligence computers startups",
    "business": "business finance economy markets stocks investment corporate trade",
    "science": "science research discovery physics biology space climate",
    "sports": "sports football basketball tennis cricket championship league match",
    "entertainment": "entertainment movies music television celebrity hollywood culture",
}

# RSS subdomain prefixes to strip when deriving domain for Google News site: filter
_RSS_SUBDOMAIN_RE = re.compile(r"^(?:feeds?2?|rss(?:feeds?)?|rssfeeds?|feed|moxie|www\d*)\.", re.IGNORECASE)
_PROXY_HOSTS = {"rsshub.app", "hnrss.org", "feedburner.com", "feeds.feedburner.com"}


@dataclass
class NewsArticle:
    title: str
    description: str
    url: str
    published_at: str
    source: str
    source_id: str
    category: str

    def to_dict(self) -> dict:
        return asdict(self)


class NewsService:
    def __init__(self):
        self._embedding_svc = None
        self._store = None
        self._category_centroids = None

    # ── Lazy accessors ────────────────────────────────────────

    def _get_embedding_service(self):
        if self._embedding_svc is None:
            from services.embedding_service import get_embedding_service
            self._embedding_svc = get_embedding_service()
        return self._embedding_svc

    def _get_store(self):
        from services.memory_client import MemoryClientService
        return MemoryClientService.create_connection()

    # ── Feed fetching ─────────────────────────────────────────

    def fetch_feeds(self, source_ids: list, timeout_per_feed: float = PER_FEED_TIMEOUT,
                    total_budget: float = TOTAL_BUDGET) -> list:
        """Fetch RSS/Atom feeds in parallel with per-feed and total budget timeouts."""
        if not source_ids:
            return []

        sources = []
        for sid in source_ids:
            src = news_sources.get_source_by_id(sid)
            if src:
                sources.append(src)

        if not sources:
            return []

        # Check cache first, collect misses
        store = self._get_store()
        all_articles = []
        to_fetch = []

        for src in sources:
            cache_key = f"news:feed:{src.id}"
            cached = store.get(cache_key)
            if cached:
                try:
                    for item in json.loads(cached):
                        all_articles.append(NewsArticle(**item))
                    continue
                except Exception:
                    pass
            to_fetch.append(src)

        if not to_fetch:
            return all_articles

        # Parallel fetch with budget
        deadline = time.monotonic() + total_budget

        def _fetch_one(src):
            remaining = max(0.5, deadline - time.monotonic())
            timeout = min(timeout_per_feed, remaining)
            return src, self._parse_feed(src, timeout)

        with ThreadPoolExecutor(max_workers=min(len(to_fetch), 8)) as pool:
            futures = {pool.submit(_fetch_one, src): src for src in to_fetch}
            for future in as_completed(futures, timeout=total_budget):
                try:
                    src, articles = future.result()
                    if articles:
                        # Cache
                        store.setex(
                            f"news:feed:{src.id}",
                            FEED_CACHE_TTL,
                            json.dumps([a.to_dict() for a in articles]),
                        )
                        all_articles.extend(articles)
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Feed future failed: {e}")

        return all_articles

    def _parse_feed(self, src, timeout: float) -> list:
        """Fetch and parse a single RSS/Atom feed."""
        try:
            resp = requests.get(
                src.feed_url,
                timeout=timeout,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
                },
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to fetch {src.id}: {e}")
            return []

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError as e:
            logger.debug(f"{LOG_PREFIX} XML parse error for {src.id}: {e}")
            return []

        # Try RSS items first, then Atom entries
        items = root.findall(".//item")
        if items:
            return self._parse_rss_items(items, src)

        # Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//atom:entry", ns)
        if not entries:
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if not entries:
            # Try without namespace
            entries = root.findall(".//entry")
        if entries:
            return self._parse_atom_entries(entries, src)

        return []

    def _parse_rss_items(self, items, src) -> list:
        articles = []
        for item in items:
            title = self._text(item, "title")
            if not title:
                continue
            desc = _strip_html(self._text(item, "description") or "")[:400]
            url = self._text(item, "link") or self._text(item, "guid") or ""
            pub_date = self._text(item, "pubDate")
            if not pub_date:
                # Try dc:date
                for child in item:
                    if child.tag.endswith("}date") or child.tag == "date":
                        pub_date = child.text
                        break
            published_at = _parse_date(pub_date)
            articles.append(NewsArticle(
                title=title.strip(),
                description=desc.strip(),
                url=url.strip(),
                published_at=published_at,
                source=src.name,
                source_id=src.id,
                category=src.category,
            ))
        return articles

    def _parse_atom_entries(self, entries, src) -> list:
        articles = []
        for entry in entries:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            title = self._text_ns(entry, "title", ns) or self._text(entry, "title")
            if not title:
                continue
            desc = self._text_ns(entry, "summary", ns) or self._text_ns(entry, "content", ns)
            if not desc:
                desc = self._text(entry, "summary") or self._text(entry, "content") or ""
            desc = _strip_html(desc)[:400]
            # Link
            url = ""
            for link_el in entry.findall("atom:link", ns) + entry.findall("link"):
                rel = link_el.get("rel", "alternate")
                if rel == "alternate":
                    url = link_el.get("href", "")
                    break
            if not url:
                url = self._text_ns(entry, "id", ns) or self._text(entry, "id") or ""
            pub_date = (self._text_ns(entry, "published", ns) or
                        self._text_ns(entry, "updated", ns) or
                        self._text(entry, "published") or
                        self._text(entry, "updated"))
            published_at = _parse_date(pub_date)
            articles.append(NewsArticle(
                title=title.strip(),
                description=desc.strip(),
                url=url.strip(),
                published_at=published_at,
                source=src.name,
                source_id=src.id,
                category=src.category,
            ))
        return articles

    @staticmethod
    def _text(el, tag: str) -> Optional[str]:
        child = el.find(tag)
        return child.text if child is not None and child.text else None

    @staticmethod
    def _text_ns(el, tag: str, ns: dict) -> Optional[str]:
        child = el.find(f"atom:{tag}", ns)
        return child.text if child is not None and child.text else None

    # ── Google News ───────────────────────────────────────────

    def fetch_google_news(self, query: str, source_id: str = None,
                          timeout: float = PER_FEED_TIMEOUT) -> list:
        """Fetch articles from Google News RSS search endpoint."""
        store = self._get_store()
        q_parts = [query]

        # Optionally scope to a source domain
        if source_id:
            src = news_sources.get_source_by_id(source_id)
            if src:
                domain = _derive_domain(src.feed_url)
                if domain:
                    q_parts.append(f"site:{domain}")

        full_query = " ".join(q_parts)
        cache_key = f"news:google:{hashlib.sha256(full_query.encode()).hexdigest()[:16]}"
        cached = store.get(cache_key)
        if cached:
            try:
                return [NewsArticle(**a) for a in json.loads(cached)]
            except Exception:
                pass

        url = f"https://news.google.com/rss/search?q={quote_plus(full_query)}&hl=en&gl=US&ceid=US:en"
        try:
            resp = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Google News fetch failed: {e}")
            return []

        # Google News RSS uses standard RSS format
        dummy_src = news_sources.Source("google_news", "Google News", "international", url, "US")
        articles = self._parse_rss_items(root.findall(".//item"), dummy_src)

        if articles:
            store.setex(cache_key, FEED_CACHE_TTL, json.dumps([a.to_dict() for a in articles]))
        return articles

    # ── Deduplication ─────────────────────────────────────────

    def deduplicate(self, articles: list) -> list:
        """Remove near-duplicate articles by Levenshtein distance on normalized titles."""
        if len(articles) <= 1:
            return articles

        seen = []
        result = []
        for article in articles:
            norm = _normalize_title(article.title)
            is_dup = False
            for prev in seen:
                if _levenshtein(norm, prev) <= LEVENSHTEIN_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(norm)
                result.append(article)
        return result

    # ── Relevance ranking ─────────────────────────────────────

    def rank_by_relevance(self, articles: list, query: str) -> list:
        """Rank articles by cosine similarity of title embeddings to query."""
        if not articles or not query:
            return sorted(articles, key=lambda a: a.published_at, reverse=True)

        emb_svc = self._get_embedding_service()
        try:
            query_emb = emb_svc.generate_embedding_np(query)
            title_embs = emb_svc.generate_embeddings_batch([a.title for a in articles])
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding failed, falling back to date sort: {e}")
            return sorted(articles, key=lambda a: a.published_at, reverse=True)

        scored = []
        for article, title_emb in zip(articles, title_embs):
            score = float(np.dot(query_emb, title_emb))
            scored.append((article, score))

        # Filter by relevance floor
        relevant = [(a, s) for a, s in scored if s >= RELEVANCE_FLOOR]
        if not relevant:
            # Fallback to date-sorted
            return sorted(articles, key=lambda a: a.published_at, reverse=True)

        # Sort by score desc, then date desc
        relevant.sort(key=lambda x: (-x[1], x[0].published_at), reverse=False)
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in relevant]

    # ── Trending clustering ───────────────────────────────────

    def cluster_trending(self, articles: list, min_sources: int = 2, limit: int = 5) -> list:
        """Jaccard similarity clustering on tokenized titles."""
        if not articles:
            return []

        clusters = []  # list of {"articles": [], "word_sets": []}

        for article in articles:
            words = _tokenize_title(article.title)
            if not words:
                continue
            merged = False
            for cluster in clusters:
                for ws in cluster["word_sets"]:
                    if _jaccard(words, ws) >= JACCARD_THRESHOLD:
                        cluster["articles"].append(article)
                        cluster["word_sets"].append(words)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                clusters.append({"articles": [article], "word_sets": [words]})

        # Finalize
        result = []
        for cluster in clusters:
            # Deduplicate within cluster
            seen_urls = set()
            seen_titles = set()
            deduped = []
            for a in cluster["articles"]:
                if a.url in seen_urls:
                    continue
                t_lower = a.title.lower()
                if t_lower in seen_titles:
                    continue
                seen_urls.add(a.url)
                seen_titles.add(t_lower)
                deduped.append(a)

            if len(deduped) < min_sources:
                continue

            # Representative title = longest
            rep_title = max(deduped, key=lambda a: len(a.title)).title
            result.append({
                "title": rep_title,
                "articles": deduped,
                "coverage": len(deduped),
            })

        result.sort(key=lambda c: c["coverage"], reverse=True)
        return result[:limit]

    # ── Category routing ──────────────────────────────────────

    def route_to_category(self, query: str) -> str:
        """Route a query to the best-matching news category."""
        q_lower = query.lower()

        # Keyword fast path
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    return category

        # Embedding fallback
        try:
            centroids = self._get_category_centroids()
            emb_svc = self._get_embedding_service()
            query_emb = emb_svc.generate_embedding_np(query)

            best_cat = "international"
            best_score = -1.0
            for cat, centroid in centroids.items():
                score = float(np.dot(query_emb, centroid))
                if score > best_score:
                    best_score = score
                    best_cat = cat
            return best_cat
        except Exception:
            return "international"

    def _get_category_centroids(self) -> dict:
        if self._category_centroids is None:
            emb_svc = self._get_embedding_service()
            self._category_centroids = {}
            for cat, text in _CATEGORY_CENTROID_TEXTS.items():
                self._category_centroids[cat] = emb_svc.generate_embedding_np(text)
        return self._category_centroids

    # ── Convenience methods ───────────────────────────────────

    def search(self, query: str, source_ids: list = None, limit: int = 10) -> list:
        """Search news: fetch + Google News → deduplicate → rank → slice."""
        if not source_ids:
            cat = self.route_to_category(query)
            source_ids = [s.id for s in news_sources.get_sources_by_category(cat)]

        articles = self.fetch_feeds(source_ids)
        articles.extend(self.fetch_google_news(query))
        articles = self.deduplicate(articles)
        articles = self.rank_by_relevance(articles, query)
        return articles[:limit]

    def get_digest(self, source_ids: list = None, topics: list = None,
                   location: str = None, limit_intl: int = 10,
                   limit_local: int = 5) -> dict:
        """Get a news digest with international and optional local sections."""
        if not source_ids:
            source_ids = [s.id for s in news_sources.get_sources_by_category("international")]

        intl = self.fetch_feeds(source_ids)
        intl = self.deduplicate(intl)
        if topics:
            intl = self.rank_by_relevance(intl, " ".join(topics))
        else:
            intl.sort(key=lambda a: a.published_at, reverse=True)
        intl = intl[:limit_intl]

        local = []
        if location:
            local = self.fetch_google_news(f"local news {location}")
            local = self.deduplicate(local)
            local.sort(key=lambda a: a.published_at, reverse=True)
            local = local[:limit_local]

        return {"international": intl, "local": local}


# ── Module-level helpers ──────────────────────────────────────

def _strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_date(date_str: Optional[str]) -> str:
    """Parse RFC 2822 or ISO 8601 date string to ISO 8601 UTC."""
    if not date_str:
        return utc_now().isoformat()
    # Try RFC 2822 first (common in RSS)
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        pass
    # Try ISO 8601
    try:
        dt = parse_utc(date_str)
        return dt.isoformat()
    except Exception:
        pass
    return utc_now().isoformat()


def _normalize_title(title: str) -> str:
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", title.lower())).strip()


def _levenshtein(s1: str, s2: str) -> int:
    """Two-row DP Levenshtein distance."""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _tokenize_title(title: str) -> set:
    words = _PUNCT_RE.sub(" ", title.lower()).split()
    return {w for w in words if w and w not in _STOPWORDS}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _derive_domain(feed_url: str) -> Optional[str]:
    """Derive a publisher domain from an RSS feed URL."""
    try:
        hostname = urlparse(feed_url).hostname
        if not hostname:
            return None
        if hostname in _PROXY_HOSTS:
            return None
        return _RSS_SUBDOMAIN_RE.sub("", hostname)
    except Exception:
        return None

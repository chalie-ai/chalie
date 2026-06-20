"""
News Service — RSS fetching, caching, ranking, clustering, and deduplication.
"""

import calendar
import hashlib
import html as _html
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus, urlparse

# feedparser: tolerant RSS/Atom/RDF parsing + media/date normalisation,
# replaces hand-rolled ElementTree pipeline
import feedparser

# rapidfuzz: SIMD-accelerated Levenshtein, replaces two-row DP implementation
from rapidfuzz.distance import Levenshtein as _Levenshtein

# nh3: HTML sanitiser strips tags from feed descriptions (same lib used in services/markup.py)
import nh3
import numpy as np
import requests

from services.time_utils import utc_now
from services import news_sources

logger = logging.getLogger(__name__)

LOG_PREFIX = "[news]"


class NewsFetchError(Exception):
    """A news provider was unreachable or returned a transport-level error.

    Raised by :meth:`NewsService.fetch_google_news` when the HTTP fetch to the
    Google News RSS endpoint fails (connection refused, timeout, non-2xx). The
    message carries the provider/URL context. The ability maps this to
    ``code=provider-unreachable`` instead of letting a dead provider masquerade
    as an empty result set.

    Per-feed RSS failures inside :meth:`NewsService._parse_feed` are NOT raised:
    a single dead feed in a multi-feed aggregate is normal and tolerated.
    """


# ── Constants ─────────────────────────────────────────────────
USER_AGENT = "Chalie-NewsAggregator/1.0 (RSS reader)"

# Google News RSS search endpoint base. Module-level so the search query, country
# code, and language are appended at call time and so tests can point the fetch at
# a closed loopback port to exercise the real transport-failure path.
GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"
FEED_CACHE_TTL = 600  # 10 minutes
PER_FEED_TIMEOUT = 5.0
TOTAL_BUDGET = 8.0
RELEVANCE_FLOOR = 0.3
LEVENSHTEIN_THRESHOLD = 5

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# RSS subdomain prefixes to strip when deriving domain for Google News site: filter
_RSS_SUBDOMAIN_RE = re.compile(r"^(?:feeds?2?|rss(?:feeds?)?|feed|moxie|www\d*)\.", re.IGNORECASE)
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
    image_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class NewsService:
    def __init__(self):
        self._embedding_svc = None
        self._store = None

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

        # feedparser is tolerant of malformed XML (sets feed.bozo but recovers what it can).
        # We return [] only when feedparser yields zero usable entries, not on bozo alone.
        feed = feedparser.parse(resp.content)
        feed_image = getattr(getattr(feed, "feed", None), "image", None)
        feed_image_url = getattr(feed_image, "href", "") if feed_image else ""
        return [
            article
            for entry in feed.entries
            if (article := _entry_to_article(entry, src, feed_image_url)) is not None
        ]

    # ── Google News ───────────────────────────────────────────

    def fetch_google_news(self, query: str, country_code: str = "US",
                          timeout: float = PER_FEED_TIMEOUT) -> list:
        store = self._get_store()
        full_query = query
        cache_key = f"news:google:{hashlib.sha256((full_query + country_code).encode()).hexdigest()[:16]}"
        cached = store.get(cache_key)
        if cached:
            try:
                return [NewsArticle(**a) for a in json.loads(cached)]
            except Exception:
                pass

        url = f"{GOOGLE_NEWS_BASE}?q={quote_plus(full_query)}&hl=en&gl={country_code}&ceid={country_code}:en"
        try:
            resp = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Google News fetch failed: {e}")
            raise NewsFetchError(f"Google News fetch failed for {url!r}: {e}") from e

        feed = feedparser.parse(resp.content)
        dummy_src = news_sources.Source("google_news", "Google News", "international", url, "US")
        articles = [
            article
            for entry in feed.entries
            if (article := _entry_to_article(entry, dummy_src, "")) is not None
        ]

        if articles:
            store.setex(cache_key, FEED_CACHE_TTL, json.dumps([a.to_dict() for a in articles]))
        return articles

    # ── Deduplication ─────────────────────────────────────────

    def deduplicate(self, articles: list) -> list:
        if len(articles) <= 1:
            return articles

        seen = []
        result = []
        for article in articles:
            norm = _normalize_title(article.title)
            is_dup = any(
                _Levenshtein.distance(norm, prev) <= LEVENSHTEIN_THRESHOLD
                for prev in seen
            )
            if not is_dup:
                seen.append(norm)
                result.append(article)
        return result

    # ── Relevance ranking ─────────────────────────────────────

    def rank_by_relevance(self, articles: list, query: str) -> list:
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
        relevant.sort(key=lambda x: (x[1], x[0].published_at), reverse=True)
        return [a for a, _ in relevant]

    # ── Convenience methods ───────────────────────────────────

    def search(self, query: str, source_ids: list = None, limit: int = 10, country_code: str = "US") -> list:
        """Search news: fetch + Google News → deduplicate → rank → slice."""
        if source_ids:
            articles = self.fetch_feeds(source_ids)
            articles.extend(self.fetch_google_news(query, country_code=country_code))
        else:
            articles = self.fetch_google_news(query, country_code=country_code)
        articles = self.deduplicate(articles)
        articles = self.rank_by_relevance(articles, query)
        return articles[:limit]


# ── Module-level helpers ──────────────────────────────────────

def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = nh3.clean(text, tags=set())
    return _WHITESPACE_RE.sub(" ", _html.unescape(cleaned).replace("\xa0", " ")).strip()


def _feedparser_date_to_utc_str(parsed) -> str:
    """Convert feedparser's UTC struct_time to an ISO 8601 UTC string.

    feedparser normalises all date formats (RFC 2822, ISO 8601, W3CDTF, POSIX)
    to a UTC struct_time. Returns utc_now() when parsed is None.
    """
    if parsed is None:
        return utc_now().isoformat()
    ts = calendar.timegm(parsed)  # treats struct_time as UTC (not local)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _feedparser_image(entry, feed_image_url: str = "") -> str:
    """Extract a thumbnail URL from a feedparser entry (RSS or Atom unified).

    Priority: media:thumbnail → media:content (image/* or medium=image)
    → enclosures (image/*, feedparser key is 'href' not 'url')
    → feed-level channel image fallback.
    """
    thumbs = entry.get("media_thumbnail") or []
    if thumbs:
        url = thumbs[0].get("url", "")
        if url:
            return url.strip()

    for media in (entry.get("media_content") or []):
        t = (media.get("type") or "").lower()
        m = (media.get("medium") or "").lower()
        if t.startswith("image/") or m == "image":
            url = media.get("url", "")
            if url:
                return url.strip()

    for enc in (entry.get("enclosures") or []):
        t = (enc.get("type") or "").lower()
        if t.startswith("image/"):
            href = enc.get("href", "")
            if href:
                return href.strip()

    return feed_image_url.strip() if feed_image_url else ""


def _entry_to_article(entry, src, feed_image_url: str) -> Optional[NewsArticle]:
    """Build a NewsArticle from a feedparser entry.

    Shared by _parse_feed and fetch_google_news so there is one parsing path.
    Returns None when the entry has no usable title.
    """
    title = (entry.get("title") or "").strip()
    if not title:
        return None

    content = entry.get("content") or []
    raw_desc = entry.get("summary") or (content[0].get("value", "") if content else "")
    desc = _strip_html(raw_desc)[:400]
    url = (entry.get("link") or entry.get("id") or "").strip()
    published_at = _feedparser_date_to_utc_str(
        entry.get("published_parsed") or entry.get("updated_parsed")
    )
    return NewsArticle(
        title=title,
        description=desc.strip(),
        url=url,
        published_at=published_at,
        source=src.name,
        source_id=src.id,
        category=src.category,
        image_url=_feedparser_image(entry, feed_image_url),
    )


def _normalize_title(title: str) -> str:
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", title.lower())).strip()


def _derive_domain(feed_url: str) -> Optional[str]:
    try:
        hostname = urlparse(feed_url).hostname
        if not hostname or hostname in _PROXY_HOSTS:
            return None
        return _RSS_SUBDOMAIN_RE.sub("", hostname)
    except Exception:
        return None

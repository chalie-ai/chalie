"""
News Tool — search, digest, trending, and source browsing across 56 RSS feeds.
"""

import logging

from services import news_sources
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_service = None


def _get_service():
    global _service
    if _service is None:
        from services.news_service import NewsService
        _service = NewsService()
    return _service


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    action = (params.get("action") or "").lower()
    config = config or {}
    telemetry = telemetry or {}

    try:
        if action == "search":
            return _do_search(params, config)
        elif action == "digest":
            return _do_digest(params, config, telemetry)
        elif action == "trending":
            return _do_trending(params)
        elif action == "sources":
            return _do_sources(params)
        else:
            return {"text": "", "error": f"Unknown action: {action}. Use: search, digest, trending, sources"}
    except Exception as e:
        logger.error(f"[news-tool] {action} failed: {e}", exc_info=True)
        return {"text": "", "error": str(e)}


def _do_search(params: dict, config: dict) -> dict:
    query = params.get("query")
    if not query:
        return {"text": "", "error": "Search requires a 'query' parameter"}

    source = params.get("source")
    limit = min(params.get("limit", 10), 20)
    svc = _get_service()

    source_ids = _resolve_source_ids(source)
    articles = svc.search(query, source_ids=source_ids, limit=limit)

    if not articles:
        return {"text": f"No news articles found for \"{query}\".", "title": f"News: \"{query}\"", "error": "No news articles found", "failure_class": "external"}

    text = _format_articles(articles, limit)
    return {"text": text, "title": f"News: \"{query}\"", "error": ""}


def _do_digest(params: dict, config: dict, telemetry: dict) -> dict:
    svc = _get_service()

    # Resolve source
    preferred = config.get("preferred_source", "bbc_world")
    source_ids = _resolve_source_ids(preferred)
    if not source_ids:
        source_ids = [s.id for s in news_sources.get_sources_by_category("international")]

    # Add fallback international sources
    fallbacks = ["reuters_world", "ap_world", "aljazeera", "france24_en", "dw_world"]
    existing = set(source_ids)
    for fb in fallbacks:
        if fb not in existing:
            source_ids.append(fb)
            if len(source_ids) >= 5:
                break

    # Location
    location = (config.get("location_override") or
                telemetry.get("city") or
                telemetry.get("country") or None)

    # Topics
    topics_str = config.get("topics", "")
    topics = [t.strip() for t in topics_str.split(",") if t.strip()] if topics_str else None

    result = svc.get_digest(source_ids=source_ids, topics=topics, location=location)

    lines = []
    if result["international"]:
        lines.append("INTERNATIONAL")
        lines.append(_format_articles(result["international"]))
    if result["local"]:
        lines.append(f"LOCAL — {location}")
        lines.append(_format_articles(result["local"]))

    if not lines:
        return {"text": "No news articles available.", "title": "News Digest", "error": "No news articles available", "failure_class": "external"}
    return {"text": "\n".join(lines), "title": "News Digest", "error": ""}


def _do_trending(params: dict) -> dict:
    svc = _get_service()
    category = params.get("category", "international")
    if category not in news_sources.CATEGORIES:
        category = "international"

    limit = min(params.get("limit", 5), 20)
    min_sources = params.get("min_sources", 2)

    source_ids = [s.id for s in news_sources.get_sources_by_category(category)]
    articles = svc.fetch_feeds(source_ids)
    articles = svc.deduplicate(articles)
    clusters = svc.cluster_trending(articles, min_sources=min_sources, limit=limit)

    if not clusters:
        return {"text": f"No trending stories found in {category}.", "title": f"Trending: {category}", "error": "No trending stories found", "failure_class": "external"}

    text = _format_clusters(clusters)
    return {"text": text, "title": f"Trending: {category}", "error": ""}


def _do_sources(params: dict) -> dict:
    category = params.get("category")
    query = params.get("source")

    if category and category in news_sources.CATEGORIES:
        sources = news_sources.get_sources_by_category(category)
    elif query:
        sources = news_sources.search_sources(query)
    else:
        sources = list(news_sources.SOURCES)

    # Intersect if both provided
    if category and query:
        cat_sources = news_sources.get_sources_by_category(category)
        name_sources = news_sources.search_sources(query)
        cat_ids = {s.id for s in cat_sources}
        sources = [s for s in name_sources if s.id in cat_ids]

    # Sort by canonical category order, then name
    cat_order = {c: i for i, c in enumerate(news_sources.CATEGORIES)}
    sources.sort(key=lambda s: (cat_order.get(s.category, 99), s.name))

    text = _format_sources(sources)
    return {"text": text, "title": "News Sources", "error": ""}


# ── Helpers ───────────────────────────────────────────────────

def _resolve_source_ids(source_str: str) -> list:
    """Resolve a source string to a list of source IDs."""
    if not source_str:
        return []
    # Try exact ID
    if news_sources.get_source_by_id(source_str):
        return [source_str]
    # Try name search
    matches = news_sources.search_sources(source_str)
    if matches:
        return [matches[0].id]
    return []


def _format_articles(articles, max_items=20) -> str:
    lines = []
    for a in articles[:max_items]:
        lines.append(f"\u2022 {a.title}")
        lines.append(f"  {a.source} \u00b7 {_relative_time(a.published_at)}")
        if a.description:
            lines.append(f"  {a.description[:150]}")
        lines.append("")
    return "\n".join(lines)


def _format_clusters(clusters) -> str:
    lines = []
    for i, c in enumerate(clusters, 1):
        lines.append(f"{i}. {c['title']} ({c['coverage']} sources)")
        for a in c["articles"][:3]:
            lines.append(f"   \u2022 {a.title} \u2014 {a.source}")
        lines.append("")
    return "\n".join(lines)


def _format_sources(sources) -> str:
    lines = []
    current_cat = None
    for s in sources:
        if s.category != current_cat:
            current_cat = s.category
            lines.append(f"\n{current_cat.upper()}")
        lines.append(f"  \u2022 {s.name} [{s.id}]")
    return "\n".join(lines)


def _relative_time(iso_str: str) -> str:
    try:
        dt = parse_utc(iso_str)
        delta = utc_now() - dt
        mins = int(delta.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return ""

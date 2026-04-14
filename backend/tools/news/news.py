"""
News Tool — search news articles across global sources.
"""

import logging

from services import news_sources
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_service = None

_COUNTRY_CODE_MAP = {
    "united states": "US", "united kingdom": "GB", "malta": "MT",
    "germany": "DE", "france": "FR", "japan": "JP", "canada": "CA",
    "australia": "AU", "italy": "IT", "spain": "ES", "netherlands": "NL",
    "ireland": "IE", "india": "IN", "brazil": "BR", "mexico": "MX",
    "qatar": "QA", "south africa": "ZA", "new zealand": "NZ",
    "singapore": "SG", "hong kong": "HK", "sweden": "SE",
    "norway": "NO", "denmark": "DK", "finland": "FI", "switzerland": "CH",
    "austria": "AT", "belgium": "BE", "portugal": "PT", "poland": "PL",
    "israel": "IL", "south korea": "KR", "taiwan": "TW",
    "united arab emirates": "AE", "saudi arabia": "SA",
}


def _get_service():
    global _service
    if _service is None:
        from services.news_service import NewsService
        _service = NewsService()
    return _service


def _resolve_country_code(country) -> str:
    if not country:
        return "US"
    return _COUNTRY_CODE_MAP.get(country.lower().strip(), "US")


def execute(_topic, params: dict, config=None, telemetry=None) -> dict:
    query = (params.get("query") or "").strip()
    if not query:
        return {"text": "", "error": "A 'query' parameter is required."}

    category = params.get("category")
    telemetry = telemetry or {}

    try:
        svc = _get_service()
        country_code = _resolve_country_code(telemetry.get("country"))

        if category:
            source_ids = [s.id for s in news_sources.get_sources_by_category(category)]
            articles = svc.fetch_feeds(source_ids)
            articles.extend(svc.fetch_google_news(query, country_code=country_code))
            articles = svc.deduplicate(articles)
            articles = svc.rank_by_relevance(articles, query)
        else:
            articles = svc.fetch_google_news(query, country_code=country_code)

        if not articles:
            return {"text": f"No news found for \"{query}\".", "title": f"News: \"{query}\""}

        text = _format_articles(articles[:10])
        return {"text": text, "title": f"News: \"{query}\""}
    except Exception as e:
        logger.error(f"[news-tool] failed: {e}", exc_info=True)
        return {"text": "", "error": str(e)}


def _format_articles(articles) -> str:
    lines = []
    for a in articles:
        lines.append(f"\u2022 {a.title}")
        lines.append(f"  {a.source} \u00b7 {_relative_time(a.published_at)}")
        if a.description:
            lines.append(f"  {a.description[:150]}")
        lines.append("")
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

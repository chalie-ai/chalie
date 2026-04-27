"""
NewsAbility — Search news articles across global sources.
"""

import logging
from typing import ClassVar

from abilities._base import Ability
from services import news_sources
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)


class NewsAbility(Ability):
    NAME = "news"
    SUMMARY = "Search news articles across global sources by query, with optional category filtering for broad topic browsing."
    EXAMPLES = [
        "what's in the news about artificial intelligence today",
        "latest tech news",
        "what happened in sports today",
        "news about the US election",
        "any recent business news about Apple",
        "what's happening in science this week",
        "show me today's UK headlines",
        "news about climate change legislation",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "category": {
                "type": "string",
                "enum": ["tech", "business", "sports", "science", "entertainment", "us", "uk"],
                "description": "Narrow to a news category. Use only for broad topic browsing.",
            },
        },
        "required": ["query"],
    }
    ALWAYS_AVAILABLE = False
    TIMEOUT = 10

    _service: ClassVar[None] = None

    _COUNTRY_CODE_MAP: ClassVar[dict] = {
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

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        query = (params.get("query") or "").strip()
        if not query:
            return {"text": "", "error": "A 'query' parameter is required."}

        category = params.get("category")
        telemetry = telemetry or {}

        try:
            svc = self._get_service()
            country_code = self._resolve_country_code(telemetry.get("country"))

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

    @classmethod
    def _get_service(cls):
        if cls._service is None:
            from services.news_service import NewsService
            cls._service = NewsService()
        return cls._service

    @classmethod
    def _resolve_country_code(cls, country) -> str:
        if not country:
            return "US"
        return cls._COUNTRY_CODE_MAP.get(country.lower().strip(), "US")


def _format_articles(articles) -> str:
    lines = []
    for a in articles:
        lines.append(f"\u2022 {a.title}")
        lines.append(f"  {a.source} \u00b7 {TimeFormatterService.ago(a.published_at)}")
        if a.description:
            lines.append(f"  {a.description[:150]}")
        lines.append("")
    return "\n".join(lines)

"""
NewsAbility — Search news articles across global sources.

Rich-media rendering:
  When ``_rich_media_ordinal`` is present (user channel only), returns a
  structured JSON payload + instruction trailer so the frontend renders an
  article card.  The first article's image (if any) populates the thumbnail.
"""

import json
import logging
from typing import ClassVar, Optional

from abilities._base import Ability
from services import news_sources
from services.news_service import NewsService
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

_RICH_MEDIA_INSTRUCTION = (
    "This tool supports rich-media rendering. You MUST present this result by "
    "wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a news article card; without it, the user sees "
    "only plain text. Write a single paragraph that synthesises the key story "
    "for the user. Example: \"Here's the latest — "
    "<span id='{tag}'>The EU opened its first audits under the AI Act today, "
    "targeting three major companies you've been tracking.</span>\""
    "\n\nThumbnail (optional): if the data above includes an `image_candidates` "
    "array, each entry has a `url`, the `source_title` of the article it came "
    "from, and `ocr_text` (text scanned by local OCR — often empty for photos). "
    "If a candidate's source article matches the story you are synthesising, "
    "add `data-image='<that url>'` to the span, e.g. "
    "<span id='{tag}' data-image='https://example.com/pic.jpg'>your synthesis</span>. "
    "Use a URL verbatim from `image_candidates`. Match by `source_title` first; "
    "treat empty `ocr_text` as neutral (most news photos have no embedded text). "
    "Only omit `data-image` if no candidate's source matches your synthesis."
)


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
    TIMEOUT = 10

    _service: ClassVar[Optional[NewsService]] = None

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

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        query = (params.get("query") or "").strip()
        if not query:
            return {"text": "", "error": "A 'query' parameter is required."}

        ordinal = params.get("_rich_media_ordinal")
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

            top = articles[:10]

            if ordinal is None:
                return {"text": _format_articles(top), "title": f"News: \"{query}\""}

            return _serialise_rich(top, ordinal)
        except Exception as e:
            logger.error(f"[news-tool] failed: {e}", exc_info=True)
            return {"text": "", "error": str(e)}

    @classmethod
    def _get_service(cls):
        if cls._service is None:
            cls._service = NewsService()
        return cls._service

    @classmethod
    def _resolve_country_code(cls, country) -> str:
        if not country:
            return "US"
        return cls._COUNTRY_CODE_MAP.get(country.lower().strip(), "US")


def _serialise_rich(articles, ordinal: int) -> str:
    tag = f"news_{ordinal}"
    results = []
    image_items: list[tuple[str, str]] = []
    for a in articles:
        entry = {"title": a.title, "source": a.source, "url": getattr(a, "url", "")}
        if a.description:
            entry["desc"] = a.description[:200]
        results.append(entry)
        img = getattr(a, "image_url", "") or ""
        if img:
            image_items.append((img, a.title or ""))

    payload: dict = {"results": results}
    if image_items:
        try:
            from services.image_candidate_service import build_image_candidates
            candidates = build_image_candidates(image_items)
        except Exception as exc:
            logger.warning("[news-tool] image candidate shortlist failed: %s", exc)
            candidates = []
        if candidates:
            payload["image_candidates"] = candidates
    data_json = json.dumps(payload)
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return f"{data_json}\n\n{instruction}"


def _format_articles(articles) -> str:
    lines = []
    for a in articles:
        lines.append(f"• {a.title}")
        lines.append(f"  {a.source} · {TimeFormatterService.ago(a.published_at)}")
        if a.description:
            lines.append(f"  {a.description[:150]}")
        lines.append("")
    return "\n".join(lines)

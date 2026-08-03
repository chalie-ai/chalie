"""
NewsAbility — Search news articles across global sources.

Result contract:
  ``run()`` returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
  success body is a JSON list of ``{title, source, url, published_at, snippet}``
  rows (``snippet`` is the description truncated) with a ``count`` meta —
  verbatim, machine-readable headlines the model quotes instead of re-prosing a
  blob. Zero articles from a provider that ANSWERED is a loud
  ``code=no-results`` error, never a quiet zero-row success; the ``degraded``
  meta is preserved on that error when the category path was degraded.

  A provider that is unreachable raises ``NewsFetchError`` from the service; the
  ability maps it to ``code=provider-unreachable`` with a recovery hint — a dead
  provider is never an empty success. On the category path (category feeds +
  Google News aggregated) a Google News failure is tolerated when the category
  feeds produced articles (``meta degraded=true``); it is ``provider-unreachable``
  only when nothing was gathered at all.
"""

import logging
from typing import ClassVar, Optional, cast

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.news_params_bag import CATEGORIES, NewsParamsBag
from contracts.params.param_bag import ParamBag
from services import news_sources
from services.news_service import NewsService
from exceptions import NewsFetchError

logger = logging.getLogger(__name__)

_SNIPPET_LEN = 200


class NewsAbility(Ability[NewsParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False  # delegate-exclusive; pinned on WebSearchConfig only

    # Pre-gated by the dispatcher BEFORE run() and BEFORE the policy gate: an
    # action-less tool whose only hard requirement is a non-empty query. The ""
    # key covers action-less tools (see ToolDispatcher._prevalidate); a missing or
    # blank query is rejected as one code=missing-params error naming 'query', so
    # run() never sees a query-less call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    # The typed input contract: the dispatch seam builds the bag via
    # NewsParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = NewsParamsBag
    NAME: ClassVar[str] = "news"
    RETURNS_UNTRUSTED_CONTENT: ClassVar[bool] = True  # third-party feed text

    def get_summary(self) -> str:
        return "Search news articles across global sources by query, with optional category filtering for broad topic browsing."

    def get_examples(self) -> list[str]:
        return [
            "what's in the news about artificial intelligence today",
            "latest tech news",
            "what happened in sports today",
            "news about the US election",
            "any recent business news about Apple",
            "what's happening in science this week",
            "show me today's UK headlines",
            "news about climate change legislation",
        ]

    def get_search_tooltip(self) -> str:
        return "news article search"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Pivot to web search when the headlines miss the user's intent."""
        from abilities.search import SearchAbility  # noqa: PLC0415

        return f"If the results don't match the context you're looking for, pivot to using the `{SearchAbility.NAME}` tool to get data from other sources online."

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "string",
                "description": "What to search for.",
            },
            Keys.category: {
                "type": "string",
                "enum": list(CATEGORIES),
                "description": "Narrow to a news category. Use only for broad topic browsing.",
            },
        },
        "required": [Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _service: ClassVar[Optional[NewsService]] = None

    _COUNTRY_CODE_MAP: ClassVar[dict[str, str]] = {
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

    def run(self, params: NewsParamsBag) -> ToolResult:
        # The bag is built (= validated) at the dispatch seam: query is proven
        # non-blank and category is None or a proven CATEGORIES member — a bogus
        # category was already rejected loudly (invalid-param), never silently
        # degraded into an uncategorised search.
        query = params.query
        category = params.category
        telemetry = self.telemetry or {}

        svc = self._get_service()
        country_code = self._resolve_country_code(telemetry.get("country"))

        degraded = False
        if category:
            source_ids = [s.id for s in news_sources.get_sources_by_category(category)]
            articles = svc.fetch_feeds(source_ids)
            try:
                articles.extend(svc.fetch_google_news(query, country_code=country_code))
            except NewsFetchError as exc:
                # Partial-outage precedent: if the category feeds produced
                # articles, Google News failing is a degraded-but-usable result;
                # only when nothing was gathered is the whole call unreachable.
                if not articles:
                    return self._provider_unreachable(exc)
                logger.warning("[news-tool] Google News degraded on category path: %s", exc)
                degraded = True
            articles = svc.deduplicate(articles)
            articles = svc.rank_by_relevance(articles, query)
        else:
            try:
                articles = svc.fetch_google_news(query, country_code=country_code)
            except NewsFetchError as exc:
                return self._provider_unreachable(exc)

        top = articles[:10]
        rows = [
            {
                "title": a.title,
                "source": a.source,
                "url": getattr(a, "url", "") or "",
                "published_at": a.published_at,
                "snippet": (a.description or "")[:_SNIPPET_LEN],
            }
            for a in top
        ]

        if not rows:
            if degraded:
                return ToolResult.no_results(degraded=True)
            return ToolResult.no_results()

        if degraded:
            return ToolResult.ok(rows, count=len(rows), degraded=True)
        return ToolResult.ok(rows, count=len(rows))

    @staticmethod
    def _provider_unreachable(exc: NewsFetchError) -> ToolResult:
        return ToolResult.err(
            f"The news provider was unreachable: {exc}",
            code="provider-unreachable",
            hint="the news source is down or unreachable — retry shortly, or tell the user news is temporarily unavailable.",
        )

    @classmethod
    def _get_service(cls) -> "NewsService":
        if cls._service is None:
            cls._service = NewsService()
        return cls._service

    @classmethod
    def _resolve_country_code(cls, country: object) -> str:
        if not country:
            return "US"
        return cls._COUNTRY_CODE_MAP.get(cast("str", country).lower().strip(), "US")

"""
SearchAbility — Multi-provider search with semantic routing and relevance ranking.

Routes queries to best provider(s) via embedding similarity.  Falls back to
DuckDuckGo (DDG) — but ONLY when a real provider was attempted and came back
empty, never to silently mask a routing miss.

Result contract:
  ``run()`` returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
  success body is an XML-like string of ``<result index= score= [date=]>`` blocks
  rendered by ``tools/search/render.py``.  Results are relevance-ranked via
  embedding cosine similarity (``tools/search/router.rank_results``) and capped
  to the global top-5 most query-relevant hits across all providers.  Summaries
  for results with blank snippets are best-effort enriched from page meta tags
  (``tools/search/enrich.enrich_missing_summaries``).  A ``count`` meta accompanies
  every success; ``fallback=ddg`` meta is set when DDG supplied any results.

The search assets (search_tool_providers.sqlite, fetcher, router) stay in
tools/search/ until Phase 4 — this ability imports them from there.
"""

import logging
import time
from typing import ClassVar, cast

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from services.database import Database
from services.file_mapper_service import FileMapperService
from tools.search.fetcher import fetch_ddg_fallback, fetch_providers
from contracts.params.param_bag import ParamBag
from contracts.params.search_params_bag import SearchParamsBag
from exceptions import RateLimitException

logger = logging.getLogger(__name__)

# DDG is a universal web fallback, not a row in the provider registry.
_DDG = "ddg"

# Per-provider fetch depth feeding the ranker.
_PER_PROVIDER = 5

# Global cap: keep only the top-N most query-relevant hits after ranking.
_MAX_RESULTS = 5


class SearchAbility(Ability[SearchParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False  # delegate-exclusive; pinned on WebSearchConfig only

    # Pre-gated by the dispatcher BEFORE run() and BEFORE the policy gate: an
    # action-less tool whose only hard requirement is a non-empty query. The ""
    # key covers action-less tools (see ToolDispatcher._prevalidate); a missing or
    # blank query is rejected as one code=missing-params error naming 'query', so
    # run() never sees a query-less call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    # The typed input contract: the dispatch seam builds the bag via
    # SearchParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SearchParamsBag
    NAME: ClassVar[str] = "search"
    RETURNS_UNTRUSTED_CONTENT: ClassVar[bool] = True  # third-party titles and snippets

    def get_summary(self) -> str:
        return "Search Wikipedia, GitHub, Reddit, arXiv, news, and more with automatic provider routing from a plain language query."

    def get_examples(self) -> list[str]:
        return [
            "what is the current price of EUR compared to USD",
            "find recent papers on transformer architecture improvements",
            "look up Python asyncio on Stack Overflow",
            "search GitHub for open source calendar sync libraries",
            "what are people on Reddit saying about the new iPhone",
            "find the Wikipedia article on the history of the internet",
            "search for news about electric vehicle battery breakthroughs",
            "look up the book 'The Pragmatic Programmer' on Open Library",
        ]

    def get_search_tooltip(self) -> str:
        return "web and knowledge search"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Nudge to fetch a promising result's full page before quoting it."""
        from abilities.web_fetch import WebFetchAbility  # noqa: PLC0415

        return f"For the pages that are aligned with your query, use the `{WebFetchAbility.NAME}(url=…)` tool with that result's url to get the full content of the page before quoting it or stating its claims as fact."

    def get_parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                Keys.query: {
                    "type": "string",
                    "description": "Plain natural language search query.",
                },
            },
            "required": [Keys.query],
        }

    _DB: ClassVar[str] = str(FileMapperService.get_search_providers_db_path())
    _providers: ClassVar[dict[str, object] | None] = None

    def run(self, params: SearchParamsBag) -> ToolResult:
        query = params.query  # pre-gated by ACTION_REQUIRED; bag's require_str rejects blank

        t0 = time.time()
        try:
            results, used, meta = self._search_routed(query)

            fell_back = False
            if not results and _DDG not in used:
                ddg = fetch_ddg_fallback(query, _PER_PROVIDER)
                if ddg:
                    results, fell_back = ddg, True
                    used.append(_DDG)
        except RateLimitException:
            return ToolResult.err(
                "Web search is rate-limited; please retry shortly.",
                code="rate-limited",
                hint="try again in a moment",
            )

        logger.info(
            "[SEARCH] query=%r providers=%s fetched=%d latency_ms=%d",
            query, used, len(results), int((time.time() - t0) * 1000),
        )

        # Normalize to internal items (snippet → summary). Only the fields the
        # renderer emits are carried — provider/source are internal routing
        # detail, not part of the output contract.
        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "summary": r.get("snippet", "") or "",
                "date": r.get("date"),
            }
            for r in results
        ]

        from tools.search.router import rank_results
        items = rank_results(query, items)

        items = items[:_MAX_RESULTS]

        from tools.search.enrich import enrich_missing_summaries
        items = enrich_missing_summaries(items)

        from tools.search.render import render_records
        body = render_records(items)

        if not items:
            if fell_back or meta.get("ddg_supplement"):
                return ToolResult.no_results(fallback=_DDG)
            return ToolResult.no_results()

        if fell_back or meta.get("ddg_supplement"):
            return ToolResult.ok(body, count=len(items), fallback=_DDG)
        return ToolResult.ok(body, count=len(items))

    # ── Provider registry ──────────────────────────────────────────────────────

    @classmethod
    def _load_providers(cls) -> dict[str, object]:
        if cls._providers is not None:
            return cls._providers

        try:
            conn = Database.conn(cls._DB)
            rows = conn.execute("SELECT * FROM providers WHERE enabled = 1").fetchall()
            cls._providers = {row["name"]: dict(row) for row in rows}
        except Exception as e:
            logger.exception("[SEARCH] Failed to load providers: %s", e)
            return {}

        return cls._providers

    def _search_routed(self, query: str) -> "tuple[list[dict[str, object]], list[str], dict[str, object]]":
        try:
            from tools.search.router import route_query, _WEAK_SCORE
            ranked = route_query(query)
            if ranked:
                providers = self._load_providers()
                provider_dicts = [cast("dict[str, object]", providers[p["name"]]) for p in ranked if p["name"] in providers]
                results = fetch_providers(provider_dicts, query, _PER_PROVIDER)
                used = [cast("str", p["name"]) for p in provider_dicts]
                meta: dict[str, object] = {"routing_method": "auto"}

                top_score = cast("float", ranked[0]["score"])
                if top_score < _WEAK_SCORE:
                    ddg_results = fetch_ddg_fallback(query, _PER_PROVIDER)
                    if ddg_results:
                        results = results + ddg_results
                        used.append(_DDG)
                        meta["ddg_supplement"] = True
                        logger.info(
                            "[SEARCH] DDG supplement fired for weak routing score %.4f on query=%r",
                            top_score, query,
                        )

                return results, used, meta
        except RateLimitException:
            raise
        except Exception as e:
            logger.warning("[SEARCH] router error, falling back to DDG: %s", e)

        return fetch_ddg_fallback(query, _PER_PROVIDER), [_DDG], {"routing_method": "fallback"}

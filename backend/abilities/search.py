"""
SearchAbility — Multi-provider search with semantic routing.

Routes queries to best provider(s) via embedding similarity,
or allows explicit provider selection. Falls back to DDG.

Rich-media rendering:
  When ``_rich_media_ordinal`` is present (user channel only), returns a
  structured JSON payload + instruction trailer so the frontend renders an
  article card (shared layout with news).

The search assets (search_tool_providers.sqlite, fetcher, router) stay in
tools/search/ until Phase 4 — this ability imports them from there.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import ClassVar

from abilities._base import Ability
from tools.search.fetcher import fetch_providers, fetch_ddg_fallback

logger = logging.getLogger(__name__)

_RICH_MEDIA_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a search results card; without it, the user sees "
    "only plain text. Write a single paragraph that synthesises the findings "
    "for the user. Example: \"Based on what I found — "
    "<span id='{tag}'>The Rust Foundation has cycled through three executive "
    "directors in four years, each departure hinging on the same question of "
    "how much a non-profit steward should do.</span>\""
    "\n\nThumbnail (optional): if the data above includes an `image_candidates` "
    "array, each entry has `url`, `source_title` (the result it came from), "
    "and `caption` (a short description derived from the image filename or "
    "article metadata). Pick the candidate whose `caption` and `source_title` "
    "together best match the topic of your synthesis. If no candidate is a "
    "clear match, omit `data-image`. To attach a thumbnail, add "
    "`data-image='<that url>'` to the span, e.g. "
    "<span id='{tag}' data-image='https://example.com/pic.jpg'>your synthesis</span>. "
    "Use a URL verbatim from `image_candidates`."
)


class SearchAbility(Ability):
    NAME = "search"
    SUMMARY = "Search Wikipedia, GitHub, Reddit, arXiv, news, and more with automatic provider routing from a plain language query."
    EXAMPLES = [
        "what is the current price of EUR compared to USD",
        "find recent papers on transformer architecture improvements",
        "look up Python asyncio on Stack Overflow",
        "search GitHub for open source calendar sync libraries",
        "what are people on Reddit saying about the new iPhone",
        "find the Wikipedia article on the history of the internet",
        "search for news about electric vehicle battery breakthroughs",
        "look up the book 'The Pragmatic Programmer' on Open Library",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Plain natural language search query.",
            },
            "provider": {
                "type": "string",
                "enum": [
                    "wikipedia", "wikidata", "arxiv", "github", "hackernews", "reddit",
                    "google_news", "stackoverflow", "open_library", "musicbrainz",
                    "itunes", "nominatim", "ddg",
                ],
                "description": "Limit your search to one provider. Only use this if the user asks you to use 1 specific provider.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results per provider (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    TIMEOUT = 180

    _DB: ClassVar[str] = str(
        Path(__file__).resolve().parent.parent / "tools" / "search" / "assets" / "search_tool_providers.sqlite"
    )
    _providers: ClassVar[dict | None] = None

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        query = (params.get("query") or "").strip()
        if not query:
            return {"text": "EMPTY: no query supplied. Tell the user no search was performed."}

        ordinal = params.get("_rich_media_ordinal")
        limit = max(1, min(10, int(params["limit"]) if params.get("limit") is not None else 5))
        forced = (params.get("provider") or "").strip().lower()

        t0 = time.time()

        if forced:
            results, used, meta = self._search_forced(query, forced, limit)
        else:
            results, used, meta = self._search_routed(query, limit)

        meta["fetch_latency_ms"] = int((time.time() - t0) * 1000)

        if not results and "ddg" not in used:
            results = fetch_ddg_fallback(query, limit)
            if results:
                used.append("ddg")
                meta["ddg_fallback"] = True

        logger.info("[SEARCH] query=%r providers=%s count=%d", query, used, len(results))

        if not results:
            return {"text": (
                f'EMPTY: zero results for query "{query}". '
                "Do NOT fabricate results, URLs, titles, or categories. "
                "Tell the user honestly that nothing was found and offer to refine the query."
            )}

        structured = [
            {"title": r.get("title", ""), "desc": r.get("snippet", ""), "url": r.get("url", "")}
            for r in results
        ]

        if ordinal is None:
            return {"text": json.dumps({"results": structured})}

        image_items: list[tuple[str, str]] = []
        og_targets: list[tuple[str, str]] = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            img = r.get("image", "")
            if img:
                image_items.append((img, title))
            elif url:
                og_targets.append((url, title))
        return _serialise_rich(structured, ordinal, image_items, og_targets)

    @classmethod
    def _load_providers(cls) -> dict:
        if cls._providers is not None:
            return cls._providers

        try:
            conn = sqlite3.connect(cls._DB)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM providers WHERE enabled = 1").fetchall()
            conn.close()
            cls._providers = {row["name"]: dict(row) for row in rows}
        except Exception as e:
            logger.error("[SEARCH] Failed to load providers: %s", e)
            return {}

        return cls._providers

    def _search_forced(self, query: str, provider_name: str, limit: int):
        if provider_name == "ddg":
            return fetch_ddg_fallback(query, limit), ["ddg"], {"routing_method": "forced"}

        providers = self._load_providers()
        provider = providers.get(provider_name)
        if not provider:
            return [], [], {"routing_method": "forced", "error": f"Unknown: {provider_name}"}

        return fetch_providers([provider], query, limit), [provider_name], {"routing_method": "forced"}

    def _search_routed(self, query: str, limit: int):
        try:
            from tools.search.router import route_query, _WEAK_SCORE
            ranked = route_query(query)
            if ranked:
                providers = self._load_providers()
                provider_dicts = [providers[p["name"]] for p in ranked if p["name"] in providers]
                results = fetch_providers(provider_dicts, query, limit)
                used = [p["name"] for p in provider_dicts]
                meta: dict = {
                    "routing_method": "auto",
                    "provider_scores": {p["name"]: p["score"] for p in ranked},
                }

                top_score = ranked[0]["score"]
                if top_score < _WEAK_SCORE:
                    ddg_results = fetch_ddg_fallback(query, limit)
                    if ddg_results:
                        results = results + ddg_results
                        used.append("ddg")
                        meta["ddg_supplement"] = True
                        logger.info(
                            "[SEARCH] DDG supplement fired for weak routing score %.4f on query=%r",
                            top_score, query,
                        )

                return results, used, meta
        except Exception as e:
            logger.warning("[SEARCH] router error, falling back to DDG: %s", e)

        return fetch_ddg_fallback(query, limit), ["ddg"], {"routing_method": "fallback"}


def _serialise_rich(
    results: list,
    ordinal: int,
    image_items: list[tuple[str, str]],
    og_targets: list[tuple[str, str]] | None = None,
) -> str:
    tag = f"search_{ordinal}"

    # og_meta maps image_url → {"image_url", "description"} for caption derivation
    og_meta: dict = {}
    if len(image_items) < 3 and og_targets:
        from services.og_image_service import resolve_og_images
        try:
            og_map = resolve_og_images([u for u, _ in og_targets[: 3 - len(image_items)]])
        except Exception as exc:
            logger.warning("[SEARCH] og:image lookup failed: %s", exc)
            og_map = {}
        for article_url, title in og_targets:
            meta = og_map.get(article_url)
            if meta:
                img = meta["image_url"]
                image_items.append((img, title))
                # Key by image URL so image_candidate_service can look it up
                og_meta[img] = meta
            if len(image_items) >= 3:
                break

    payload: dict = {"results": results}
    if image_items:
        try:
            from services.image_candidate_service import build_image_candidates
            candidates = build_image_candidates(image_items, og_meta=og_meta or None)
        except Exception as exc:
            logger.warning("[SEARCH] image candidate shortlist failed: %s", exc)
            candidates = []
        if candidates:
            payload["image_candidates"] = candidates
    data_json = json.dumps(payload)
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return f"{data_json}\n\n{instruction}"

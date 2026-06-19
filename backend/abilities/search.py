"""
SearchAbility — Multi-provider search with semantic routing.

Routes queries to best provider(s) via embedding similarity, or allows explicit
provider selection. Falls back to DuckDuckGo (DDG) — but ONLY when a real
provider was attempted and came back empty, never to silently mask a misspelled
or unknown forced provider.

Provider guardrail (TKT-891):
  The set of real provider names is read from the on-disk registry
  (``search_tool_providers.sqlite``), so the schema enum the model reads and the
  guardrail the runtime enforces can never drift from what actually exists. When
  the model FORCES a provider that is not a real name, ``run()`` returns
  ``code=unknown-provider`` with a ``valid:`` ladder of the real names — it does
  NOT fall through to DDG and present web results as the named engine. A legit
  runtime fallback (the named provider was real and configured but returned
  nothing) still fires, but is surfaced as ``meta fallback=ddg`` so the model
  knows the engine it asked for did not answer.

Result contract (TKT-882):
  ``run()`` returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
  success body is a JSON list of ``{title, url, snippet, source}`` hits (``source``
  is the engine that actually answered) with a ``count`` meta — verbatim,
  machine-readable hits the model quotes without re-prosing. Zero results is a
  success with ``count=0`` and an explicit empty list (the model is told plainly
  that nothing was found), never a blank body. The rich article card travels via
  ``ToolResult(rich=…)``; the dispatcher owns the ordinal + the single span
  instruction and injects the card only when the invoking channel broadcasts to
  the user. This ability never formats a wire envelope.

The search assets (search_tool_providers.sqlite, fetcher, router) stay in
tools/search/ until Phase 4 — this ability imports them from there.
"""

import logging
import sqlite3
import time
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult
from services.file_mapper_service import FileMapperService
from tools.search.fetcher import fetch_providers, fetch_ddg_fallback

logger = logging.getLogger(__name__)

# DDG is a universal web fallback, not a row in the provider registry, so it is a
# valid forced provider in its own right but is added to the known set explicitly.
_DDG = "ddg"


class SearchAbility(Ability):
    # Pre-gated by the dispatcher BEFORE run() and BEFORE the policy gate: an
    # action-less tool whose only hard requirement is a non-empty query. The ""
    # key covers action-less tools (see ToolDispatcher._prevalidate); a missing or
    # blank query is rejected as one code=missing-params error naming 'query', so
    # run() never sees a query-less call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    def get_name(self) -> str:
        return "search"

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

    def get_parameters(self) -> dict[str, object]:
        """The ``provider`` enum is sourced LIVE from the real registry (+ ``ddg``),
        so the schema can never advertise a provider that the guardrail then rejects.
        """
        return {
            "type": "object",
            "properties": {
                Keys.query: {
                    "type": "string",
                    "description": "Plain natural language search query.",
                },
                Keys.provider: {
                    "type": "string",
                    "enum": sorted(self._known_providers()),
                    "description": (
                        "Limit your search to one provider. Only use this if the "
                        "user asks you to use 1 specific provider. Must be one of "
                        "the listed names exactly — an unknown name is rejected, "
                        "not silently web-searched."
                    ),
                },
                Keys.limit: {
                    "type": "integer",
                    "description": "Max results per provider (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": [Keys.query],
        }

    _DB: ClassVar[str] = str(FileMapperService.get_search_providers_db_path())
    _providers: ClassVar[dict[str, dict[str, object]] | None] = None

    def run(self, params: dict[str, object]) -> ToolResult:
        # query presence is pre-gated by ACTION_REQUIRED; .strip() guards a
        # whitespace-only value that the truthiness pre-gate lets through.
        query = cast(str, params.get(Keys.query) or "").strip()
        if not query:
            return ToolResult.err(
                "query is required and cannot be blank.",
                code="missing-params",
                hint="pass a plain natural-language query.",
                valid=("query",),
            )

        limit = cast(int, self.param(params, Keys.limit, default=5, clamp=(1, 10)))
        forced = cast(str, params.get(Keys.provider) or "").strip().lower()

        if forced:
            guard = self._reject_unknown_forced(forced)
            if guard is not None:
                return guard

        t0 = time.time()
        if forced:
            results, used, meta = self._search_forced(query, forced, limit)
        else:
            results, used, meta = self._search_routed(query, limit)

        # Legit runtime fallback: a real, configured provider was attempted but
        # came back empty. Surface it as meta fallback=ddg so the model knows the
        # engine it asked for did not answer (NOT silent masking).
        fell_back = False
        if not results and _DDG not in used:
            ddg = fetch_ddg_fallback(query, limit)
            if ddg:
                results = ddg
                used.append(_DDG)
                fell_back = True

        logger.info(
            "[SEARCH] query=%r providers=%s count=%d latency_ms=%d",
            query, used, len(results), int((time.time() - t0) * 1000),
        )

        structured = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "source": r.get("provider", "") or _DDG,
            }
            for r in results
        ]

        result_meta: dict[str, object] = {"count": len(structured)}
        if fell_back:
            result_meta["fallback"] = _DDG
        elif meta.get("ddg_supplement"):
            # Weak-routing supplement mixes DDG into a real result set — name it.
            result_meta["fallback"] = _DDG

        # --- Rich media card temporarily disabled (web_search latency work) ---
        # _build_rich_card resolves og:image URLs with synchronous network calls
        # on every search. Inside the web_search delegate (which never broadcasts
        # the card to the user) that is pure latency tax, and even on the chat
        # channel it serialises the turn behind 1-3 extra HTTP fetches. Left in
        # place but not invoked; to be re-wired properly with a design spec.
        # See follow-up ticket: "Re-wire rich media card for web_search results".
        # rich = self._build_rich_card(results) if structured else None
        rich = None
        return ToolResult.ok(structured, rich=rich, **result_meta)

    # ── Provider registry ──────────────────────────────────────────────────────

    @classmethod
    def _load_providers(cls) -> dict[str, dict[str, object]]:
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

    @classmethod
    def _known_providers(cls) -> set[str]:
        return set(cls._load_providers().keys()) | {_DDG}

    def _reject_unknown_forced(self, provider_name: str) -> ToolResult | None:
        """Returns an error when *provider_name* is unknown (never a real registry
        row and not ``ddg``) — the named engine must NOT be silently web-searched.
        Returns None when the name is known (search proceeds, with a legit runtime
        DDG fallback if empty).
        """
        known = self._known_providers()
        if provider_name in known:
            return None
        return ToolResult.err(
            f"Unknown search provider {provider_name!r}.",
            code="unknown-provider",
            hint="name one of the valid providers below exactly, or omit provider to auto-route.",
            valid=tuple(sorted(known)),
        )

    def _search_forced(
        self, query: str, provider_name: str, limit: int
    ) -> "tuple[list[dict[str, object]], list[str], dict[str, object]]":
        if provider_name == _DDG:
            return fetch_ddg_fallback(query, limit), [_DDG], {"routing_method": "forced"}

        # Known but possibly disabled at runtime: _reject_unknown_forced already
        # cleared genuinely-unknown names, so a miss here means the row was
        # disabled after the guardrail's snapshot — fetch nothing and let the
        # legit DDG fallback fire.
        providers = self._load_providers()
        provider = providers.get(provider_name)
        if not provider:
            return [], [], {"routing_method": "forced"}

        return fetch_providers([provider], query, limit), [provider_name], {"routing_method": "forced"}

    def _search_routed(
        self, query: str, limit: int
    ) -> "tuple[list[dict[str, object]], list[str], dict[str, object]]":
        try:
            from tools.search.router import route_query, _WEAK_SCORE
            ranked = route_query(query)
            if ranked:
                providers = self._load_providers()
                provider_dicts = [providers[p["name"]] for p in ranked if p["name"] in providers]
                results = fetch_providers(provider_dicts, query, limit)
                used = [cast(str, p["name"]) for p in provider_dicts]
                meta: dict[str, object] = {"routing_method": "auto"}

                top_score = cast(float, ranked[0]["score"])
                if top_score < _WEAK_SCORE:
                    ddg_results = fetch_ddg_fallback(query, limit)
                    if ddg_results:
                        results = results + ddg_results
                        used.append(_DDG)
                        meta["ddg_supplement"] = True
                        logger.info(
                            "[SEARCH] DDG supplement fired for weak routing score %.4f on query=%r",
                            top_score, query,
                        )

                return results, used, meta
        except Exception as e:
            logger.warning("[SEARCH] router error, falling back to DDG: %s", e)

        return fetch_ddg_fallback(query, limit), [_DDG], {"routing_method": "fallback"}

    # ── Rich article card (via ToolResult.rich) ─────────────────────────────────

    @staticmethod
    def _build_rich_card(results: list[object]) -> dict[str, object]:
        """Build the rich article-card payload: the structured hits plus up to three
        image candidates the frontend can render as a thumbnail.

        Returned as the ``rich`` payload of the success ToolResult; the dispatcher
        serialises it as the card JSON head and appends the single span
        instruction (and the ordinal-keyed tag) ONLY when the invoking channel
        broadcasts to the user. This ability never formats the envelope.
        """
        structured = [
            {
                "title": cast("dict[str, object]", r).get("title", ""),
                "url": cast("dict[str, object]", r).get("url", ""),
                "snippet": cast("dict[str, object]", r).get("snippet", ""),
                "source": cast("dict[str, object]", r).get("provider", "") or _DDG,
            }
            for r in results
        ]
        payload: dict[str, object] = {"results": structured}

        image_items: list[tuple[str, str]] = []
        og_targets: list[tuple[str, str]] = []
        for r in results:
            title = cast(str, cast("dict[str, object]", r).get("title", ""))
            url = cast(str, cast("dict[str, object]", r).get("url", ""))
            img = cast(str, cast("dict[str, object]", r).get("image", ""))
            if img:
                image_items.append((img, title))
            elif url:
                og_targets.append((url, title))

        og_meta: dict[str, object] = {}
        if len(image_items) < 3 and og_targets:
            from services.og_image_service import resolve_og_images
            try:
                og_map = resolve_og_images([u for u, _ in og_targets[: 3 - len(image_items)]])
            except Exception as exc:
                logger.warning("[SEARCH] og:image lookup failed: %s", exc)
                og_map = {}
            for article_url, title in og_targets:
                resolved = og_map.get(article_url)
                if resolved:
                    img = resolved["image_url"]
                    image_items.append((img, title))
                    og_meta[img] = resolved
                if len(image_items) >= 3:
                    break

        if image_items:
            from services.image_candidate_service import build_image_candidates
            try:
                candidates = build_image_candidates(image_items, og_meta=og_meta or None)
            except Exception as exc:
                logger.warning("[SEARCH] image candidate shortlist failed: %s", exc)
                candidates = []
            if candidates:
                payload["image_candidates"] = candidates

        return payload

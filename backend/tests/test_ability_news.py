# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import hashlib
import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.news import NewsAbility
from configs.channels import DmnConfig, UserConfig
from services import news_service
from services.act_trail import ActTrail
from services.memory_client import MemoryClientService
from services.news_service import NewsArticle, NewsService
from tests._tool_result_harness import MP, body as _harness_body, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def user_mp(db):
    return MP(seed_transcript(db, "chat", "what's in the news"), UserConfig({}))


@pytest.fixture
def dmn_mp(db):
    return MP(seed_transcript(db, "subconscious", "what's in the news"), DmnConfig())



@pytest.fixture(autouse=True)
def _reset_news_service():
    NewsAbility._service = None
    yield
    NewsAbility._service = None


def _body(rendered: str, tool: str = "news") -> str:
    return _harness_body(rendered, tool)


def _cache_key(query: str, country_code: str = "US") -> str:
    digest = hashlib.sha256((query + country_code).encode()).hexdigest()[:16]
    return f"news:google:{digest}"


def _seed_google_cache(query: str, articles: list, country_code: str = "US") -> None:
    """Seed the REAL shared store at the production cache key so a dispatch of
    ``news`` for *query* returns these articles through the genuine cache-hit branch."""
    store = MemoryClientService.create_connection()
    store.setex(
        _cache_key(query, country_code),
        600,
        json.dumps([a.to_dict() for a in articles]),
    )


# ── Invalid category: rejected by the param helper, never silently degraded ────


def test_invalid_category_rejected_with_valid_enum(db, user_mp):
    """invalid category must be rejected as ``code=invalid-param`` — not silently
    fall back to an uncategorised search."""
    out = ToolDispatcher(user_mp).dispatch(
        "news",
        {"query": "anything", "category": "politics", "act_summary": "x"},
    )

    assert "[news(status=error" in out, out
    assert "code=invalid-param" in out, out
    assert "code=error]" not in out, out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    advertised = {p.strip() for p in valid_line[len("valid:"):].split("|")}
    assert advertised == {"tech", "business", "sports", "science", "entertainment", "us", "uk"}, advertised


# ── Provider unreachable: loud error + hint, NEVER an empty success ────────────


def test_dead_provider_is_provider_unreachable_not_empty_success(db, user_mp, monkeypatch):
    """dead provider must surface ``code=provider-unreachable`` with a recovery Hint — NEVER an empty success."""
    monkeypatch.setattr(news_service, "GOOGLE_NEWS_BASE", "http://127.0.0.1:9/rss/search")

    out = ToolDispatcher(user_mp).dispatch(
        "news",
        {"query": "a query with no cache entry so the real fetch fires", "act_summary": "x"},
    )

    assert "[news(status=error" in out, out
    assert "code=provider-unreachable" in out, out
    assert "code=error]" not in out, out
    assert "hint:" in out, out
    # NOT an empty success: the count meta / success status must be absent.
    assert "status=success" not in out, out
    body = _body(out)
    assert body.strip() not in ("[]", ""), body

    trail = ActTrail().fetch_by_transcript_id(user_mp.uid)
    assert any("code=provider-unreachable" in row["result"] for row in trail), trail


# ── Happy path (offline, real cache-hit branch): structured rows + count ───────


def _sample_articles() -> list:
    return [
        NewsArticle(
            title="EU opens first audits under the AI Act",
            description="The EU regulator opened formal audits into three large AI providers today, a milestone for the AI Act enforcement regime that companies have been tracking closely. " * 2,
            url="https://example.com/eu-ai-act-audits",
            published_at="2026-06-10T08:00:00+00:00",
            source="Reuters",
            source_id="google_news",
            category="international",
            image_url="https://example.com/img/eu.jpg",
        ),
        NewsArticle(
            title="Chipmaker unveils 2nm process node",
            description="A leading foundry announced volume production of its 2nm node.",
            url="https://example.com/2nm-node",
            published_at="2026-06-10T07:30:00+00:00",
            source="The Verge",
            source_id="google_news",
            category="international",
            image_url="",
        ),
    ]


def test_happy_path_structured_rows_offline_via_cache(db, dmn_mp):
    """real cache-hit returns rich ``count=N`` envelope with rows of {title, source, url, published_at, snippet}."""
    query = "tkt904 happy path query"
    _seed_google_cache(query, _sample_articles())

    out = ToolDispatcher(dmn_mp).dispatch(
        "news", {"query": query, "act_summary": "x"}
    )

    assert "[news(status=success" in out, out
    assert "count=2" in out, out
    # Non-broadcast turn → no rich trailer.
    body = _body(out)
    assert "\n\n" not in body, body
    assert "span id=" not in out and "You MUST present" not in out, out

    rows = json.loads(body)
    assert isinstance(rows, list) and len(rows) == 2, rows
    for row in rows:
        assert set(row.keys()) == {"title", "source", "url", "published_at", "snippet"}, row
    first = rows[0]
    assert first["title"] == "EU opens first audits under the AI Act"
    assert first["source"] == "Reuters"
    assert first["url"] == "https://example.com/eu-ai-act-audits"
    assert first["published_at"] == "2026-06-10T08:00:00+00:00"
    # snippet is the description truncated to ~200 chars.
    assert len(first["snippet"]) <= 200, len(first["snippet"])


# ── Zero-articles success: a provider that ANSWERED with nothing ───────────────


def test_zero_articles_is_success_with_empty_list(db, dmn_mp, monkeypatch):
    """answered provider returning zero articles is SUCCESS with ``count=0`` and an empty list — never an error."""

    def _empty_answer(self, query, country_code="US", timeout=5.0):
        return []

    monkeypatch.setattr(NewsService, "fetch_google_news", _empty_answer)
    NewsAbility._service = None  # rebind so the patched method is used

    out = ToolDispatcher(dmn_mp).dispatch(
        "news", {"query": "nothing matches this query", "act_summary": "x"}
    )

    assert "[news(status=success" in out, out
    assert "count=0" in out, out
    assert "code=" not in out.splitlines()[0], out
    body = _body(out)
    assert json.loads(body) == [], body

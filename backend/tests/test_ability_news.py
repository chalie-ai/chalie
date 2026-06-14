# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""news-specific business-logic tests migrated from the per-ability conformance
file removed in TKT-975. Drives the real ToolDispatcher end-to-end hot path with
zero mocks, exercising category validation, provider-unreachable errors, offline
cache-seeded happy paths, rich card rendering, and zero-article success."""

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
    """A real user-broadcast mp (``broadcast_to == 'user'``). On this channel the
    dispatcher assigns a rich-media ordinal and renders the news card trailer."""
    return MP(seed_transcript(db, "chat", "what's in the news"), UserConfig({}))


@pytest.fixture
def dmn_mp(db):
    """A real non-user-broadcast mp (``broadcast_to is None``). On this channel the
    dispatcher drops the rich card, so the rendered body is the raw model-facing
    rows with no span/instruction trailer."""
    return MP(seed_transcript(db, "subconscious", "what's in the news"), DmnConfig())


@pytest.fixture(autouse=True)
def _reset_news_service():
    """The ability caches a single ``NewsService`` on the class. Reset it around
    each test so a base-URL override in one test never leaks into another, and so
    every test exercises a fresh service against the real store."""
    NewsAbility._service = None
    yield
    NewsAbility._service = None


def _body(rendered: str, tool: str = "news") -> str:
    """Return the text between the open tag and ``[end:<tool>]`` (verbatim, no
    rich-card split — the news tests own the ``partition`` themselves)."""
    return _harness_body(rendered, tool)


def _cache_key(query: str, country_code: str = "US") -> str:
    """The exact cache key the production ``fetch_google_news`` computes."""
    digest = hashlib.sha256((query + country_code).encode()).hexdigest()[:16]
    return f"news:google:{digest}"


def _seed_google_cache(query: str, articles: list, country_code: str = "US") -> None:
    """Seed the REAL shared store at the production Google-News cache key so a
    dispatch of ``news`` for *query* returns these articles through the genuine
    cache-hit branch — zero network, zero mock."""
    store = MemoryClientService.create_connection()
    store.setex(
        _cache_key(query, country_code),
        600,
        json.dumps([a.to_dict() for a in articles]),
    )


# ── Invalid category: rejected by the param helper, never silently degraded ────


def test_invalid_category_rejected_with_valid_enum(db, user_mp):
    """A bogus ``category`` (passes the query pre-gate, then reaches run()) is
    rejected by the base ``param`` helper as ``code=invalid-param`` with a
    ``valid:`` ladder of the real enum — it does NOT silently fall back to an
    uncategorised search."""
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
    """Pointing the REAL Google News base URL at a closed loopback port makes the
    genuine ``requests.get`` raise a real ``ConnectionError`` the service wraps
    into ``NewsFetchError``; the ability surfaces ``code=provider-unreachable``
    with a recovery hint. The exact "Bad" the ticket closes — a dead provider
    must NOT return an empty success masquerading as "no news"."""
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
    """Seeding the REAL shared store at the production cache key makes
    ``fetch_google_news`` return real articles through its genuine cache-hit
    branch — no network. The success envelope carries ``count=N`` and the body is
    a JSON list whose rows carry EXACTLY {title, source, url, published_at,
    snippet} so the model quotes real headlines verbatim. (dmn channel → no card
    trailer.)"""
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


def test_happy_path_user_broadcast_renders_rich_card(db, user_mp):
    """On a user-broadcast turn the dispatcher pairs the card: the body becomes the
    rich payload JSON + blank line + the span instruction (``<span id='news_1'>``).
    The card payload rows carry {title, url, snippet, source} (card shape, no
    published_at) so the FE article card renders domain pills from ``url``/``source``."""
    query = "tkt904 broadcast query"
    _seed_google_cache(query, _sample_articles())

    out = ToolDispatcher(user_mp).dispatch(
        "news", {"query": query, "act_summary": "x"}
    )

    assert "[news(status=success" in out, out
    assert "count=2" in out, out
    assert "<span id='news_1'>" in out, out

    body = _body(out)
    payload_json, _, instruction = body.partition("\n\n")
    assert "news_1" in instruction, instruction
    payload = json.loads(payload_json)
    assert isinstance(payload["results"], list) and payload["results"], payload
    for row in payload["results"]:
        assert set(row.keys()) == {"title", "url", "snippet", "source"}, row
    assert payload["results"][0]["url"] == "https://example.com/eu-ai-act-audits"
    assert payload["results"][0]["source"] == "Reuters"


# ── Zero-articles success: a provider that ANSWERED with nothing ───────────────


def test_zero_articles_is_success_with_empty_list(db, dmn_mp, monkeypatch):
    """A provider that ANSWERS but yields no usable articles is a SUCCESS with
    ``count=0`` and an explicit empty list — never an error, never a blank body.
    We force the REAL ``fetch_google_news`` to answer empty by replacing it with a
    real method that returns ``[]`` (the genuine "answered, but nothing matched"
    shape — distinct from the unreachable path which RAISES). No network."""

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

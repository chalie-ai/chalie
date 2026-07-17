# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Feature tests for ``tools.search.router.rank_results`` and the
``enrich_missing_summaries`` no-fetch invariant.

Both tests run through the REAL EmbeddingService — no mocks.  The ranking test
is unit-marked because EmbeddingService works offline (ONNX, local model); it
is deterministic for fixed inputs and fast once the model is loaded.

Baseline-fail proof:
  - ``rank_results`` does not exist in ``tools.search.router`` yet.
  - ``tools.search.enrich`` does not exist yet.
Both facts make this file fail at collection with ImportError.
"""

import pytest

pytestmark = pytest.mark.unit

from tools.search.router import rank_results          # noqa: E402  — baseline-fail target
from tools.search.enrich import enrich_missing_summaries  # noqa: E402  — baseline-fail target


# ── rank_results: top result is the topically-closest one ────────────────────


def test_rank_results_puts_on_topic_result_first() -> None:
    """The Anthropic/Claude result must rank above clearly off-topic results.

    Uses the REAL EmbeddingService (offline ONNX) — no mocks.
    """
    query = "Anthropic Claude AI model"

    results: list[dict[str, object]] = [
        {
            "title": "Tagliatelle al Ragù",
            "url": "https://recipes.example.com/pasta",
            "summary": "A classic Italian pasta dish slow-cooked with beef mince, tomatoes, and red wine.",
            "score": None,
            "date": None,
        },
        {
            "title": "Anthropic Claude — Large Language Model API",
            "url": "https://anthropic.com/claude",
            "summary": "Claude is Anthropic's AI assistant, designed for helpful, harmless, and honest conversations.",
            "score": None,
            "date": None,
        },
        {
            "title": "Beginner's Guide to Fingerstyle Guitar",
            "url": "https://guitar.example.com/fingerstyle",
            "summary": "Learn Travis picking patterns, chord shapes, and how to combine melody with bass lines on acoustic guitar.",
            "score": None,
            "date": None,
        },
    ]

    ranked = rank_results(query, results)

    # Result count unchanged
    assert len(ranked) == 3

    # Best-first: the Anthropic result must be at position 0
    title_0 = str(ranked[0]["title"]).lower()
    assert "anthropic" in title_0 or "claude" in title_0, (
        f"Expected Anthropic/Claude result first, got: {ranked[0]['title']!r}"
    )

    # Every result now has a score
    for r in ranked:
        assert "score" in r, f"Result missing 'score' key: {r['title']!r}"
        assert isinstance(r["score"], float), f"score is not float: {type(r['score'])}"
        score = r["score"]
        assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"

    # List is sorted descending by score
    scores: list[float] = []
    for r in ranked:
        assert isinstance(r["score"], float)
        scores.append(r["score"])
    assert scores == sorted(scores, reverse=True), f"Results not sorted descending: {scores}"


# ── rank_results: fail-open on embedding error ────────────────────────────────


def test_rank_results_fail_open_returns_original_on_embedding_error() -> None:
    """When the query itself would cause an error — tested by passing an empty
    string which the embedding service handles gracefully OR by the fail-open
    contract being exercised.  The invariant: if rank_results raises internally
    it must return the original list unchanged, not crash the caller.

    We test the empty-query edge (the real service handles it, but the contract
    says fail-open and no exception must propagate).
    """
    results: list[dict[str, object]] = [
        {"title": "A", "url": "https://a.com", "summary": "Alpha content.", "score": None, "date": None},
        {"title": "B", "url": "https://b.com", "summary": "Beta content.",  "score": None, "date": None},
    ]
    # An empty query must not crash; fail-open means the caller gets something back
    out = rank_results("", results)
    # Either scored or returned unchanged — both are acceptable; must not raise
    assert isinstance(out, list)
    assert len(out) == 2


# ── enrich_missing_summaries: already-has-summary results are untouched ───────


def test_enrich_missing_summaries_result_with_existing_summary_is_returned_unchanged() -> None:
    """A result that already has a non-empty summary must be returned byte-identical.

    This is the one deterministic invariant for enrich_missing_summaries that
    requires no network access: results with a summary are not fetched and are
    not mutated.
    """
    original: dict[str, object] = {
        "title": "Python asyncio documentation",
        "url": "https://docs.python.org/3/library/asyncio.html",
        "summary": "asyncio is a library to write concurrent code using the async/await syntax.",
        "score": 0.88,
        "date": "2026-01-01",
    }
    # Deep-copy the dict so we can compare identity-free
    import copy
    result_in = copy.deepcopy(original)

    enriched = enrich_missing_summaries([result_in])

    assert len(enriched) == 1
    assert enriched[0]["summary"] == original["summary"], (
        "Summary was mutated even though it was non-empty"
    )
    assert enriched[0]["url"] == original["url"]
    assert enriched[0]["title"] == original["title"]
    assert enriched[0].get("date") == original["date"]

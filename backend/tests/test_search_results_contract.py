# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Feature tests for the search RESULTS contract (post ruling: all embedding/
ranking of search results is killed; embeddings are used ONLY to route
providers; summary enrichment from page fetches is deleted — a blank
summary simply renders as an empty string).

Covers:
  - ``tools.search.transformers.cap_result_fields`` — title/summary clipping
    with the ellipsis-in-cap rule, and byte-identical pass-through for text
    at or under the cap;
  - the pipeline constants: ``abilities.search._PER_PROVIDER == 10``,
    ``abilities.search._MAX_RESULTS == 20``, ``tools.search.router._MAX == 2``
    (top-2 providers selected, 10 fetched per provider, up to 20 merged
    results returned in provider-merge order).

All tests run against the real production code — no mocks, no network, fully
deterministic.
"""

import pytest

pytestmark = pytest.mark.unit

from abilities.search import _MAX_RESULTS, _PER_PROVIDER  # noqa: E402
from tools.search import router  # noqa: E402
from tools.search.transformers import cap_result_fields  # noqa: E402


def _item(title: str, summary: str) -> dict[str, object]:
    return {
        "title": title,
        "url": "https://example.com/x",
        "summary": summary,
        "date": None,
    }


# ── cap_result_fields: over-cap fields are clipped with an ellipsis ──────────


def test_cap_result_fields_clips_long_title_and_summary_to_the_caps() -> None:
    """A 250-char title and a 400-char summary must come out exactly at the
    caps, each ending with "…" — and the ellipsis counts toward the cap."""
    out = cap_result_fields([_item("t" * 250, "s" * 400)])

    title = str(out[0]["title"])
    summary = str(out[0]["summary"])
    assert len(title) == 200, f"expected 200 chars, got {len(title)}"
    assert title.endswith("…")
    assert len(summary) == 300, f"expected 300 chars, got {len(summary)}"
    assert summary.endswith("…")

    # Non-text fields and item order are carried over untouched
    assert out[0]["url"] == "https://example.com/x"
    assert out[0]["date"] is None


# ── cap_result_fields: under-cap text passes through byte-identical ──────────


def test_cap_result_fields_under_cap_text_is_byte_identical_without_ellipsis() -> None:
    out = cap_result_fields([_item("a" * 199, "b" * 299)])

    assert out[0]["title"] == "a" * 199
    assert out[0]["summary"] == "b" * 299
    assert "…" not in str(out[0]["title"])
    assert "…" not in str(out[0]["summary"])


# ── cap_result_fields: exactly-at-cap text is untouched, NO ellipsis ─────────


def test_cap_result_fields_exactly_at_cap_is_untouched_without_ellipsis() -> None:
    """200-char titles and 300-char summaries are exactly at the caps: they
    must pass through unchanged and must NOT gain an ellipsis (that would
    push them to 201/301)."""
    out = cap_result_fields([_item("x" * 200, "y" * 300)])

    assert out[0]["title"] == "x" * 200
    assert out[0]["summary"] == "y" * 300
    assert "…" not in str(out[0]["title"])
    assert "…" not in str(out[0]["summary"])


# ── Pipeline constants: lock the ruling ──────────────────────────────────────


def test_pipeline_constants_match_the_provider_selection_ruling() -> None:
    """The ruling: top-2 providers selected, 10 results fetched per provider,
    up to 20 merged results returned in provider-merge order (no ranking)."""
    assert _PER_PROVIDER == 10
    assert _MAX_RESULTS == 20
    assert router._MAX == 2

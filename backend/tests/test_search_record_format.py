# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Feature tests for ``tools.search.render.render_records``.

This is the ONE permitted deterministic pure-function test in the search
redesign suite: ``render_records`` has no collaborators, no IO, and no state.
All other search tests go through the real production entry point.

Baseline-fail proof: ``tools/search/render.py`` does not exist yet.
Every test in this file will error with ``ModuleNotFoundError`` until the
coder creates the file with the correct implementation.
"""

import pytest

pytestmark = pytest.mark.unit

# Import of the not-yet-existing module is intentionally at module scope so
# that pytest reports a collection error (baseline-fail) on every test here.
from tools.search.render import render_records  # noqa: E402


# ── Multi-result indexing 1..N ────────────────────────────────────────────────


def test_render_multi_result_indexes_are_1_based_and_sequential():
    results = [
        {"title": "Alpha", "url": "https://a.example.com", "summary": "First.", "score": 0.9, "date": "2026-01-01"},
        {"title": "Beta",  "url": "https://b.example.com", "summary": "Second.", "score": 0.8, "date": None},
        {"title": "Gamma", "url": "https://c.example.com", "summary": "Third.", "score": None, "date": None},
    ]
    rendered = render_records(results)

    # 1-based indexing
    assert 'index="1"' in rendered
    assert 'index="2"' in rendered
    assert 'index="3"' in rendered
    # No zero-based or N+1 index
    assert 'index="0"' not in rendered
    assert 'index="4"' not in rendered


# ── Best-first order is preserved as-supplied ─────────────────────────────────


def test_render_preserves_list_order_first_result_is_index_1():
    results = [
        {"title": "Winner",   "url": "https://win.example.com", "summary": "Top result.", "score": 0.95, "date": None},
        {"title": "Runner-up", "url": "https://run.example.com", "summary": "Second.",    "score": 0.70, "date": None},
    ]
    rendered = render_records(results)

    idx_winner   = rendered.index("Winner")
    idx_runnerup = rendered.index("Runner-up")

    # The first supplied result must appear first in output
    assert idx_winner < idx_runnerup

    lines = rendered.splitlines()
    first_open = next(line for line in lines if line.startswith("<result "))
    assert 'index="1"' in first_open
    assert "Winner" in rendered.split('index="1"')[1].split("</result>")[0]


# ── Score rendered to 2 decimal places ───────────────────────────────────────


def test_render_score_attribute_is_2_decimal_places():
    results = [
        {"title": "A", "url": "https://a.com", "summary": "Summary.", "score": 0.87654, "date": None},
        {"title": "B", "url": "https://b.com", "summary": "Summary.", "score": 1.0,     "date": None},
        {"title": "C", "url": "https://c.com", "summary": "Summary.", "score": 0.5,     "date": None},
    ]
    rendered = render_records(results)

    assert 'score="0.88"' in rendered
    assert 'score="1.00"' in rendered
    assert 'score="0.50"' in rendered

    # Raw unformatted value must not appear
    assert "0.87654" not in rendered


# ── score= attribute omitted when score is None ───────────────────────────────


def test_render_score_attribute_omitted_when_score_is_none():
    results = [
        {"title": "No score", "url": "https://example.com", "summary": "Body.", "score": None, "date": None},
    ]
    rendered = render_records(results)

    assert "<result " in rendered
    assert "score=" not in rendered


# ── date= attribute omitted when date is None or empty ───────────────────────


def test_render_date_attribute_omitted_when_date_is_none():
    results = [
        {"title": "No date", "url": "https://example.com", "summary": "Body.", "score": 0.5, "date": None},
    ]
    rendered = render_records(results)

    assert "date=" not in rendered


def test_render_date_attribute_omitted_when_date_is_empty_string():
    results = [
        {"title": "No date", "url": "https://example.com", "summary": "Body.", "score": 0.5, "date": ""},
    ]
    rendered = render_records(results)

    assert "date=" not in rendered


# ── date= attribute present when date is set ─────────────────────────────────


def test_render_date_attribute_present_when_date_is_set():
    results = [
        {"title": "Dated", "url": "https://example.com", "summary": "Body.", "score": 0.7, "date": "2026-06-17"},
    ]
    rendered = render_records(results)

    assert 'date="2026-06-17"' in rendered


# ── Long summary is NOT truncated ─────────────────────────────────────────────


def test_render_very_long_summary_survives_in_full():
    long_summary = "x" * 2000
    results = [
        {"title": "Long", "url": "https://example.com", "summary": long_summary, "score": 0.8, "date": None},
    ]
    rendered = render_records(results)

    # The full summary string must be present byte-for-byte
    assert long_summary in rendered

    # No truncation sentinel (ellipsis)
    assert "..." not in rendered


# ── Empty list produces a sentinel without <result index= ────────────────────


def test_render_empty_list_returns_sentinel_without_result_tag():
    rendered = render_records([])

    assert "<result index=" not in rendered
    # There must be some non-empty content
    assert rendered.strip() != ""


# ── Free-text content cannot forge a record boundary ─────────────────────────


def test_render_neutralizes_result_tokens_in_content():
    """A title/summary containing a literal ``</result>`` or ``<result …>`` must
    not forge a record boundary: a single-result render has exactly ONE genuine
    opening tag and ONE genuine closing tag."""
    results = [
        {
            "title": "Parsing <result index=2> blocks",
            "url": "https://example.com/xml",
            "summary": "To close a record emit </result>; a new one opens with <result index=...>.",
            "score": 0.9,
            "date": None,
        },
    ]
    rendered = render_records(results)

    # The content's tokens were defanged, leaving only the real delimiters.
    assert rendered.count("</result>") == 1
    assert rendered.count("<result index=") == 1


def test_render_preserves_non_result_angle_brackets():
    """Generic ``<``/``>`` in content (code, math) survive untouched — only the
    record-boundary tokens are escaped."""
    results = [
        {"title": "Generics in C++", "url": "https://example.com",
         "summary": "Use std::vector<int> and assert a < b.", "score": 0.5, "date": None},
    ]
    rendered = render_records(results)

    assert "std::vector<int>" in rendered
    assert "a < b" in rendered

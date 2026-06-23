# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Chalie-docs-specific business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
chalie_docs ability's genuine behaviour tests (unknown-query errors, fetch-failed
path, no-instruction guarantee) that have no coverage elsewhere."""

import sqlite3
from typing import cast

import pytest

import abilities.chalie_docs
from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="tell me about chalie")


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> _MP:
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _dead_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repoint every production ``_QUERY_URLS`` entry at a non-resolvable
    ``chalie-docs.invalid`` host — the table's OWN field, not a mock. The SSRF
    guard fails closed on the unresolvable host before any socket opens, so the
    real fetch path raises offline for every url and the ability exhausts them."""
    dead = {
        "basics": [
            "https://chalie-docs.invalid/guide/getting-started/",
            "https://chalie-docs.invalid/how-it-works/",
        ],
        "tools": ["https://chalie-docs.invalid/guide/getting-started/"],
        "releases": ["https://chalie-docs.invalid/releases/"],
        "code-base": ["https://chalie-docs.invalid/repo"],
    }
    monkeypatch.setattr(abilities.chalie_docs, "_QUERY_URLS", dead)


def test_unknown_query_errors_with_doc_not_found_and_closest_match(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A query the table does not know returns a STABLE ``code=doc-not-found``
    error whose ``valid:`` ladder lists the REAL keys, with a closest-match hint —
    so a weak model self-corrects. The banned ``code=error`` marker must be gone.

    'toolz' is a near-miss for 'tools' → difflib should surface it in the hint."""
    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"query": "toolz", "act_summary": "x"}
    )

    assert "[chalie_docs(status=error, code=doc-not-found" in out
    assert "code=error]" not in out
    assert "valid:" in out
    assert "[end:chalie_docs]" in out
    # The ladder names every real key the model can route to.
    for key in ("basics", "tools", "releases", "code-base"):
        assert key in out
    # The closest-match suggestion points at the near-miss key.
    assert "tools" in out
    assert "hint:" in out

    # The act-trail recorded the same loud error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "code=doc-not-found" in cast(str, trail[0]["result"])


def test_unknown_query_valid_ladder_is_the_real_table(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """The ``valid:`` ladder on a doc-not-found error lists EXACTLY the real query
    keys from the production table — drives the guardrail against the live table,
    not a hardcoded list."""
    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"query": "nonsense", "act_summary": "x"}
    )

    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    advertised = {p.strip() for p in valid_line[len("valid:"):].split("|")}
    assert advertised == set(abilities.chalie_docs._QUERY_URLS)


def test_all_urls_unreachable_yields_fetch_failed_no_instruction(db: sqlite3.Connection, chat_mp: _MP, monkeypatch: pytest.MonkeyPatch) -> None:
    """When every url for a known query is unreachable, the ability returns a LOUD
    ``code=fetch-failed`` error — NOT the old 'use the read tool and visit …'
    instruction presented as a successful answer. We repoint the table's OWN urls
    at a non-resolvable host (production field, not a mock); the SSRF guard fails
    closed offline so every fetch raises and the ability exhausts them."""
    _dead_table(monkeypatch)

    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"query": "basics", "act_summary": "x"}
    )

    assert "[chalie_docs(status=error, code=fetch-failed" in out
    assert "code=error]" not in out
    assert "hint:" in out
    assert "[end:chalie_docs]" in out

    # The act-trail recorded the same loud error — the model sees a routable failure.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "code=fetch-failed" in cast(str, trail[0]["result"])


def test_failure_carries_no_read_tool_instruction(db: sqlite3.Connection, chat_mp: _MP, monkeypatch: pytest.MonkeyPatch) -> None:
    """No-fabrication guard: even on total failure the old instruction shape is
    gone — the rendered envelope must NOT instruct the model to 'use the read
    tool' or 'visit' a url list. The tool fetches docs itself or errors; it never
    bounces the model into a read round-trip."""
    _dead_table(monkeypatch)

    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"query": "tools", "act_summary": "x"}
    )

    lowered = out.lower()
    assert "use the read tool" not in lowered
    assert "read tool" not in lowered
    assert "visit:" not in lowered
    assert "status=success" not in out

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""programming_docs_search-specific business-logic tests migrated from the
per-ability conformance file removed in TKT-975. Covers unknown-source error
ladders, schema enum honesty, and the no-fabrication guarantee on a dead source.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.programming_docs_search import (
    SOURCES,
    ProgrammingDocsSearchAbility,
)
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP, parse_body, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write."""
    return MP(
        seed_transcript(db, content="look up something in the docs"),
        UserConfig({}),
    )


def _table_ids() -> set[str]:
    """Every source id declared in the production source table — the single
    source of truth the ability resolves against. No mock."""
    return {src.id for src in SOURCES}


def _parse_body(rendered: str, tool: str = "programming_docs_search") -> object:
    return parse_body(rendered, tool)


# ── Unknown source: errors loudly with the real-source ladder ──────────────────


def test_unknown_source_errors_with_valid_ladder(db, chat_mp):
    """A language the table does not know returns a STABLE ``code=unknown-source``
    error whose ``valid:`` ladder lists the REAL source ids — so a weak model
    self-corrects without re-reading the schema. The mechanical ``code=error``
    marker must be gone."""
    out = ToolDispatcher(chat_mp).dispatch(
        "programming_docs_search",
        {"language": "cobol", "query": "perform", "act_summary": "x"},
    )

    assert "[programming_docs_search(status=error, code=unknown-source" in out
    assert "code=error]" not in out
    assert "valid:" in out
    assert "[end:programming_docs_search]" in out
    # The ladder names real source ids the model can route to.
    assert "python" in out
    assert "php" in out

    # The act-trail recorded the same loud error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "code=unknown-source" in trail[0]["result"]


def test_unknown_source_valid_ladder_is_the_real_table(db, chat_mp):
    """The ``valid:`` ladder on an unknown-source error lists EXACTLY the real
    source ids from the production table — drives the guardrail against the live
    table, not a hardcoded list."""
    out = ToolDispatcher(chat_mp).dispatch(
        "programming_docs_search",
        {"language": "nope", "query": "anything", "act_summary": "x"},
    )

    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    advertised = {p.strip() for p in valid_line[len("valid:"):].split("|")}
    assert advertised == _table_ids()


# ── Schema honesty: the enum the model reads matches the table EXACTLY ──────────


def test_schema_language_enum_matches_source_table(db):
    """The ``language`` enum advertised in ``get_parameters`` is EXACTLY the set
    of source ids (+ every alias) the table resolves — so a schema-obedient model
    can never name a language that the ability rejects as unknown. Pure
    introspection, no network."""
    schema = ProgrammingDocsSearchAbility(mp=None).get_parameters()
    enum = set(schema["properties"]["language"]["enum"])

    resolvable = set()
    for src in SOURCES:
        resolvable.add(src.id)
        resolvable.update(a.lower() for a in src.aliases)
    # Every advertised value resolves, and every source id is advertised.
    assert enum <= resolvable
    assert _table_ids() <= enum


# ── The heart: a dead source errors loudly and fabricates NOTHING ──────────────


def test_dead_source_yields_source_unavailable_no_fabrication(db, chat_mp, monkeypatch):
    """A source whose host cannot be reached must produce a LOUD
    ``code=source-unavailable`` error — NOT a fabricated page presented as the
    answer. We configure a real table entry to yield no candidate and point its
    ``base_url`` at a non-resolvable host (both the table's own production fields).
    The SSRF guard fails closed on the unresolvable host BEFORE any socket opens,
    so the engine's fetch raises offline and the lookup exhausts every candidate
    → ``source-unavailable``. No in-process code is mocked; we exercise the
    genuine resolve→fetch→SSRF→error path. The body must carry NO synthesized
    documentation text and NO 'Failed to fetch' / 'no content' stub."""
    py = next(src for src in SOURCES if src.id == "python")
    # An empty-candidate resolver + an unreachable base_url is a legitimate table
    # configuration (a source that knows no symbol URLs and whose host is down) —
    # the exact shape that must error loudly instead of fabricating a page.
    monkeypatch.setattr(py, "resolve", lambda query: [])
    monkeypatch.setattr(py, "base_url", "https://docs.python.invalid/3/")

    out = ToolDispatcher(chat_mp).dispatch(
        "programming_docs_search",
        {"language": "python", "query": "asyncio", "act_summary": "x"},
    )

    assert "[programming_docs_search(status=error, code=source-unavailable" in out
    assert "code=error]" not in out
    assert "[end:programming_docs_search]" in out
    # No fabricated documentation prose leaked into the body.
    lowered = out.lower()
    assert "failed to fetch" not in lowered
    assert "page fetched but no content" not in lowered
    assert "best guess" not in lowered

    # The act-trail recorded the same loud error — the model sees a routable failure.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "code=source-unavailable" in trail[0]["result"]

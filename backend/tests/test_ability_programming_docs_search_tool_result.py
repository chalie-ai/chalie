# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for programming_docs_search's ToolResult contract (TKT-892).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``ProgrammingDocsSearchAbility``, the real ``PolicyManager.wrap`` gate
(``programming_docs_search`` is allowed on chat), the real
``ProgrammingDocsSearchAbility.run``, and the real ``ActTrail`` write.

The regression under test (the worst failure mode the audit named): when a
source lookup FAILS, the old ability fabricated fallback page text and presented
it as the answer — the model relayed invented documentation with status=success.
The fix: a dead/unreachable source must produce a LOUD ``code=source-unavailable``
error the model can route around, and the body must carry NO synthesized page
text. The mechanical ``code=error`` marker from the pre-TKT-882 era must be gone.

NETWORK CONSTRAINT: unit tests are deterministic and offline. We never hit a
real documentation host. Every path asserted here short-circuits BEFORE any
successful HTTP round-trip:

* unknown-source rejection + the ``valid:`` ladder of real source ids (no network),
* missing-param pre-gate (no network),
* schema honesty — the enum advertised in ``get_parameters`` matches the source
  table EXACTLY (pure introspection), and
* the no-fabrication guarantee — a source resolving to a non-resolvable host
  fails the SSRF guard fail-closed (``services.ssrf.resolve_and_check`` rejects an
  unresolvable host BEFORE any socket opens), so the fetch raises offline and the
  ability returns ``code=source-unavailable`` with NO invented body. The dead
  host is supplied through the production ``base_url`` override seam the table
  already exposes, not by mocking any in-process code.

The success body (real fetched documentation) requires a live host; there is no
respx/responses precedent in this suite, so it is documented as a network
coverage gap in the ticket report rather than mocked.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.programming_docs_search import (
    SOURCES,
    ProgrammingDocsSearchAbility,
)
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "look up something in the docs"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the chat policy channel) and ``uid`` (the transcript
    anchor the trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _table_ids() -> set[str]:
    """Every source id declared in the production source table — the single
    source of truth the ability resolves against. No mock."""
    return {src.id for src in SOURCES}


def _parse_body(rendered: str, tool: str = "programming_docs_search") -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:<tool>]``."""
    head = rendered.index("]\n") + 2
    tail = rendered.index(f"\n[end:{tool}]")
    return json.loads(rendered[head:tail])


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


# ── Missing params: pre-gated, never reaches the network ───────────────────────


def test_missing_query_reports_error(db, chat_mp):
    """A call with no ``query`` is rejected with a stable error — BEFORE run()
    ever touches a source or the network. The ``code=error`` marker is gone."""
    out = ToolDispatcher(chat_mp).dispatch(
        "programming_docs_search", {"language": "python", "act_summary": "x"}
    )

    assert "[programming_docs_search(status=error" in out
    assert "code=error]" not in out
    assert "query" in out
    assert "[end:programming_docs_search]" in out


def test_missing_language_reports_error(db, chat_mp):
    """A call with no ``language`` is rejected with a stable error, never DDG-style
    masked — no network."""
    out = ToolDispatcher(chat_mp).dispatch(
        "programming_docs_search", {"query": "asyncio", "act_summary": "x"}
    )

    assert "[programming_docs_search(status=error" in out
    assert "code=error]" not in out
    assert "language" in out
    assert "[end:programming_docs_search]" in out


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

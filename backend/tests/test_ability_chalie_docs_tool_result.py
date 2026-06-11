# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for chalie_docs's ToolResult contract (TKT-901).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint exactly as ``MessageProcessor._loop``
does — real ``AbilityRegistry`` resolution of the production
``ChalieDocsAbility``, the real ``PolicyManager.wrap`` gate (``chalie_docs`` is in
``PolicyManager.INTERNAL`` so it needs no policy row), the real
``ChalieDocsAbility.run``, the real ``services.web_fetch`` fetch stack + SSRF
guard, and the real ``ActTrail`` write read back from the db.

The regression under test: the old ability answered with an INSTRUCTION to call
the ``read`` tool and visit a URL list — a round-trip a weak model fumbles —
instead of fetching the documentation itself. The fix fetches the doc server-side
and returns the prose; an unknown query is a stable ``code=doc-not-found`` (the
banned ``code=error`` marker is gone) and an all-urls-down outage is a loud
``code=fetch-failed`` carrying NO instruction text.

NETWORK CONSTRAINT: unit tests are deterministic and offline. We never hit a real
chalie.ai host. The no-fabrication / fetch-failed path is exercised by repointing
the production ``_QUERY_URLS`` table's OWN entries at non-resolvable
``chalie-docs.invalid`` hosts (the same production seam, not a mock): the SSRF
guard's resolver rejects the unresolvable host BEFORE any socket opens, so the
real fetch raises ``requests.RequestException`` offline and the ability returns
``code=fetch-failed`` with no body. Unknown-query and missing-query short-circuit
BEFORE any fetch.

Network coverage gap (documented, same as prior tickets): the happy-path content
fetch (a real chalie.ai page extracted to prose) and the PARTIAL-outage path (one
live url + one dead url → success with ``failed_sources`` in meta) both require
the wire — there is no respx/responses precedent in this suite, so they are
documented here as a coverage gap rather than mocked.
"""

import pytest

import abilities.chalie_docs
from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs
    its recorded rows off, and return its id."""
    return seed_transcript(db, channel=channel, content="tell me about chalie")


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _dead_table(monkeypatch) -> None:
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


# ── Unknown query: errors loudly with the real valid ladder + closest match ─────


def test_unknown_query_errors_with_doc_not_found_and_closest_match(db, chat_mp):
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
    assert "code=doc-not-found" in trail[0]["result"]


def test_unknown_query_valid_ladder_is_the_real_table(db, chat_mp):
    """The ``valid:`` ladder on a doc-not-found error lists EXACTLY the real query
    keys from the production table — drives the guardrail against the live table,
    not a hardcoded list."""
    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"query": "nonsense", "act_summary": "x"}
    )

    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    advertised = {p.strip() for p in valid_line[len("valid:"):].split("|")}
    assert advertised == set(abilities.chalie_docs._QUERY_URLS)


# ── Missing query: pre-gated by the dispatcher, never reaches run() ─────────────


def test_missing_query_reports_missing_params(db, chat_mp):
    """A call with no ``query`` is rejected by the dispatcher's ACTION_REQUIRED
    pre-gate with ``code=missing-params`` — BEFORE run() ever touches a url or the
    network. The banned ``code=error`` marker is gone."""
    out = ToolDispatcher(chat_mp).dispatch(
        "chalie_docs", {"act_summary": "x"}
    )

    assert "[chalie_docs(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "query" in out
    assert "[end:chalie_docs]" in out


# ── The heart: all urls unreachable errors loudly, no instruction fabrication ───


def test_all_urls_unreachable_yields_fetch_failed_no_instruction(db, chat_mp, monkeypatch):
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
    assert "code=fetch-failed" in trail[0]["result"]


def test_failure_carries_no_read_tool_instruction(db, chat_mp, monkeypatch):
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

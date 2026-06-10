# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the search tool's ToolResult contract (TKT-891).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``SearchAbility``, the real ``PolicyManager.wrap`` gate (``search`` is an INTERNAL
tool that bypasses the gate unconditionally), the real ``SearchAbility.run`` with
the real on-disk provider registry (``search_tool_providers.sqlite``), and the
real ``ActTrail`` write.

The regression under test (the exact silent-masking bug TKT-891 closes): when the
model FORCES a provider that is not a real provider name, today the DDG fallback
silently masks it and the model believes it searched the engine it named. The
guardrail must instead loudly error with ``code=unknown-provider`` and a ``valid:``
ladder of the REAL provider names — and crucially the schema enum the model reads
must itself name only real providers (the shipped enum advertised ``hackernews``
and ``stackoverflow``, which do NOT exist — the real rows are ``hn_algolia`` and
``stack_exchange`` — so a schema-obedient model was silently DDG-faded every time).

NETWORK CONSTRAINT: unit tests are deterministic and offline. We never hit a real
search engine. Every path asserted here short-circuits BEFORE any HTTP call:
unknown-provider rejection (no network), missing-query pre-gate (no network), and
schema honesty (pure introspection against the real registry). The structured
result body and the runtime ``meta fallback=ddg`` path require a live engine
response; there is no respx/responses precedent in this suite, so those are
covered against the real registry only at the rejection boundary and documented as
a network-coverage gap in the ticket report rather than mocked.
"""

import json
import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.search import SearchAbility
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "search for something"),
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


def _configured_providers() -> set[str]:
    """The REAL set of enabled provider names from the on-disk registry — the same
    DB ``SearchAbility._load_providers`` reads at runtime. No mock."""
    conn = sqlite3.connect(str(FileMapperService.get_search_providers_db_path()))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT name FROM providers WHERE enabled = 1").fetchall()
    conn.close()
    return {r["name"] for r in rows}


def _meta_head(rendered: str, tool: str = "search") -> str:
    """The open-tag line ``[search(status=…, …)]`` the model reads first."""
    return rendered.splitlines()[0]


def _parse_body(rendered: str, tool: str = "search") -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:<tool>]``."""
    head = rendered.index("]\n") + 2
    tail = rendered.index(f"\n[end:{tool}]")
    return json.loads(rendered[head:tail])


# ── The heart: a forced UNKNOWN provider errors loudly, never DDG-masked ───────


def test_forced_unknown_provider_errors_not_silent_ddg(db, chat_mp):
    """Forcing a misspelled/unknown provider (``kagi``) returns a STABLE
    ``code=unknown-provider`` error with a ``valid:`` ladder of REAL provider
    names — and crucially NO DDG results are returned masquerading as the named
    engine. This is the exact silent-masking bug TKT-891 closes."""
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "rust foundation", "provider": "kagi", "act_summary": "x"}
    )

    assert "[search(status=error, code=unknown-provider" in out
    assert "code=error]" not in out
    assert "valid:" in out
    assert "[end:search]" in out
    # The valid ladder names a real configured provider so a weak model self-corrects.
    assert "wikipedia" in out
    # The silent-masking bug: NO web results came back dressed up as 'kagi'.
    assert "http" not in out.split("valid:")[0]

    # The act-trail recorded the same loud error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "[search(status=error, code=unknown-provider" in trail[0]["result"]


def test_unknown_provider_valid_ladder_is_the_real_registry(db, chat_mp):
    """The ``valid:`` ladder on an unknown-provider error lists EXACTLY the real
    configured providers (+ ``ddg``) — not the stale ``hackernews``/``stackoverflow``
    aliases that never existed. Drives the guardrail against the live registry."""
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "anything", "provider": "stackoverflow", "act_summary": "x"}
    )

    assert "[search(status=error, code=unknown-provider" in out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    advertised = {p.strip() for p in valid_line[len("valid:"):].split("|")}

    expected = _configured_providers() | {"ddg"}
    assert advertised == expected
    # The stale aliases the old schema advertised are gone — they were the bug.
    assert "hackernews" not in advertised
    assert "stackoverflow" not in advertised


# ── Schema honesty: the enum the model reads names only real providers ─────────


def test_schema_provider_enum_matches_real_registry(db):
    """The ``provider`` enum advertised in ``get_parameters`` is EXACTLY the real
    configured provider names plus ``ddg`` — so a schema-obedient model can never
    name a provider that silently DDG-fades. Pure introspection, no network."""
    schema = SearchAbility(mp=None).get_parameters()
    enum = set(schema["properties"]["provider"]["enum"])

    expected = _configured_providers() | {"ddg"}
    assert enum == expected
    # The shipped enum's phantom providers must be gone.
    assert "hackernews" not in enum
    assert "stackoverflow" not in enum


# ── Missing query: pre-gated, never reaches the network ────────────────────────


def test_missing_query_reports_missing_params(db, chat_mp):
    """A call with no ``query`` is rejected with a stable error naming the missing
    param — BEFORE run() ever touches a provider or the network."""
    out = ToolDispatcher(chat_mp).dispatch("search", {"act_summary": "x"})

    assert "[search(status=error" in out
    assert "code=error]" not in out
    assert "query" in out
    assert "[end:search]" in out


def test_blank_query_reports_missing_params(db, chat_mp):
    """A whitespace-only ``query`` is treated the same as missing — no network."""
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "   ", "act_summary": "x"}
    )

    assert "[search(status=error" in out
    assert "code=error]" not in out
    assert "[end:search]" in out

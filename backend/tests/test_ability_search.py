# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""search-specific business-logic tests migrated from the per-ability conformance
file removed in TKT-975. Covers the silent-DDG-masking regression (forced unknown
provider must error loudly), the real-registry valid ladder, schema enum honesty,
and blank-query missing-params rejection.
"""

import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.search import SearchAbility
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService
from tests._tool_result_harness import MP, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write."""
    return MP(seed_transcript(db, "chat", "search for something"), UserConfig({}))


def _configured_providers() -> set[str]:
    """The REAL set of enabled provider names from the on-disk registry — the same
    DB ``SearchAbility._load_providers`` reads at runtime. No mock."""
    conn = sqlite3.connect(str(FileMapperService.get_search_providers_db_path()))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT name FROM providers WHERE enabled = 1").fetchall()
    conn.close()
    return {r["name"] for r in rows}


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


# ── Missing/blank query: pre-gated, never reaches the network ─────────────────


def test_blank_query_reports_missing_params(db, chat_mp):
    """A whitespace-only ``query`` is treated the same as missing — no network."""
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "   ", "act_summary": "x"}
    )

    assert "[search(status=error" in out
    assert "code=error]" not in out
    assert "[end:search]" in out

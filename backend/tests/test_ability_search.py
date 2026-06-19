# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.search import SearchAbility
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService
from tests._tool_result_harness import MP, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    return MP(seed_transcript(db, "chat", "search for something"), UserConfig({}))


def _configured_providers() -> set[str]:
    conn = sqlite3.connect(str(FileMapperService.get_search_providers_db_path()))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT name FROM providers WHERE enabled = 1").fetchall()
    conn.close()
    return {r["name"] for r in rows}


# ── The heart: a forced UNKNOWN provider errors loudly, never DDG-masked ───────


def test_forced_unknown_provider_errors_not_silent_ddg(db: sqlite3.Connection, chat_mp: MP) -> None:
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
    assert "[search(status=error, code=unknown-provider" in cast(str, trail[0]["result"])


def test_unknown_provider_valid_ladder_is_the_real_registry(db: sqlite3.Connection, chat_mp: MP) -> None:
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


def test_schema_provider_enum_matches_real_registry(db: sqlite3.Connection) -> None:
    schema = SearchAbility(mp=None).get_parameters()
    enum = set(cast(list[object], cast(dict[str, object], cast(dict[str, object], schema["properties"])["provider"])["enum"]))

    expected = _configured_providers() | {"ddg"}
    assert enum == expected
    # The shipped enum's phantom providers must be gone.
    assert "hackernews" not in enum
    assert "stackoverflow" not in enum


# ── Blank query: slips the pre-gate, caught by search's own .strip() guard ─────


def test_blank_query_reports_missing_params(db: sqlite3.Connection, chat_mp: MP) -> None:
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "   ", "act_summary": "x"}
    )

    assert "[search(status=error, code=missing-params" in out
    assert "query" in out
    assert "[end:search]" in out

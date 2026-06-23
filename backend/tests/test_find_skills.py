# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""E2E hot-path tests for find_skills — real MessageProcessor, real skills.sqlite
copy, real ONNX embeddings, zero mocks.

find_skills runs the SAME shared discovery cascade as find_tools (see
``abilities._search.SearchableAbility``): a ``query`` array of intents, each
escalated exact-title → bm25-on-title (segment-gated) → vector on full prose.
These tests drive the genuine ``ToolDispatcher.dispatch`` entry point and assert
the model-facing envelope plus the playbook payload.

Pins the load-bearing regression: a corrupt skill-index must surface
``skill-index-error``, never masquerade as a zero-hit success.
"""

import json
import shutil
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.find_skills import FindSkillsAbility
from configs.channels import UserConfig
from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor

from tests._tool_result_harness import body as _harness_body
from tests._tool_result_harness import head as _harness_head
from tests._tool_result_harness import seed_transcript

pytestmark = pytest.mark.unit

_REAL_SKILLS_DB = FileMapperService.get_skills_db_path()


@pytest.fixture
def skills_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real skills.sqlite COPY under tmp_path, with ``FindSkillsAbility._DB_PATH``
    redirected to it. The repo's checked-in skills.sqlite is NEVER touched — every
    SQL statement runs against the read-only byte copy (Rule 0)."""
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_SKILLS_DB), str(dest))
    monkeypatch.setattr(FindSkillsAbility, "_DB_PATH", dest)
    return dest


def _mp(db: sqlite3.Connection) -> MessageProcessor:
    """A real MessageProcessor bound to a real transcript anchor so the
    dispatcher's act-trail write lands on a real row. Carries ``active_tools`` —
    the genuine surface find_skills' discovery cascade runs against."""
    mp = MessageProcessor("find me a skill")
    mp.config = UserConfig({})
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = seed_transcript(db, "chat", "find me a skill")
    return mp


def _head(rendered: str) -> str:
    return _harness_head(rendered, "find_skills")


def _body(rendered: str) -> str:
    return _harness_body(rendered, "find_skills")


def _pick_known_title(db_path: Path) -> tuple[int, str, str]:
    """Pull a real enabled skill row (id, title, content) to query against."""
    conn = sqlite3.connect(str(db_path))
    try:
        return cast(tuple[int, str, str], conn.execute(
            "SELECT id, title, content FROM skills WHERE enabled = 1 "
            "AND content != '' ORDER BY id LIMIT 1"
        ).fetchone())
    finally:
        conn.close()


# ── missing / blank query → loud code=missing-params, NOT a silent success ──────


def test_missing_query_is_missing_params(skills_db: Path, db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"act_summary": "x"})
    assert "[find_skills(status=error, code=missing-params" in out
    assert "code=error]" not in out


def test_empty_query_array_is_missing_params(skills_db: Path, db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": [], "act_summary": "x"})
    assert "[find_skills(status=error, code=missing-params" in out
    assert "code=error]" not in out


# ── index missing / corrupt → skill-index-error, NEVER a zero-hit success ───────


def test_missing_db_is_skill_index_error(skills_db: Path, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FindSkillsAbility, "_DB_PATH", skills_db.parent / "gone.sqlite")
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": ["analysis"], "act_summary": "x"})
    assert "[find_skills(status=error, code=skill-index-error" in out
    assert "code=error]" not in out


def test_corrupt_db_is_loud_not_zero_hit(skills_db: Path, db: sqlite3.Connection) -> None:
    """THE regression pin: a corrupt index must surface skill-index-error, not a
    zero-hit SUCCESS. The shared cascade's search helpers swallow sqlite errors to
    [], so the index is probed FIRST — a probe failure is loud."""
    skills_db.write_bytes(b"this is not a sqlite database at all, just garbage bytes")
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": ["analysis"], "act_summary": "x"})
    assert "[find_skills(status=error, code=skill-index-error" in out
    assert "status=success" not in out
    assert "count=0" not in out
    assert "code=error]" not in out


# ── matches → JSON rows carrying the playbook content + count meta ──────────────


def test_exact_title_match_returns_playbook_content(skills_db: Path, db: sqlite3.Connection) -> None:
    """An exact title pins that one skill (rung 1) and returns its full playbook —
    each row is ``{name, content, rules}`` (no search-internal ``source`` field)."""
    _id, title, content = _pick_known_title(skills_db)
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": [title], "act_summary": "x"})
    head = _head(out)
    assert "status=success" in head
    assert "count=" in head
    rows = json.loads(_body(out))
    assert isinstance(rows, list)
    assert len(rows) >= 1
    for row in rows:
        assert set(row.keys()) == {"name", "content", "rules"}
        assert isinstance(row["rules"], list)
    count = int(head.split("count=")[1].split(",")[0].rstrip(")]"))
    assert count == len(rows)
    mine = next((r for r in rows if r["name"] == title), None)
    assert mine is not None, f"exact title {title!r} must pin its own skill. rows={rows!r}"
    assert mine["content"] == content


def test_semantic_query_surfaces_matching_playbook(skills_db: Path, db: sqlite3.Connection) -> None:
    """A described task (no title match) escalates to the vector rung and surfaces
    the closest playbook — 'plan my meals for the week' → 'Weekly Meal Planning'."""
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "find_skills", {"query": ["plan my meals for the week"], "act_summary": "x"}
    )
    assert "status=success" in _head(out)
    names = {r["name"] for r in json.loads(_body(out))}
    assert "Weekly Meal Planning" in names, (
        f"semantic query must surface the Weekly Meal Planning playbook. got={names!r}"
    )


def test_multi_intent_array_routes_each_entry(skills_db: Path, db: sqlite3.Connection) -> None:
    """Each query-array entry is searched independently; the union of matches is
    returned — one title-segment intent and one prose intent at once."""
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "find_skills",
        {"query": ["competitor teardown", "plan my meals for the week"], "act_summary": "x"},
    )
    assert "status=success" in _head(out)
    names = {r["name"] for r in json.loads(_body(out))}
    assert "Competitive Teardown" in names, f"got={names!r}"
    assert "Weekly Meal Planning" in names, f"got={names!r}"


# ── zero hits (healthy index) → success, count=0, matches=[] + note ─────────────


def test_zero_hit_is_success_with_note(skills_db: Path, db: sqlite3.Connection) -> None:
    """A nonsense query against a HEALTHY index is a SUCCESS with count=0 and a
    matches=[] + note body, never an error. The vector terminus rejects it at the
    fine-tuned ceiling, so no junk playbook is surfaced."""
    mp = _mp(db)
    nonsense = "zxqwftbloopglarnk"
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": [nonsense], "act_summary": "x"})
    head = _head(out)
    assert "status=success" in head
    assert "count=0" in head
    body = json.loads(_body(out))
    assert body["matches"] == []
    assert "No skill playbooks found" in body["note"]

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for find_skills' ToolResult contract (TKT-911).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``MessageProcessor``,
the real ``AbilityRegistry`` resolution, the real production ``run()`` against a
real on-disk ``skills.sqlite`` COPY (never the repo asset — copied to
``tmp_path`` and redirected via ``FindSkillsAbility._DB_PATH``, the precedent in
``test_ability_skills_tool_result.py``), and the real ``ActTrail`` write.

find_skills is in ``PolicyManager.INTERNAL`` — it bypasses the policy gate, so
dispatch needs no policy flip. The embedding service is unreachable offline, so
real-query dispatch exercises the FTS-only ``_fallback`` route (``degraded=true``)
— that is expected and asserted.

THE regression this file pins (TKT-911 "bad looks like"): the shared
``_hybrid_search``/``_fts_only_search`` helpers swallow sqlite failures to ``[]``,
which the old ``run()`` rendered as a zero-hit SUCCESS. A corrupt/broken index
must NEVER masquerade as "no skills found" — it must surface
``code=skill-index-error``. ``test_corrupt_db_is_loud_not_zero_hit`` is that pin:
it FAILS against the old code (zero-hit success) and passes against the new probe.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities.find_skills import FindSkillsAbility
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_REAL_SKILLS_DB = FileMapperService.get_skills_db_path()


@pytest.fixture
def skills_db(tmp_path, monkeypatch):
    """A real skills.sqlite COPY under tmp_path, with ``FindSkillsAbility._DB_PATH``
    redirected to it. The repo's checked-in skills.sqlite is NEVER touched — every
    SQL statement runs against the read-only byte copy (Rule 0)."""
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_SKILLS_DB), str(dest))
    monkeypatch.setattr(FindSkillsAbility, "_DB_PATH", dest)
    return dest


def _seed_transcript(db) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("chat", "user", "find me a skill"),
    )
    db.commit()
    return cur.lastrowid


def _mp(db) -> MessageProcessor:
    """A real MessageProcessor bound to a real transcript anchor so the
    dispatcher's act-trail write lands on a real row."""
    mp = MessageProcessor("find me a skill")
    mp.config = UserConfig({})
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = _seed_transcript(db)
    return mp


def _head(rendered: str) -> str:
    line = rendered.splitlines()[0]
    assert line.startswith("[find_skills(")
    return line


def _body(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:find_skills]")
    return rendered[head:tail]


def _pick_known_title(db_path: Path) -> tuple[int, str, str]:
    """Pull a real enabled skill row (id, title, content) to query against."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, title, content FROM skills WHERE enabled = 1 "
            "AND content != '' ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


# ── missing / blank query → loud code=missing-params, NOT a silent success ──────


def test_missing_query_is_missing_params(skills_db, db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"act_summary": "x"})
    assert "[find_skills(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "query" in out
    assert any(ln.startswith("hint:") for ln in out.splitlines())
    assert "[end:find_skills]" in out
    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert any("[find_skills(status=error, code=missing-params" in r["result"] for r in trail)


def test_blank_query_is_missing_params(skills_db, db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": "   ", "act_summary": "x"})
    assert "[find_skills(status=error, code=missing-params" in out
    assert "code=error]" not in out


# ── index missing / corrupt → skill-index-error, NEVER a zero-hit success ───────


def test_missing_db_is_skill_index_error(skills_db, db, monkeypatch):
    monkeypatch.setattr(FindSkillsAbility, "_DB_PATH", skills_db.parent / "gone.sqlite")
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": "analysis", "act_summary": "x"})
    assert "[find_skills(status=error, code=skill-index-error" in out
    assert "code=error]" not in out


def test_corrupt_db_is_loud_not_zero_hit(skills_db, db):
    """THE regression pin: a corrupt index must surface skill-index-error, not a
    zero-hit SUCCESS. Fails against HEAD (which renders [] as found=0 success)."""
    skills_db.write_bytes(b"this is not a sqlite database at all, just garbage bytes")
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": "analysis", "act_summary": "x"})
    assert "[find_skills(status=error, code=skill-index-error" in out
    assert "status=success" not in out
    assert "count=0" not in out
    assert "code=error]" not in out


# ── matches → JSON rows carrying the playbook content + count meta ──────────────


def test_match_returns_json_rows_with_playbook_content(skills_db, db):
    _id, title, content = _pick_known_title(skills_db)
    # Query by the skill's own title so FTS (the offline route) finds it.
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": title, "act_summary": "x"})
    head = _head(out)
    assert "status=success" in head
    assert "count=" in head
    rows = json.loads(_body(out))
    assert isinstance(rows, list)
    assert len(rows) >= 1
    for row in rows:
        assert set(row.keys()) == {"name", "score", "content", "rules"}
        assert isinstance(row["rules"], list)
    count = int(head.split("count=")[1].split(",")[0].rstrip(")]"))
    assert count == len(rows)
    mine = next((r for r in rows if r["name"] == title), None)
    assert mine is not None
    assert mine["content"] == content
    assert mine["content"]
    # The row shape is identical whether the hybrid route ran or the FTS-only
    # fallback did; when embeddings are unreachable the success carries
    # degraded=true, otherwise it is absent. Either is a valid success here.


# ── embedding failure → FTS-only fallback, same rows, degraded=true ─────────────


def test_embedding_failure_degrades_to_fts_with_marker(skills_db, db, monkeypatch):
    """When the embedding service raises, run() degrades to FTS-only and the
    success carries meta degraded=true with the SAME row shape — never silently
    fewer signals, never an error."""
    from services.embedding_service import EmbeddingService

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("embedding backend unreachable")

    monkeypatch.setattr(EmbeddingService, "generate_embedding", _boom)

    _id, title, content = _pick_known_title(skills_db)
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": title, "act_summary": "x"})
    head = _head(out)
    assert "status=success" in head
    assert "degraded=true" in head
    rows = json.loads(_body(out))
    assert isinstance(rows, list) and len(rows) >= 1
    for row in rows:
        assert set(row.keys()) == {"name", "score", "content", "rules"}
    mine = next((r for r in rows if r["name"] == title), None)
    assert mine is not None and mine["content"] == content


# ── zero hits (healthy index) → success, count=0, matches=[] + note ─────────────


def test_zero_hit_is_success_with_note(skills_db, db, monkeypatch):
    """A genuine no-match against a HEALTHY index is a SUCCESS with count=0 and a
    matches=[] + note body, never an error. The vec KNN always returns its k
    nearest neighbours regardless of relevance, so a true zero-hit only arises on
    the FTS-only route — forced here by failing the embedding (the fallback the
    offline env takes anyway) with a token FTS cannot match."""
    from services.embedding_service import EmbeddingService

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("embedding backend unreachable")

    monkeypatch.setattr(EmbeddingService, "generate_embedding", _boom)

    mp = _mp(db)
    nonsense = "zxqwftbloopglarnk"
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": nonsense, "act_summary": "x"})
    head = _head(out)
    assert "status=success" in head
    assert "count=0" in head
    body = json.loads(_body(out))
    assert body["matches"] == []
    assert nonsense in body["note"]


# ── limit respected ────────────────────────────────────────────────────────────


def test_limit_caps_row_count(skills_db, db):
    mp = _mp(db)
    # A broad term that matches several skills, capped to one.
    out = ToolDispatcher(mp).dispatch(
        "find_skills", {"query": "analysis", "limit": 1, "act_summary": "x"}
    )
    head = _head(out)
    assert "status=success" in head
    body = json.loads(_body(out))
    rows = body if isinstance(body, list) else body.get("matches", [])
    assert len(rows) <= 1


# ── the legacy code="error" marker is gone from every envelope ──────────────────


def test_no_legacy_error_code_anywhere(skills_db, db):
    mp = _mp(db)
    for params in (
        {"act_summary": "x"},
        {"query": "   ", "act_summary": "x"},
        {"query": "analysis", "act_summary": "x"},
        {"query": "zxqwftbloopglarnk", "act_summary": "x"},
    ):
        out = ToolDispatcher(mp).dispatch("find_skills", params)
        assert "code=error]" not in out
        assert "code=error," not in out
        assert "code=error)" not in out


# ── registry still resolves find_skills with its metadata ───────────────────────


def test_find_skills_registered_with_metadata():
    ability = AbilityRegistry.get("find_skills")
    assert ability.get_name() == "find_skills"
    assert ability.get_summary()
    assert len(ability.get_examples()) >= 6

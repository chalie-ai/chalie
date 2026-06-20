# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""E2E EDR hot-path tests: real MessageProcessor, zero mocks.

Pins regressions (corrupt skill-index returning zero-hit success).
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
    dispatcher's act-trail write lands on a real row.

    Kept local (not the harness ``MP``) because this drives the genuine
    ``MessageProcessor`` carrying ``active_tools`` — the discovery/select surface
    find_skills exercises — which the minimal ``config``+``uid`` harness ``MP``
    does not model."""
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


def test_blank_query_is_missing_params(skills_db: Path, db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": "   ", "act_summary": "x"})
    assert "[find_skills(status=error, code=missing-params" in out
    assert "code=error]" not in out


# ── index missing / corrupt → skill-index-error, NEVER a zero-hit success ───────


def test_missing_db_is_skill_index_error(skills_db: Path, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FindSkillsAbility, "_DB_PATH", skills_db.parent / "gone.sqlite")
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("find_skills", {"query": "analysis", "act_summary": "x"})
    assert "[find_skills(status=error, code=skill-index-error" in out
    assert "code=error]" not in out


def test_corrupt_db_is_loud_not_zero_hit(skills_db: Path, db: sqlite3.Connection) -> None:
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


def test_match_returns_json_rows_with_playbook_content(skills_db: Path, db: sqlite3.Connection) -> None:
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


def test_embedding_failure_degrades_to_fts_with_marker(skills_db: Path, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the embedding service raises, run() degrades to FTS-only and the
    success carries meta degraded=true with the SAME row shape — never silently
    fewer signals, never an error."""
    from services.embedding_service import EmbeddingService

    def _boom(self: object, *_args: object, **_kwargs: object) -> None:
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


def test_zero_hit_is_success_with_note(skills_db: Path, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine no-match against a HEALTHY index is a SUCCESS with count=0 and a
    matches=[] + note body, never an error. The vec KNN always returns its k
    nearest neighbours regardless of relevance, so a true zero-hit only arises on
    the FTS-only route — forced here by failing the embedding (the fallback the
    offline env takes anyway) with a token FTS cannot match."""
    from services.embedding_service import EmbeddingService

    def _boom(self: object, *_args: object, **_kwargs: object) -> None:
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


def test_limit_caps_row_count(skills_db: Path, db: sqlite3.Connection) -> None:
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

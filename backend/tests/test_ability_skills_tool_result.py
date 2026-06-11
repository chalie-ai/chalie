# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the merged skill tool pair's ToolResult contract (TKT-896).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``MessageProcessor``
carrying the production ``UserConfig`` (for ``skill_builder``, gated by the real
``PolicyManager`` against the real ``policy`` table the ``db`` fixture binds) or
``SkillSuggestionConfig`` (for ``skill_manager``, which is INTERNAL and bypasses
the gate), the real ``AbilityRegistry`` resolution, the real production
``run()`` against a real on-disk ``skills.sqlite`` COPY (never the repo asset —
copied to ``tmp_path`` and redirected via the module Path constants, exactly the
precedent in ``test_skill_builder.py``), and the real ``ActTrail`` write.

THE regression this file pins (TKT-896 "good looks like"): a skill whose BODY
legitimately contains the literal string ``skill_builder`` must survive a
``skill_manager`` operation **byte-identical**. The old ``skill_manager.run()``
ran a blanket ``content.replace('skill_builder', 'skill_manager')`` over the
whole rendered result, silently rewriting any such occurrence. The merge deletes
that path entirely; this test stores such a skill through the real create path,
runs a real ``skill_manager`` list over it, fetches the row back from the real DB
and asserts the content is unchanged to the byte.

Plus the contract: list returns JSON rows + count meta; missing params hit the
ACTION_REQUIRED pre-gate; an unknown action returns ``code=unknown-action`` with a
``valid:`` ladder for BOTH tool names; the legacy ``code=error`` marker is gone;
and both names stay registered with their own metadata.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from configs.channels import SkillSuggestionConfig, UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor
from tests._tool_result_harness import body, head, seed_transcript

pytestmark = pytest.mark.unit

_REAL_SKILLS_DB = FileMapperService.get_skills_db_path()


@pytest.fixture
def skills_db(tmp_path, monkeypatch):
    """A real skills.sqlite COPY under tmp_path, with the module Path constants in
    both ``utils.skills_io`` and ``abilities.skill_builder`` redirected to it.

    This is the exact precedent in test_skill_builder.py: only the file-path
    constants are redirected (infrastructure); every SQL statement, YAML write,
    embedding call and FTS index update runs through real production code. The
    repo's checked-in skills.sqlite is NEVER touched — Rule 0.
    """
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_SKILLS_DB), str(dest))
    yaml_dir = tmp_path / "user_skills"
    yaml_dir.mkdir()

    import abilities.skill_builder as sb
    import utils.skills_io as sio

    monkeypatch.setattr(sio, "SKILLS_DB_PATH", dest)
    monkeypatch.setattr(sio, "USER_SKILLS_DIR", yaml_dir)
    monkeypatch.setattr(sb, "SKILLS_DB_PATH", dest)
    return {"db_path": dest, "yaml_dir": yaml_dir}


def _mp_for(config, db) -> MessageProcessor:
    """A real MessageProcessor bound to a real channel config and a real
    transcript anchor (so the dispatcher's act-trail write lands on a real row),
    seeded exactly as ``_setup`` seeds it."""
    mp = MessageProcessor("manage my skills")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    pc = getattr(config, "policy_channel", None)
    mp.uid = seed_transcript(db, pc.value if pc else "chat", "manage my skills")
    return mp


def _head(rendered: str, tool: str) -> str:
    return head(rendered, tool)


def _body(rendered: str, tool: str) -> object:
    return body(rendered, tool)


def _fetch_content(db_path: Path, title: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT content FROM skills WHERE lower(title) = lower(?)", (title,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ── THE regression: skill_builder substring survives a skill_manager op byte-for-byte ──


def test_skill_body_containing_skill_builder_survives_manager_op_byte_identical(
    skills_db, db
):
    """A skill whose body literally says "use skill_builder for X" is created
    through the real skill_builder path, then a real skill_manager list runs over
    it. The stored content must be byte-identical — the blanket .replace the merge
    kills would have rewritten "skill_builder" -> "skill_manager" inside it."""
    title = "Tool Authoring Reminder"
    content = (
        "1. Use `skill_builder` to create the playbook for X.\n"
        "2. Remember: use skill_builder for X, never skill_builder for Y.\n"
        "3. Use `memory` to recall the user's preferences."
    )

    # Create through the real production hot path (skill_builder, user channel).
    builder_mp = _mp_for(UserConfig({}), db)
    created = ToolDispatcher(builder_mp).dispatch(
        "skill_builder",
        {
            "action": "create",
            "title": title,
            "use_for": "authoring new skills",
            "content": content,
            "act_summary": "x",
        },
    )
    assert _head(created, "skill_builder").startswith("[skill_builder(status=success")
    assert _fetch_content(skills_db["db_path"], title) == content  # stored verbatim

    # Now run the SYSTEM variant (skill_manager) over the same DB via real dispatch.
    mgr_mp = _mp_for(SkillSuggestionConfig(), db)
    listed = ToolDispatcher(mgr_mp).dispatch(
        "skill_manager", {"action": "list", "act_summary": "x"}
    )
    assert _head(listed, "skill_manager").startswith("[skill_manager(status=success")

    # The acceptance bar: content survives the manager operation byte-identical —
    # "skill_builder" was NOT silently rewritten to "skill_manager".
    after = _fetch_content(skills_db["db_path"], title)
    assert after == content
    assert "skill_builder" in after
    assert "skill_manager" not in after


# ── list returns JSON rows + count meta ────────────────────────────────────────


def test_list_returns_json_rows_with_count_meta(skills_db, db):
    """list renders a success envelope with a ``count=`` meta and a JSON array
    body of skill rows — structured data, not legacy prose."""
    mp = _mp_for(UserConfig({}), db)
    ToolDispatcher(mp).dispatch(
        "skill_builder",
        {
            "action": "create",
            "title": "Weekly Expense Review",
            "use_for": "reviewing weekly expenses",
            "content": "1. Use `list` to gather expenses.",
            "act_summary": "x",
        },
    )

    out = ToolDispatcher(mp).dispatch(
        "skill_builder", {"action": "list", "act_summary": "x"}
    )
    head = _head(out, "skill_builder")
    assert "status=success" in head
    assert "count=" in head
    rows = json.loads(_body(out, "skill_builder"))
    assert isinstance(rows, list)
    titles = [r["title"] for r in rows]
    assert "Weekly Expense Review" in titles
    mine = next(r for r in rows if r["title"] == "Weekly Expense Review")
    assert mine["source"] == "user"
    assert mine["version"] == 1
    assert mine["use_for"] == "reviewing weekly expenses"


# ── missing params hit the ACTION_REQUIRED pre-gate ────────────────────────────


def test_create_missing_params_hits_pregate_with_named_params(skills_db, db):
    """create without title/use_for/content is rejected by the ACTION_REQUIRED
    pre-gate (before run() and before the policy gate) with a single
    ``code=missing-params`` error naming the missing params — not legacy
    ``code=error``."""
    mp = _mp_for(UserConfig({}), db)
    out = ToolDispatcher(mp).dispatch(
        "skill_builder", {"action": "create", "act_summary": "x"}
    )
    assert "[skill_builder(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "title" in out and "use_for" in out and "content" in out
    assert "[end:skill_builder]" in out


# ── unknown action → valid ladder, for BOTH tool names ─────────────────────────


@pytest.mark.parametrize(
    ("tool", "config_factory"),
    [
        ("skill_builder", lambda: UserConfig({})),
        ("skill_manager", SkillSuggestionConfig),
    ],
)
def test_unknown_action_returns_valid_ladder(skills_db, db, tool, config_factory):
    """An unknown action on either tool returns ``code=unknown-action`` with a
    ``valid:`` ladder of the real action names so a weak model self-corrects."""
    mp = _mp_for(config_factory(), db)
    out = ToolDispatcher(mp).dispatch(
        tool, {"action": "frobnicate", "act_summary": "x"}
    )
    assert f"[{tool}(status=error, code=unknown-action" in out
    assert "code=error]" not in out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    for action in ("create", "edit", "delete", "list"):
        assert action in valid_line
    assert f"[end:{tool}]" in out


# ── both names still registered, each with its own metadata ────────────────────


def test_both_tool_names_registered_with_metadata():
    """The merge keeps two registry entries — ``skill_builder`` (user-facing) and
    ``skill_manager`` (SYSTEM) — each resolving to a real ability with stable
    name, a non-empty summary, and 6+ examples."""
    builder = AbilityRegistry.get("skill_builder")
    manager = AbilityRegistry.get("skill_manager")

    assert builder.get_name() == "skill_builder"
    assert manager.get_name() == "skill_manager"
    assert builder.get_summary()
    assert manager.get_summary()
    assert len(builder.get_examples()) >= 6
    assert len(manager.get_examples()) >= 6
    # skill_manager is the SYSTEM variant (policy-bypass identity).
    assert getattr(manager, "SYSTEM", False) is True
    assert getattr(builder, "SYSTEM", False) is False


# ── create round-trips through the real DB and writes the YAML file ────────────


def test_create_persists_row_and_writes_yaml_and_records_trail(skills_db, db):
    """create writes a real DB row, a real YAML file, and the dispatcher records
    the success envelope on the real act-trail — the full downstream chain."""
    mp = _mp_for(UserConfig({}), db)
    out = ToolDispatcher(mp).dispatch(
        "skill_builder",
        {
            "action": "create",
            "title": "Track Flights",
            "use_for": "tracking flight status and delays",
            "content": "1. Use `search` to find the flight.\n2. Use `memory` to recall the route.",
            "tags": "travel, flights",
            "act_summary": "x",
        },
    )
    assert _head(out, "skill_builder").startswith("[skill_builder(status=success")

    row = _fetch_content(skills_db["db_path"], "Track Flights")
    assert row is not None and "search" in row

    yaml_files = list(skills_db["yaml_dir"].glob("*.yaml"))
    assert len(yaml_files) == 1
    assert "Track Flights" in yaml_files[0].read_text(encoding="utf-8")

    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert any("[skill_builder(status=success" in row["result"] for row in trail)

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the search_files tool's ToolResult contract (TKT-907).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``SearchFilesAbility``, the real ``ToolDispatcher._prevalidate`` ACTION_REQUIRED
pre-gate, the real ``PolicyManager.wrap`` gate reading the real ``policy`` table
(``search_files.glob`` / ``search_files.grep`` are seeded ``allow`` on the chat
channel the ``UserConfig`` mp routes through), the real ``SearchFilesAbility.run``
walking a real ``tmp_path`` file tree, and the real ``ActTrail`` write.

The file trees are real files written under ``tmp_path``; the walk, the glob
match, the grep regex, and the truncation guards all run for real.

What TKT-907 changes, exercised end to end:

* **Errors stop masquerading as success:** unknown action / bad max_files /
  invalid regex no longer ship as ``status=success`` prose; they carry a stable
  kebab ``code`` (``unknown-action`` / ``invalid-param`` / ``invalid-regex``).
* **Missing query is pre-gated:** the ACTION_REQUIRED map turns a missing
  ``query`` into a ``code=missing-params`` error BEFORE the policy gate and
  before run().
* **Structured bodies:** glob → ``{"files": [...]}`` rows; grep → ``{file, line,
  text[, context]}`` match rows the model feeds straight into read/file tools —
  with ``meta count=<n>``.
* **Truncation is never silent:** a cap that bit (more matches than max_files OR
  the walk budget expiring) surfaces as ``meta truncated=true``.
* **Zero hits is SUCCESS:** ``count=0`` + an empty row set + a broaden
  suggestion in the body, not an error.

RED-before-change: against the pre-TKT-907 ability the unknown-action,
bad-max_files, and invalid-regex cases all render ``status=success`` (the old
file returned ``ToolResult.ok(...)`` on every error path); the glob/grep bodies
were newline-joined prose, not JSON rows; and a capped result carried no
``truncated`` meta. These tests fail against that ability.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "find the config file"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the policy channel) and ``uid`` (the transcript anchor
    the trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat mp: ``search_files.glob`` / ``search_files.grep`` are seeded
    ``allow`` on the chat channel ``UserConfig`` routes through, so the real
    policy gate passes and run() executes without a WebSocket ask-hang."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _body_text(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:search_files]")
    return rendered[head:tail]


def _parse_body(rendered: str) -> object:
    return json.loads(_body_text(rendered))


# ── 1. unknown action → code=unknown-action + valid ladder ──────────────────────


def test_unknown_action_lists_valid_actions(db, chat_mp):
    """An unknown action errors ``code=unknown-action`` whose ``valid:`` line names
    glob + grep — not a ``status=success`` prose body (the old shape). The
    ACTION_REQUIRED pre-gate fires before the policy gate."""
    out = ToolDispatcher(chat_mp).dispatch(
        "search_files", {"action": "teleport", "query": "*.py"}
    )

    assert "[search_files(status=error, code=unknown-action" in out
    assert "status=success" not in out
    assert "valid: glob | grep" in out


# ── 2. missing query → pre-gate missing-params ──────────────────────────────────


def test_missing_query_is_pre_gated_missing_params(db, chat_mp):
    """``glob`` with no ``query`` is pre-gated by ACTION_REQUIRED into a
    ``code=missing-params`` error BEFORE run() and before the policy gate."""
    out = ToolDispatcher(chat_mp).dispatch("search_files", {"action": "glob"})

    assert "[search_files(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert "query" in out


# ── 3. bad max_files → kebab-coded invalid-param ────────────────────────────────


def test_bad_max_files_returns_invalid_param(db, chat_mp, tmp_path):
    """A max_files outside 1..200 errors ``code=invalid-param`` naming the bound —
    not a ``status=success`` prose body."""
    (tmp_path / "a.py").write_text("x")
    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "glob", "query": "*.py", "directory": str(tmp_path), "max_files": 0},
    )

    assert "[search_files(status=error, code=invalid-param" in out
    assert "status=success" not in out
    assert "between 1 and 200" in out


# ── 4. invalid regex → code=invalid-regex ───────────────────────────────────────


def test_invalid_regex_returns_invalid_regex(db, chat_mp, tmp_path):
    """A malformed grep regex errors ``code=invalid-regex`` — the ONLY caught
    error in run(); not a success body and not the broad catch-all the old file
    had."""
    (tmp_path / "a.py").write_text("hello\n")
    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "grep", "query": "[invalid", "directory": str(tmp_path)},
    )

    assert "[search_files(status=error, code=invalid-regex" in out
    assert "status=success" not in out


# ── 5. glob happy path → structured files rows + count ──────────────────────────


def test_glob_happy_path_returns_file_rows(db, chat_mp, tmp_path):
    """``glob`` returns a structured ``{"files": [...]}`` body of absolute paths and
    a ``count`` meta — not a newline-joined prose blob (the old shape)."""
    (tmp_path / "alpha.py").write_text("# a")
    (tmp_path / "beta.txt").write_text("# b")
    (tmp_path / "gamma.py").write_text("# c")

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "glob", "query": "*.py", "directory": str(tmp_path)},
    )

    assert "[search_files(status=success" in out
    assert "count=2" in out
    body = _parse_body(out)
    assert isinstance(body, dict)
    names = sorted(p.rsplit("/", 1)[-1] for p in body["files"])
    assert names == ["alpha.py", "gamma.py"]
    assert all(p.startswith("/") for p in body["files"])


# ── 6. grep happy path → match rows with file/line/text + count ─────────────────


def test_grep_happy_path_returns_match_rows(db, chat_mp, tmp_path):
    """``grep`` returns one structured row per matched line carrying file/line/text
    — exactly the shape the model feeds into read — with a ``count`` meta."""
    content = "\n".join(f"line {i}" for i in range(1, 12))
    (tmp_path / "a.py").write_text(content)

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "grep", "query": "line 6", "directory": str(tmp_path)},
    )

    assert "[search_files(status=success" in out
    assert "count=1" in out
    rows = _parse_body(out)
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert row["file"].endswith("a.py")
    assert row["file"].startswith("/")
    assert row["line"] == 6
    assert row["text"] == "line 6"


# ── 7. context_lines preserved on the rows ──────────────────────────────────────


def test_grep_context_lines_carried_on_rows(db, chat_mp, tmp_path):
    """``context_lines`` survives the contract: each match row carries a
    ``context`` field with the surrounding lines."""
    content = "\n".join(f"line {i}" for i in range(1, 12))
    (tmp_path / "a.py").write_text(content)

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {
            "action": "grep",
            "query": "line 6",
            "directory": str(tmp_path),
            "context_lines": 2,
        },
    )

    rows = _parse_body(out)
    ctx = rows[0]["context"]
    assert "line 4" in ctx
    assert "line 6" in ctx
    assert "line 8" in ctx


# ── 8. zero hits → SUCCESS, count=0, broaden suggestion (NOT an error) ──────────


def test_glob_zero_hits_is_success_with_broaden_suggestion(db, chat_mp, tmp_path):
    """Zero glob hits is a SUCCESS with ``count=0``, empty rows, and a broaden
    suggestion in the body — not an error (the model must not treat 'nothing
    matched' as a failure to retry forever)."""
    (tmp_path / "a.txt").write_text("x")

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "glob", "query": "*.nonexistent", "directory": str(tmp_path)},
    )

    assert "[search_files(status=success" in out
    assert "count=0" in out
    assert "code=" not in out.split("\n", 1)[0]
    body = _parse_body(out)
    assert body["files"] == []
    assert "broaden" in body["note"].lower()


def test_grep_zero_hits_is_success_with_broaden_suggestion(db, chat_mp, tmp_path):
    """Zero grep hits is a SUCCESS with ``count=0`` and a broaden suggestion — not
    an error."""
    (tmp_path / "a.py").write_text("nothing here\n")

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "grep", "query": "NONEXISTENT", "directory": str(tmp_path)},
    )

    assert "[search_files(status=success" in out
    assert "count=0" in out
    body = _parse_body(out)
    assert body["matches"] == []
    assert "broaden" in body["note"].lower()


# ── 9. truncation → meta truncated=true when the max_files cap bites ────────────


def test_glob_truncation_flags_truncated_meta(db, chat_mp, tmp_path):
    """More matches than ``max_files`` surfaces ``meta truncated=true`` — the cut is
    never silent (the old file dropped the overflow with no signal)."""
    for i in range(15):
        (tmp_path / f"f{i:02d}.dat").write_text("x")

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {
            "action": "glob",
            "query": "*.dat",
            "directory": str(tmp_path),
            "max_files": 5,
        },
    )

    assert "[search_files(status=success" in out
    assert "truncated=true" in out
    assert "count=5" in out
    body = _parse_body(out)
    assert len(body["files"]) == 5


def test_grep_truncation_flags_truncated_meta(db, chat_mp, tmp_path):
    """The grep file cap biting surfaces ``meta truncated=true`` on the success
    envelope."""
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("NEEDLE\n")

    out = ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {
            "action": "grep",
            "query": "NEEDLE",
            "directory": str(tmp_path),
            "max_files": 3,
        },
    )

    assert "[search_files(status=success" in out
    assert "truncated=true" in out
    body = _parse_body(out)
    files = {r["file"] for r in body["matches"]}
    assert len(files) == 3


# ── 10. act-trail records the rendered envelope ─────────────────────────────────


def test_act_trail_records_the_envelope(db, chat_mp, tmp_path):
    """The act-trail records the same rendered search_files envelope against the
    transcript anchor — the cross-step write really lands."""
    (tmp_path / "a.py").write_text("# a")
    ToolDispatcher(chat_mp).dispatch(
        "search_files",
        {"action": "glob", "query": "*.py", "directory": str(tmp_path)},
    )

    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("[search_files(status=" in row["result"] for row in trail)

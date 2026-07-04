# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests that drive the real ToolDispatcher end-to-end hot path for file_write
with zero mocks: read-guard, empty-contents handling, and structured
success/error bodies."""

import os
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


class _MP:

    def __init__(self, uid: int, turn_id: int, config: object) -> None:
        self.config = config
        self.uid = uid
        self._uid = uid
        self.turn_id = turn_id


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> _MP:
    allow_policy(db, "file_write")
    # Seed under the real UserConfig transcript channel ("user"), not "chat":
    # the read-guard joins tool_calls→transcript on (channel, turn_id), so the
    # input row and the read rows must share the config's own channel.
    uid = seed_transcript(db, "user", "save this to a file")
    turn_id = cast("int", db.execute("SELECT turn_id FROM transcript WHERE id = ?", (uid,)).fetchone()[0])
    return _MP(uid, turn_id, UserConfig({}))


def _parse_body(rendered: str) -> object:
    return parse_body(rendered, "file_write")


# ── 1. THE KILL SHOT: empty contents writes a real 0-byte file ──────────────────



def test_empty_contents_writes_zero_byte_file(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": ""}
    )

    assert "[file_write(status=success" in out
    body = cast("dict[str, object]", _parse_body(out))
    assert body["bytes"] == 0
    assert body["created"] is True
    assert body["path"] == str(target.resolve())
    # The real file exists on disk with size 0.
    assert target.exists()
    assert target.stat().st_size == 0


# ── 2. happy-path new file → created=true, bytes==len, content round-trips ──────


def test_new_file_written_and_round_trips(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    target = tmp_path / "sub" / "note.txt"
    text = "hello chalie line one\nand a second line"

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": text}
    )

    assert "[file_write(status=success" in out
    body = cast("dict[str, object]", _parse_body(out))
    assert body["created"] is True
    assert body["bytes"] == len(text.encode("utf-8"))
    assert target.read_text(encoding="utf-8") == text


# ── 3. overwrite WITH a prior read → success, created=false ─────────────────────


def test_overwrite_with_prior_read_succeeds(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """An existing file plus a prior ``read`` call on the same resolved path in
    the transcript clears the read-guard: the write succeeds with
    ``created=false``.

    The prior ``read`` tool_calls row is written through ``ActTrail().record`` —
    the SAME production write path the dispatcher uses to record every tool call,
    keyed to the same ``transcript_id`` the read-guard queries (``mp._uid``). We
    do not drive a literal ``read`` dispatch here only because the ``read`` tool
    carries its own network/SSRF surface; the act-trail row it would produce is
    reproduced through the real repository, not mocked."""
    target = tmp_path / "existing.txt"
    target.write_text("old content\n", encoding="utf-8")

    ActTrail().record(
        tool_name="read",
        params={"source": str(target)},
        result="[read(status=success)]\nold content\n[end:read]",
        transcript_id=chat_mp._uid,
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": "new content"}
    )

    assert "[file_write(status=success" in out
    body = cast("dict[str, object]", _parse_body(out))
    assert body["created"] is False
    assert target.read_text(encoding="utf-8") == "new content"


# ── 4. overwrite WITHOUT a prior read → read-required, file UNCHANGED ────────────


def test_overwrite_without_prior_read_is_read_required(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """Overwriting an existing file with NO prior ``read`` on the path is
    ``code=read-required`` (not a ``status=success`` prose body) and leaves the
    file content untouched on disk."""
    target = tmp_path / "existing.txt"
    target.write_text("original\n", encoding="utf-8")

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": "clobbered\n"}
    )

    assert "[file_write(status=error, code=read-required" in out
    assert "status=success" not in out
    # The hard guarantee: the file was NOT overwritten.
    assert target.read_text(encoding="utf-8") == "original\n"


# ── 5. missing contents key (from run(), not the pre-gate) → missing-params ─────


def test_missing_contents_key_is_missing_params(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """A MISSING ``contents`` key (not pre-gated, since an empty string is valid)
    is guarded by run() as ``code=missing-params`` — not a success body."""
    target = tmp_path / "x.txt"

    out = ToolDispatcher(chat_mp).dispatch("file_write", {"path": str(target)})

    assert "[file_write(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert not target.exists()


# ── 6. relative path → invalid-path ─────────────────────────────────────────────


def test_relative_path_is_invalid_path(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A non-absolute path errors ``code=invalid-path`` — the tool requires an
    absolute path, not a success body."""
    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": "relative/note.txt", "contents": "x"}
    )

    assert "[file_write(status=error, code=invalid-path" in out
    assert "status=success" not in out


# ── 7. permission denied → permission-denied ────────────────────────────────────


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses unix file permissions, so the chmod 0o500 dir is writable",
)
def test_permission_denied_into_readonly_dir(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        target = locked / "denied.txt"
        out = ToolDispatcher(chat_mp).dispatch(
            "file_write", {"path": str(target), "contents": "x"}
        )

        assert "[file_write(status=error, code=permission-denied" in out
        assert "status=success" not in out
    finally:
        # Restore so tmp_path teardown can remove the tree.
        locked.chmod(0o700)


# ── 8. REGRESSION: read recorded under a later STEP row (not the input row) ──────


def test_overwrite_with_read_under_later_step_row_succeeds(
    db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path
) -> None:
    """The read-guard keys on the TURN, not the input row. A real multi-step turn
    records each ``read`` against the assistant STEP row (``mp.anchor``), whose
    transcript_id is NOT the input row's ``uid``. Before this fix the guard queried
    ``transcript_id = _uid`` only, so a read on any step after the first never
    satisfied it and the model looped forever on ``read-required``."""
    target = tmp_path / "existing.txt"
    target.write_text("old content\n", encoding="utf-8")

    # A later assistant step row in the SAME turn (channel + turn_id), exactly as
    # MessageProcessor._store_row writes before that step's tools dispatch.
    step_row = db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id) VALUES ('user', 'assistant', '', ?)",
        (chat_mp.turn_id,),
    ).lastrowid
    db.commit()
    ActTrail().record(
        tool_name="read",
        params={"source": str(target)},
        result="[read(status=success)]\nold content\n[end:read]",
        transcript_id=step_row,  # NOT chat_mp._uid — the bug was querying only _uid
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": "new content"}
    )

    assert "[file_write(status=success" in out
    assert target.read_text(encoding="utf-8") == "new content"


# ── 9. read recorded with a ``~`` path clears the guard for the resolved target ──


def test_read_recorded_with_tilde_path_clears_guard(
    db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``read`` expands ``~`` before reading but records the raw ``~/…`` arg. The
    guard must expanduser too, or a tilde-path read can never match the resolved
    absolute target file_write is called with."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "noted.txt"
    target.write_text("old\n", encoding="utf-8")

    ActTrail().record(
        tool_name="read",
        params={"source": "~/noted.txt"},
        result="[read(status=success)]\nold\n[end:read]",
        transcript_id=chat_mp._uid,
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": "new"}
    )

    assert "[file_write(status=success" in out
    assert target.read_text(encoding="utf-8") == "new"


# ── 10. LOOP-GUARD: a second identical erroring call is steered, the first is not ─


def test_repeat_identical_error_injects_loop_guard(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """Two identical calls that both error with the same code: the first carries no
    steer, the second carries the loop-guard instruction (re-check schema via
    find_tools, then escalate to the user). Drives the real dispatcher record→
    fetch_by_turn path with zero mocks."""
    args = {"path": "relative/note.txt", "contents": "x"}  # relative → deterministic invalid-path

    first = ToolDispatcher(chat_mp).dispatch("file_write", dict(args))
    second = ToolDispatcher(chat_mp).dispatch("file_write", dict(args))

    assert "[file_write(status=error, code=invalid-path" in first
    assert "loop-guard" not in first
    assert "[file_write(status=error, code=invalid-path" in second
    assert "loop-guard" in second
    assert "find_tools" in second

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the file_write tool's ToolResult contract (TKT-908).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``FileWriteAbility``, the real ``ToolDispatcher._prevalidate`` ACTION_REQUIRED
pre-gate, the real ``PolicyManager.wrap`` gate reading the real ``policy`` table
(``file_write`` is seeded ``ask``; the fixture flips the real row to ``allow`` on
the chat channel ``UserConfig`` routes through, exactly as a human "always allow"
does, so the headless test does not hang on a WebSocket ask), the real
``FileWriteAbility.run`` writing a real file under ``tmp_path``, the real
act-trail read-guard query over the real ``tool_calls`` table, and the real
``ActTrail`` write.

What TKT-908 changes, exercised end to end:

* **Empty contents is valid user data:** ``contents=""`` writes a real 0-byte
  file and echoes ``{"path", "bytes": 0, "created": true}`` — it is no longer
  rejected (the old code's ``if not contents:`` truthiness guard returned an
  ``ok()`` "Error: 'contents' is required." prose body).
* **Errors stop masquerading as success:** read-required / invalid-path /
  permission-denied / missing-params no longer ship as ``status=success`` prose;
  they carry a stable kebab ``code``.
* **Structured success body:** ``{"path", "bytes", "created"}`` — not a
  ``json.dumps`` string with a redundant ``"success": True`` key.
* **Read-guard fires as an error code:** overwriting an existing file with no
  prior ``read`` call is ``code=read-required`` and leaves the file UNCHANGED.

RED-before-change: against the pre-TKT-908 ability the empty-contents case
renders ``status=success`` carrying "Error: 'contents' is required." prose (and
no file is written); the success body is a stringified
``{"success": true, ..., "file_size": ...}`` blob with no ``created`` key; and
the read-required / relative-path / permission paths all render
``status=success``. These tests fail against that ability.
"""

import json
import os

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "save this to a file"),
    )
    db.commit()
    return cur.lastrowid


def _allow_write(db, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so ``file_write`` is ``allow`` on the
    channel — the same row the real ``PolicyManager`` gate reads. ``file_write``
    ships as ``ask`` by seed; on a headless test channel an ``ask`` would block
    waiting for a human POST. Flipping the real row to ``allow`` (exactly what a
    user does when they pick "always allow") lets the gate pass through to the
    production ``run()``. No mock — this is the production policy store."""
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) "
        "VALUES (?, ?, 'allow')",
        (channel, "file_write"),
    )
    db.commit()


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch + the read-guard
    read off the live processor: ``config`` (the chat policy channel), ``uid``
    (the act-trail write anchor) and ``_uid`` (the read-guard's transcript
    anchor). Production binds all three to the same transcript id; the fixture
    mirrors that."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid
        self._uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat mp bound to the test database: a seeded transcript anchor for
    the act-trail + read-guard, and ``file_write`` flipped to ``allow`` in the
    real policy table so the gate passes through to the production run()."""
    _allow_write(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _body_text(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:file_write]")
    return rendered[head:tail]


def _parse_body(rendered: str) -> object:
    return json.loads(_body_text(rendered))


# ── 1. THE KILL SHOT: empty contents writes a real 0-byte file ──────────────────


def test_empty_contents_writes_zero_byte_file(db, chat_mp, tmp_path):
    """``contents=""`` to a NEW path is VALID user data: a real 0-byte file is
    written and the body echoes ``{"path", "bytes": 0, "created": true}``. The old
    ``if not contents:`` truthiness guard rejected this as a ``status=success``
    "Error: 'contents' is required." prose body and wrote nothing."""
    target = tmp_path / "empty.txt"

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": ""}
    )

    assert "[file_write(status=success" in out
    body = _parse_body(out)
    assert body["bytes"] == 0
    assert body["created"] is True
    assert body["path"] == str(target.resolve())
    # The real file exists on disk with size 0.
    assert target.exists()
    assert target.stat().st_size == 0


# ── 2. happy-path new file → created=true, bytes==len, content round-trips ──────


def test_new_file_written_and_round_trips(db, chat_mp, tmp_path):
    """A non-empty write to a new path returns ``created=true`` and ``bytes`` equal
    to the encoded length, and the file content round-trips from disk.

    The dispatcher's real ``_sanitize_llm_args`` strips surrounding whitespace off
    every string arg before run() (a production sentinel-scrub step), so the body
    used here is whitespace-clean and what lands on disk is byte-identical to it —
    we assert against the genuine post-sanitisation value, not a pre-sanitisation
    assumption."""
    target = tmp_path / "sub" / "note.txt"
    text = "hello chalie line one\nand a second line"

    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": str(target), "contents": text}
    )

    assert "[file_write(status=success" in out
    body = _parse_body(out)
    assert body["created"] is True
    assert body["bytes"] == len(text.encode("utf-8"))
    assert target.read_text(encoding="utf-8") == text


# ── 3. overwrite WITH a prior read → success, created=false ─────────────────────


def test_overwrite_with_prior_read_succeeds(db, chat_mp, tmp_path):
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
    body = _parse_body(out)
    assert body["created"] is False
    assert target.read_text(encoding="utf-8") == "new content"


# ── 4. overwrite WITHOUT a prior read → read-required, file UNCHANGED ────────────


def test_overwrite_without_prior_read_is_read_required(db, chat_mp, tmp_path):
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


def test_missing_contents_key_is_missing_params(db, chat_mp, tmp_path):
    """A MISSING ``contents`` key (not pre-gated, since an empty string is valid)
    is guarded by run() as ``code=missing-params`` — not a success body."""
    target = tmp_path / "x.txt"

    out = ToolDispatcher(chat_mp).dispatch("file_write", {"path": str(target)})

    assert "[file_write(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert not target.exists()


# ── 6. missing path → pre-gate missing-params, recorded on the act-trail ────────


def test_missing_path_is_pre_gated_missing_params(db, chat_mp):
    """A missing ``path`` is pre-gated by ACTION_REQUIRED into a
    ``code=missing-params`` error BEFORE run() and the policy gate, and the
    rejection is recorded on the act-trail against the transcript anchor."""
    out = ToolDispatcher(chat_mp).dispatch("file_write", {"contents": "x"})

    assert "[file_write(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert "path" in out

    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any(
        "[file_write(status=error, code=missing-params" in row["result"]
        for row in trail
    )


# ── 7. relative path → invalid-path ─────────────────────────────────────────────


def test_relative_path_is_invalid_path(db, chat_mp):
    """A non-absolute path errors ``code=invalid-path`` — the tool requires an
    absolute path, not a success body."""
    out = ToolDispatcher(chat_mp).dispatch(
        "file_write", {"path": "relative/note.txt", "contents": "x"}
    )

    assert "[file_write(status=error, code=invalid-path" in out
    assert "status=success" not in out


# ── 8. permission denied → permission-denied ────────────────────────────────────


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses unix file permissions, so the chmod 0o500 dir is writable",
)
def test_permission_denied_into_readonly_dir(db, chat_mp, tmp_path):
    """Writing a NEW file into a 0o500 (read+execute, no write) directory errors
    ``code=permission-denied`` — the real ``open`` raises ``PermissionError`` and
    is mapped, not swallowed as success."""
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

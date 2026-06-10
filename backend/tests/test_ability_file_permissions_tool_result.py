# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the file_permissions tool's ToolResult contract (TKT-909).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``FilePermissionsAbility``, the real ``ToolDispatcher._prevalidate``
ACTION_REQUIRED pre-gate, the real ``PolicyManager.wrap`` gate reading the real
``policy`` table (``file_permissions`` is seeded ``ask``; the fixture flips the
real row to ``allow`` on the chat channel ``UserConfig`` routes through, exactly
as a human "always allow" does, so the headless test does not hang on a
WebSocket ask), the real ``FilePermissionsAbility.run`` chmod-ing a real file
under ``tmp_path``, and the real ``ActTrail`` write.

What TKT-909 changes, exercised end to end:

* **Intent / symbolic forms are coherent with the examples:** the examples teach
  "make this script executable" / "read-only" / "owner-only", and the validator
  now accepts ``"+x"``, ``"readonly"``, ``"owner-only"`` and chmod-style clauses
  alongside octal. The old code accepted ONLY 3-4 octal digits, so ``"+x"``
  rendered ``{"error": "invalid-permissions"}`` as ``status=success``.
* **Errors stop masquerading as success:** invalid-mode / path-not-found /
  permission-denied / missing-params carry a stable kebab ``code`` on
  ``status=error`` — not a ``ToolResult.ok(json.dumps({"error": ...}))`` body.
* **Structured success body:** ``{"path", "mode_octal", "mode_symbolic"}`` with
  ``meta previous`` carrying the pre-chmod octal — not a stringified blob with a
  redundant ``"status": "success"`` key.

RED-before-change: against the pre-TKT-909 ability the ``"+x"`` /
``"readonly"`` / ``"owner-only"`` / symbolic cases render
``[file_permissions(status=success)]`` carrying ``{"error":
"invalid-permissions", ...}`` (and chmod nothing); the success body is a
stringified ``{"status": "success", "permissions_before": ...,
"permissions_after": ...}`` blob with no ``mode_octal`` / ``mode_symbolic``
keys and no ``previous`` meta; and the error paths all render
``status=success``. These tests fail against that ability.
"""

import json
import os
import stat

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "make this script executable"),
    )
    db.commit()
    return cur.lastrowid


def _allow_chmod(db, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so ``file_permissions`` is ``allow`` on the
    channel — the same row the real ``PolicyManager`` gate reads.
    ``file_permissions`` ships as ``ask`` by seed; on a headless test channel an
    ``ask`` would block waiting for a human POST. Flipping the real row to
    ``allow`` (exactly what a user does when they pick "always allow") lets the
    gate pass through to the production ``run()``. No mock — this is the
    production policy store."""
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) "
        "VALUES (?, ?, 'allow')",
        (channel, "file_permissions"),
    )
    db.commit()


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the chat policy channel) and ``uid`` (the act-trail
    write anchor). Production binds both to the same transcript id; the fixture
    mirrors that."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid
        self._uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat mp bound to the test database: a seeded transcript anchor for
    the act-trail, and ``file_permissions`` flipped to ``allow`` in the real
    policy table so the gate passes through to the production run()."""
    _allow_chmod(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _body_text(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:file_permissions]")
    return rendered[head:tail]


def _parse_body(rendered: str) -> object:
    return json.loads(_body_text(rendered))


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ── 1. THE KILL SHOT: '+x' on a 0o644 file → 0o755, structured body ─────────────


def test_symbolic_plus_x_makes_executable(db, chat_mp, tmp_path):
    """``permissions="+x"`` on a real 0o644 file chmods it to 0o755 (absolute bit
    ops — umask-independent), and the structured body carries
    ``mode_octal="0755"`` / ``mode_symbolic="rwxr-xr-x"`` with ``meta previous``
    showing 0644. The old validator accepted ONLY octal, so this rendered
    ``{"error": "invalid-permissions"}`` as ``status=success`` and chmod-ed
    nothing."""
    target = tmp_path / "deploy.sh"
    target.write_text("#!/bin/bash\necho hi\n")
    os.chmod(target, 0o644)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": "+x"}
    )

    assert "[file_permissions(status=success" in out
    assert "previous=0644" in out
    body = _parse_body(out)
    assert body["path"] == str(target.resolve())
    assert body["mode_octal"] == "0755"
    assert body["mode_symbolic"] == "rwxr-xr-x"
    assert _mode(target) == 0o755


# ── 2. intent keyword 'readonly' on a 0o755 file → 0o444 ────────────────────────


def test_intent_readonly_strips_write_and_execute(db, chat_mp, tmp_path):
    """``permissions="readonly"`` (the examples' own vocabulary) on a 0o755 file
    chmods it to 0o444 with ``mode_symbolic="r--r--r--"``."""
    target = tmp_path / "config.yaml"
    target.write_text("k: v\n")
    os.chmod(target, 0o755)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": "readonly"}
    )

    assert "[file_permissions(status=success" in out
    body = _parse_body(out)
    assert body["mode_octal"] == "0444"
    assert body["mode_symbolic"] == "r--r--r--"
    assert _mode(target) == 0o444


# ── 3. intent keyword 'owner-only' on a 0o666 file → 0o600 ──────────────────────


def test_intent_owner_only_strips_group_and_other(db, chat_mp, tmp_path):
    """``permissions="owner-only"`` (= ``go-rwx``) on a 0o666 file strips the
    group/other bits, leaving 0o600 with ``mode_symbolic="rw-------"``."""
    target = tmp_path / "id_rsa"
    target.write_text("KEY\n")
    os.chmod(target, 0o666)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": "owner-only"}
    )

    assert "[file_permissions(status=success" in out
    body = _parse_body(out)
    assert body["mode_octal"] == "0600"
    assert body["mode_symbolic"] == "rw-------"
    assert _mode(target) == 0o600


# ── 4. octal still works (bare + leading-zero), body keys exactly per spec ───────


@pytest.mark.parametrize("perm", ["600", "0600"])
def test_octal_still_works(db, chat_mp, tmp_path, perm):
    """``"600"`` and ``"0600"`` both chmod to 0o600, and the success body carries
    EXACTLY ``{"path", "mode_octal", "mode_symbolic"}`` with ``meta previous``."""
    target = tmp_path / "key.pem"
    target.write_text("x")
    os.chmod(target, 0o644)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": perm}
    )

    assert "[file_permissions(status=success" in out
    assert "previous=0644" in out
    body = _parse_body(out)
    assert set(body.keys()) == {"path", "mode_octal", "mode_symbolic"}
    assert body["mode_octal"] == "0600"
    assert body["mode_symbolic"] == "rw-------"
    assert _mode(target) == 0o600


# ── 5. compound symbolic clauses: 'u+x,go-rwx' from 0o644 → 0o700 ───────────────


def test_compound_symbolic_clauses(db, chat_mp, tmp_path):
    """``"u+x,go-rwx"`` resolves both clauses against the current 0o644 mode:
    ``u+x`` → owner rwx, ``go-rwx`` → group/other cleared, leaving 0o700."""
    target = tmp_path / "tool"
    target.write_text("x")
    os.chmod(target, 0o644)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": "u+x,go-rwx"}
    )

    assert "[file_permissions(status=success" in out
    body = _parse_body(out)
    assert body["mode_octal"] == "0700"
    assert body["mode_symbolic"] == "rwx------"
    assert _mode(target) == 0o700


# ── 6. unparseable permissions → invalid-mode, file UNCHANGED ───────────────────


@pytest.mark.parametrize("bad", ["banana", "u=g"])
def test_invalid_mode_leaves_file_unchanged(db, chat_mp, tmp_path, bad):
    """An unparseable value (gibberish ``"banana"`` and the unsupported copy-form
    ``"u=g"``) errors ``code=invalid-mode`` with a hint showing BOTH accepted
    forms, and the file's mode on disk is UNCHANGED."""
    target = tmp_path / "f.txt"
    target.write_text("x")
    os.chmod(target, 0o644)

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target), "permissions": bad}
    )

    assert "[file_permissions(status=error, code=invalid-mode" in out
    assert "status=success" not in out
    # The hint shows both accepted forms.
    assert "octal" in out and "symbolic" in out
    # The file was not touched.
    assert _mode(target) == 0o644


# ── 7. path-not-found → path-not-found ──────────────────────────────────────────


def test_path_not_found(db, chat_mp, tmp_path):
    """A non-existent target errors ``code=path-not-found`` — not a success
    body."""
    missing = tmp_path / "nope.txt"

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(missing), "permissions": "644"}
    )

    assert "[file_permissions(status=error, code=path-not-found" in out
    assert "status=success" not in out


# ── 8. missing permissions → pre-gate missing-params, recorded on act-trail ─────


def test_missing_permissions_is_pre_gated_missing_params(db, chat_mp, tmp_path):
    """A missing ``permissions`` is pre-gated by ACTION_REQUIRED into a
    ``code=missing-params`` error BEFORE run() and the policy gate, and the
    rejection is recorded on the act-trail against the transcript anchor."""
    target = tmp_path / "f.txt"
    target.write_text("x")

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": str(target)}
    )

    assert "[file_permissions(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert "permissions" in out

    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any(
        "[file_permissions(status=error, code=missing-params" in row["result"]
        for row in trail
    )


# ── 9. permission-denied: chmod a root-owned path as a non-root user ────────────


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root may chmod any path, so the PermissionError never fires",
)
def test_permission_denied_on_root_owned_path(db, chat_mp):
    """chmod-ing a root-owned path (``/``) as a non-root user makes the real
    ``os.chmod`` raise ``PermissionError`` → ``code=permission-denied`` (not
    swallowed as success). Verified against the live OS first so the assertion is
    grounded, not assumed."""
    try:
        os.chmod("/", 0o755)
        pytest.skip("chmod('/') did not raise — cannot exercise permission-denied")
    except PermissionError:
        pass
    except OSError:
        pytest.skip("chmod('/') raised a non-PermissionError OSError")

    out = ToolDispatcher(chat_mp).dispatch(
        "file_permissions", {"path": "/", "permissions": "755"}
    )

    assert "[file_permissions(status=error, code=permission-denied" in out
    assert "status=success" not in out

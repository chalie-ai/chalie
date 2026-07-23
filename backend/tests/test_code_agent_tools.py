# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — code_agent's file toolset (replace_all,
manage_files, move, run_script).

These run the REAL ``Ability.run()`` implementations against a real
``tmp_path`` directory and, for ``run_script``, a REAL Deno subprocess.
No mocks of any in-process behavior anywhere in this file.

All tools take ABSOLUTE paths; there is no containment, so tests point
tools directly at ``tmp_path`` — NO patching, NO mocks, NO fixture that
swaps ``FileMapperService``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from abilities.manage_files import ManageFilesAbility
from abilities.move import MoveAbility
from abilities.replace_all import ReplaceAllAbility
from abilities.run_script import RunScriptAbility
from configs.enums.param_key import Keys
from contracts.params.manage_files_params_bag import ManageFilesParamsBag

pytestmark = pytest.mark.unit

_HAS_DENO = shutil.which("deno") is not None


# ── replace_all ──────────────────────────────────────────────────────────────

def test_replace_all_directory(tmp_path: Path) -> None:
    """replace_all with path= scans the tree, skips .git, rewrites
    matching files, and reports per-file counts in the body."""
    root = tmp_path
    (root / "a.ts").write_text("old_api\nold_api\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("old_api\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "c.ts").write_text("old_api\n", encoding="utf-8")

    result = ReplaceAllAbility().run({
        Keys.search: "old_api",
        Keys.replace_: "new_api",
        Keys.path: str(root),
    })

    assert result.status == "success"
    body = result.body
    assert isinstance(body, str)
    assert "a.ts: 2" in body
    assert "sub/b.txt: 1" in body
    assert ".git/c.ts" not in body

    assert (root / "a.ts").read_text(encoding="utf-8") == "new_api\nnew_api\n"
    assert (root / "sub" / "b.txt").read_text(encoding="utf-8") == "new_api\n"
    assert (root / ".git" / "c.ts").read_text(encoding="utf-8") == "old_api\n"


def test_replace_all_glob(tmp_path: Path) -> None:
    """A second run with glob='*.ts' on a fresh tree touches only .ts files."""
    root = tmp_path
    (root / "a.ts").write_text("old_api\nnew_api\n", encoding="utf-8")
    (root / "b.txt").write_text("old_api\n", encoding="utf-8")

    result = ReplaceAllAbility().run({
        Keys.search: "old_api",
        Keys.replace_: "new_api",
        Keys.glob: "*.ts",
        Keys.path: str(root),
    })

    assert result.status == "success"
    body = result.body
    assert isinstance(body, str)
    assert "a.ts: 1" in body
    assert "b.txt" not in body

    assert (root / "a.ts").read_text(encoding="utf-8") == "new_api\nnew_api\n"
    assert (root / "b.txt").read_text(encoding="utf-8") == "old_api\n"


def test_replace_all_errors(tmp_path: Path) -> None:
    """replace_all rejects a relative path and a nonexistent absolute path."""
    rel = ReplaceAllAbility().run({
        Keys.search: "x",
        Keys.replace_: "y",
        Keys.path: "main.ts",
    })
    assert rel.status == "error"
    assert rel.code == "invalid-path"

    missing = ReplaceAllAbility().run({
        Keys.search: "x",
        Keys.replace_: "y",
        Keys.path: str(tmp_path / "nope"),
    })
    assert missing.status == "error"
    assert missing.code == "not-found"


def test_replace_all_single_file(tmp_path: Path) -> None:
    """replace_all with path=FILE replaces every occurrence in that file;
    a non-matching search returns a no-occurrences message and leaves the
    file untouched."""
    target = tmp_path / "main.ts"
    target.write_text("old_api\nold_api\n", encoding="utf-8")

    result = ReplaceAllAbility().run({
        Keys.search: "old_api",
        Keys.replace_: "new_api",
        Keys.path: str(target),
    })

    assert result.status == "success"
    assert result.body == f"{target}: 2"
    assert target.read_text(encoding="utf-8") == "new_api\nnew_api\n"

    result2 = ReplaceAllAbility().run({
        Keys.search: "zzz",
        Keys.replace_: "y",
        Keys.path: str(target),
    })

    assert result2.status == "success"
    assert result2.body == "No occurrences were found."
    assert target.read_text(encoding="utf-8") == "new_api\nnew_api\n"


# ── manage_files ─────────────────────────────────────────────────────────────

def test_manage_files_create(tmp_path: Path) -> None:
    """manage_files action=create makes an empty file at the given path."""
    target = tmp_path / "new_file.txt"
    result = ManageFilesAbility().run(ManageFilesParamsBag.from_params({
        Keys.action: "create",
        Keys.path: str(target),
    }))

    assert result.status == "success"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""


def test_manage_files_delete(tmp_path: Path) -> None:
    """manage_files action=delete removes the file; missing target errors."""
    target = tmp_path / "to_delete.txt"
    target.write_text("content", encoding="utf-8")

    result = ManageFilesAbility().run(ManageFilesParamsBag.from_params({
        Keys.action: "delete",
        Keys.path: str(target),
    }))
    assert result.status == "success"
    assert not target.exists()

    missing = ManageFilesAbility().run(ManageFilesParamsBag.from_params({
        Keys.action: "delete",
        Keys.path: str(target),
    }))
    assert missing.status == "error"
    assert missing.code == "not-found"


def test_manage_files_update_permission(tmp_path: Path) -> None:
    """manage_files action=update_permission changes the file mode on disk."""
    target = tmp_path / "perm_file.txt"
    target.write_text("content", encoding="utf-8")

    result = ManageFilesAbility().run(ManageFilesParamsBag.from_params({
        Keys.action: "update_permission",
        Keys.path: str(target),
        Keys.permission_code: "0755",
    }))
    assert result.status == "success"

    mode = os.stat(target).st_mode & 0o7777
    assert mode == 0o755


# ── move ─────────────────────────────────────────────────────────────────────

def test_move_file(tmp_path: Path) -> None:
    """move relocates a real file: old path gone, new path has same content."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("hello world", encoding="utf-8")

    result = MoveAbility().run({
        Keys.current_path: str(src),
        Keys.new_path: str(dst),
    })

    assert result.status == "success"
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "hello world"


# ── run_script ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_DENO, reason="deno is not installed on this machine")
def test_run_script_success(tmp_path: Path) -> None:
    """run_script executes a real .ts file with Deno, forwards args as
    Deno.args, and captures stdout and exit_code=0."""
    script = tmp_path / "hello.ts"
    script.write_text(
        'console.log("chalie-test-marker-42", Deno.args.join("|"));\n', encoding="utf-8"
    )

    result = RunScriptAbility().run({
        Keys.path: str(script),
        Keys.args: ["alpha", "--beta=2"],
    })

    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    assert body["exit_code"] == 0
    assert "chalie-test-marker-42 alpha|--beta=2" in body["stdout"]


def test_run_script_errors(tmp_path: Path) -> None:
    """run_script refuses a non-.ts file, a relative path, and non-list
    args before any subprocess is spawned."""
    (tmp_path / "notes.txt").write_text("not a script", encoding="utf-8")

    not_ts = RunScriptAbility().run({Keys.path: str(tmp_path / "notes.txt")})
    assert not_ts.status == "error"
    assert not_ts.code == "unsupported-file-type"

    relative = RunScriptAbility().run({Keys.path: "hello.ts"})
    assert relative.status == "error"
    assert relative.code == "invalid-path"

    script = tmp_path / "ok.ts"
    script.write_text("console.log(1);\n", encoding="utf-8")
    bad_args = RunScriptAbility().run({Keys.path: str(script), Keys.args: "alpha"})
    assert bad_args.status == "error"
    assert bad_args.code == "invalid-args"

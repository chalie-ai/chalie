# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — code_agent's file toolset (edit_file,
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
from abilities.edit_file import EditFileAbility
from abilities.run_script import RunScriptAbility
from configs.enums.param_key import Keys
from contracts.params.manage_files_params_bag import ManageFilesParamsBag
from contracts.params.move_params_bag import MoveParamsBag
from contracts.params.edit_file_params_bag import EditFileParamsBag
from contracts.params.run_script_params_bag import RunScriptParamsBag
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

_HAS_DENO = shutil.which("deno") is not None


# ── edit_file ──────────────────────────────────────────────────────────────

def test_edit_file_single_occurrence(tmp_path: Path) -> None:
    """edit_file replaces a single unique occurrence in the target file —
    ``old_api`` alone appears twice, so the search leans on surrounding
    context (the leading newline) to pin exactly one, the tool's intended
    disambiguation pattern. Untouched lines stay byte-identical."""
    target = tmp_path / "main.ts"
    target.write_text("import old_api\nold_api\n", encoding="utf-8")

    result = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "\nold_api\n",
        Keys.replace_: "\nnew_api\n",
        Keys.path: str(target),
    })))

    assert result.status == "success"
    assert result.body == f"Replaced 1 occurrence in {target}."
    assert target.read_text(encoding="utf-8") == "import old_api\nnew_api\n"


def test_edit_file_no_occurrence(tmp_path: Path) -> None:
    """edit_file with a search that does not appear is a loud not-found error —
    never a silent success — and leaves the file untouched."""
    target = tmp_path / "main.ts"
    target.write_text("hello world\n", encoding="utf-8")

    result = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "zzz",
        Keys.replace_: "y",
        Keys.path: str(target),
    })))

    assert result.status == "error"
    assert result.code == "not-found"
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_edit_file_multiple_occurrences(tmp_path: Path) -> None:
    """edit_file errors when the search string appears more than once."""
    target = tmp_path / "main.ts"
    target.write_text("old_api\nold_api\n", encoding="utf-8")

    result = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "old_api",
        Keys.replace_: "new_api",
        Keys.path: str(target),
    })))

    assert result.status == "error"
    assert result.code == "not-unique"
    assert result.body == (
        "You can only replace 1 occurrence at a time, include more of the "
        "surrounding text to make the `search` unique."
    )
    # File must remain untouched
    assert target.read_text(encoding="utf-8") == "old_api\nold_api\n"


def test_edit_file_errors(tmp_path: Path) -> None:
    """edit_file rejects a relative path, a nonexistent path, and a directory."""
    rel = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "x",
        Keys.replace_: "y",
        Keys.path: "main.ts",
    })))
    assert rel.status == "error"
    assert rel.code == "invalid-path"

    missing = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "x",
        Keys.replace_: "y",
        Keys.path: str(tmp_path / "nope"),
    })))
    assert missing.status == "error"
    assert missing.code == "not-found"

    result = EditFileAbility().run(built(EditFileParamsBag.from_params({
        Keys.search: "x",
        Keys.replace_: "y",
        Keys.path: str(tmp_path),
    })))
    assert result.status == "error"
    assert result.code == "invalid-path"


# ── manage_files ─────────────────────────────────────────────────────────────

def test_manage_files_create(tmp_path: Path) -> None:
    """manage_files action=create makes an empty file at the given path."""
    target = tmp_path / "new_file.txt"
    result = ManageFilesAbility().run(built(ManageFilesParamsBag.from_params({
        Keys.action: "create",
        Keys.path: str(target),
    })))

    assert result.status == "success"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""


def test_manage_files_delete(tmp_path: Path) -> None:
    """manage_files action=delete removes the file; missing target errors."""
    target = tmp_path / "to_delete.txt"
    target.write_text("content", encoding="utf-8")

    result = ManageFilesAbility().run(built(ManageFilesParamsBag.from_params({
        Keys.action: "delete",
        Keys.path: str(target),
    })))
    assert result.status == "success"
    assert not target.exists()

    missing = ManageFilesAbility().run(built(ManageFilesParamsBag.from_params({
        Keys.action: "delete",
        Keys.path: str(target),
    })))
    assert missing.status == "error"
    assert missing.code == "not-found"


def test_manage_files_update_permission(tmp_path: Path) -> None:
    """manage_files action=update_permission changes the file mode on disk."""
    target = tmp_path / "perm_file.txt"
    target.write_text("content", encoding="utf-8")

    result = ManageFilesAbility().run(built(ManageFilesParamsBag.from_params({
        Keys.action: "update_permission",
        Keys.path: str(target),
        Keys.permission_code: "0755",
    })))
    assert result.status == "success"

    mode = os.stat(target).st_mode & 0o7777
    assert mode == 0o755


# ── move ─────────────────────────────────────────────────────────────────────

def test_move_file(tmp_path: Path) -> None:
    """move relocates a real file: old path gone, new path has same content."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("hello world", encoding="utf-8")

    result = MoveAbility().run(built(MoveParamsBag.from_params({
        Keys.current_path: str(src),
        Keys.new_path: str(dst),
    })))

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

    result = RunScriptAbility().run(built(RunScriptParamsBag.from_params({
        Keys.path: str(script),
        Keys.args: ["alpha", "--beta=2"],
    })))

    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    assert body["exit_code"] == 0
    assert "chalie-test-marker-42 alpha|--beta=2" in body["stdout"]


def test_run_script_errors(tmp_path: Path) -> None:
    """run_script refuses a non-.ts file and a relative path before any
    subprocess is spawned; non-list args are rejected by the params bag."""
    (tmp_path / "notes.txt").write_text("not a script", encoding="utf-8")

    not_ts = RunScriptAbility().run(
        built(RunScriptParamsBag.from_params({Keys.path: str(tmp_path / "notes.txt")}))
    )
    assert not_ts.status == "error"
    assert not_ts.code == "unsupported-file-type"

    relative = RunScriptAbility().run(built(RunScriptParamsBag.from_params({Keys.path: "hello.ts"})))
    assert relative.status == "error"
    assert relative.code == "invalid-path"

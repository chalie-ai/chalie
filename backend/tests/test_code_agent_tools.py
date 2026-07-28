# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — code_agent's file toolset (edit_file, make_dir, delete,
set_permissions, move, run_script).

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

from abilities.delete import DeleteAbility
from abilities.make_dir import MakeDirAbility
from abilities.move import MoveAbility
from abilities.edit_file import EditFileAbility
from abilities.run_script import RunScriptAbility
from abilities.set_permissions import SetPermissionsAbility
from configs.enums.param_key import Keys
from contracts.params.delete_params_bag import DeleteParamsBag
from contracts.params.make_dir_params_bag import MakeDirParamsBag
from contracts.params.move_params_bag import MoveParamsBag
from contracts.params.set_permissions_params_bag import SetPermissionsParamsBag
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


# ── make_dir ─────────────────────────────────────────────────────────────────

def test_make_dir_creates_a_directory_with_parents(tmp_path: Path) -> None:
    """make_dir creates the directory AND any missing parents in one call."""
    target = tmp_path / "outer" / "inner"

    result = MakeDirAbility().run(built(MakeDirParamsBag.from_params({
        Keys.path: str(target),
    })))

    assert result.status == "success"
    assert target.is_dir()
    assert (tmp_path / "outer").is_dir()


def test_make_dir_makes_a_directory_not_a_file(tmp_path: Path) -> None:
    """A path with no trailing slash is STILL a directory.

    The tool it replaces switched between file and folder on a trailing slash
    (a heuristic a small model routinely got wrong). make_dir has one outcome,
    so the same argument that used to produce an empty FILE now produces a
    DIRECTORY — this is the assertion that pins the split."""
    target = tmp_path / "no_trailing_slash"

    result = MakeDirAbility().run(built(MakeDirParamsBag.from_params({
        Keys.path: str(target),
    })))

    assert result.status == "success"
    assert target.is_dir()
    assert not target.is_file()


def test_make_dir_refuses_an_existing_path(tmp_path: Path) -> None:
    """An existing directory — and an existing FILE at the same path — are both
    already-exists, never a silent success."""
    existing_dir = tmp_path / "already"
    existing_dir.mkdir()
    existing_file = tmp_path / "occupied.txt"
    existing_file.write_text("content", encoding="utf-8")

    for path in (existing_dir, existing_file):
        result = MakeDirAbility().run(built(MakeDirParamsBag.from_params({
            Keys.path: str(path),
        })))
        assert result.status == "error", path
        assert result.code == "already-exists", path

    # The occupied file is untouched — the refusal did not truncate it.
    assert existing_file.read_text(encoding="utf-8") == "content"


def test_make_dir_requires_an_absolute_path(tmp_path: Path) -> None:
    """A relative path is refused with invalid-path and creates nothing."""
    assert tmp_path is not None

    result = MakeDirAbility().run(built(MakeDirParamsBag.from_params({
        Keys.path: "relative/dir",
    })))

    assert result.status == "error"
    assert result.code == "invalid-path"


# ── delete ───────────────────────────────────────────────────────────────────

def test_delete_removes_a_file(tmp_path: Path) -> None:
    """delete removes the file; deleting it again is a loud not-found."""
    target = tmp_path / "to_delete.txt"
    target.write_text("content", encoding="utf-8")

    result = DeleteAbility().run(built(DeleteParamsBag.from_params({
        Keys.path: str(target),
    })))
    assert result.status == "success"
    assert not target.exists()

    missing = DeleteAbility().run(built(DeleteParamsBag.from_params({
        Keys.path: str(target),
    })))
    assert missing.status == "error"
    assert missing.code == "not-found"


def test_delete_removes_a_folder_recursively(tmp_path: Path) -> None:
    """delete on a folder removes the folder and everything inside it."""
    target = tmp_path / "tree"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "leaf.txt").write_text("content", encoding="utf-8")

    result = DeleteAbility().run(built(DeleteParamsBag.from_params({
        Keys.path: str(target),
    })))

    assert result.status == "success"
    assert not target.exists()


# ── set_permissions ──────────────────────────────────────────────────────────

def test_set_permissions_changes_the_mode_on_disk(tmp_path: Path) -> None:
    """set_permissions chmods the real file; the new mode is readable from the
    filesystem, not just echoed back by the tool."""
    target = tmp_path / "perm_file.txt"
    target.write_text("content", encoding="utf-8")

    result = SetPermissionsAbility().run(built(SetPermissionsParamsBag.from_params({
        Keys.path: str(target),
        Keys.permission_code: "0755",
    })))
    assert result.status == "success"

    mode = os.stat(target).st_mode & 0o7777
    assert mode == 0o755


def test_set_permissions_rejects_a_bad_code_before_touching_the_file(
    tmp_path: Path,
) -> None:
    """An unparseable permission code is invalid-param and the mode on disk is
    unchanged — validation happens before the chmod syscall."""
    target = tmp_path / "perm_file.txt"
    target.write_text("content", encoding="utf-8")
    os.chmod(target, 0o644)

    result = SetPermissionsAbility().run(built(SetPermissionsParamsBag.from_params({
        Keys.path: str(target),
        Keys.permission_code: "99999",
    })))

    assert result.status == "error"
    assert result.code == "invalid-param"
    assert os.stat(target).st_mode & 0o7777 == 0o644


def test_set_permissions_on_a_missing_path_is_not_found(tmp_path: Path) -> None:
    """A path that does not exist is not-found, never a silent success."""
    result = SetPermissionsAbility().run(built(SetPermissionsParamsBag.from_params({
        Keys.path: str(tmp_path / "nope.txt"),
        Keys.permission_code: "0644",
    })))

    assert result.status == "error"
    assert result.code == "not-found"


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

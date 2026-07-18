# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — code_agent's sandboxed workspace tools.

These run the REAL inner-tool `Ability.run()` implementations
(`abilities/read_file.py`, `replace_one.py`, `update_file.py`, `run_script.py`)
against a real (temporary) directory and, for run_script, a REAL Deno
subprocess — no mocks of any in-process behavior anywhere in this file.

The only thing redirected is WHERE the workspace lives on disk: the
`FileMapperService._CODE_AGENT_WORKSPACE_DIR` class-level path constant is
substituted for an isolated `tmp_path` for the duration of each test, via the
same `patch.object(FileMapperService, "<_CLASS_LEVEL_PATH_CONSTANT>", ...)`
pattern already established for `_DOCUMENTS_DIR` in test_document_api.py and
for `get_db_path` in conftest.py. This is a path-constant substitution, not a
mock of production logic — every ability under test still runs its real
containment check (`abilities/_workspace.py::resolve_in_root`), its real
uniqueness/destructive-overwrite gates, and (for run_script) a real
`subprocess.run(["deno", ...])` call. Without this redirection every test
would read and write the real, shared `data/code_agent_workspace/` directory.

The outer `code_agent` delegate ability itself (`abilities/code_agent.py`) is
NOT covered here: `run()` requires a bound `MessageProcessor` and drives a
real `MessageProcessor.process()` sub-loop against a live LLM provider. That
is not something a deterministic unit-tier test can exercise without faking
the model, which these house rules forbid — it belongs in live end-to-end
verification instead.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from abilities.read_file import ReadFileAbility
from abilities.replace_one import ReplaceOneAbility
from abilities.run_script import RunScriptAbility
from abilities.update_file import UpdateFileAbility
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_HAS_DENO = shutil.which("deno") is not None


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path) -> Iterator[Path]:
    """Point the code_agent workspace root at an isolated tmp dir for every
    test in this file.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    with patch.object(FileMapperService, "_CODE_AGENT_WORKSPACE_DIR", workspace):
        yield workspace


def test_workspace_containment_rejects_absolute_and_dotdot_paths(_isolated_workspace: Path) -> None:
    """An absolute path and a `..`-escaping relative path are both refused
    before any file access is attempted — the workspace sandbox is not
    optional, regardless of whether the target actually exists."""
    absolute = ReadFileAbility().run({"path": "/etc/passwd"})
    assert absolute.status == "error"
    assert absolute.code == "path-escapes-root"

    dotdot = ReadFileAbility().run({"path": "../escape.txt"})
    assert dotdot.status == "error"
    assert dotdot.code == "path-escapes-root"


def test_workspace_containment_rejects_symlink_escape(_isolated_workspace: Path) -> None:
    """A symlink planted inside the workspace that resolves to a target
    OUTSIDE it must be refused too — resolve_in_root follows symlinks via
    Path.resolve() before checking containment, so the escape cannot hide
    behind an on-disk indirection."""
    workspace = _isolated_workspace
    outside_secret = workspace.parent / "outside_secret.txt"
    outside_secret.write_text("do not leak this", encoding="utf-8")
    (workspace / "escape_link").symlink_to(outside_secret)

    result = ReadFileAbility().run({"path": "escape_link"})

    assert result.status == "error"
    assert result.code == "path-escapes-root"


def test_replace_one_gates_on_uniqueness_before_writing(_isolated_workspace: Path) -> None:
    """replace_one refuses both a missing match and an ambiguous
    (multi-occurrence) match, leaving the file byte-for-byte untouched in
    both cases — the write only ever happens once uniqueness is proven."""
    workspace = _isolated_workspace
    target = workspace / "main.ts"

    no_match_content = "const greeting = 'hello';\n"
    target.write_text(no_match_content, encoding="utf-8")
    no_match = ReplaceOneAbility().run(
        {"path": "main.ts", "search": "nonexistent_symbol", "replace": "x"}
    )
    assert no_match.status == "error"
    assert no_match.code == "no-match"
    assert target.read_text(encoding="utf-8") == no_match_content

    ambiguous_content = "let x = 1;\nlet x = 1;\n"
    target.write_text(ambiguous_content, encoding="utf-8")
    ambiguous = ReplaceOneAbility().run(
        {"path": "main.ts", "search": "let x = 1;", "replace": "let x = 2;"}
    )
    assert ambiguous.status == "error"
    assert ambiguous.code == "ambiguous-match"
    assert target.read_text(encoding="utf-8") == ambiguous_content


def test_replace_one_succeeds_on_unique_match_and_persists_to_disk(_isolated_workspace: Path) -> None:
    """A unique match is replaced exactly once, and the new content is
    actually persisted to the real file on disk — not just reported in the
    tool's response."""
    workspace = _isolated_workspace
    target = workspace / "main.ts"
    target.write_text("const port = 3000;\nconsole.log(port);\n", encoding="utf-8")

    result = ReplaceOneAbility().run(
        {"path": "main.ts", "search": "const port = 3000;", "replace": "const port = 8080;"}
    )

    assert result.status == "success"
    assert target.read_text(encoding="utf-8") == "const port = 8080;\nconsole.log(port);\n"


def test_update_file_destructive_guard_blocks_snippet_but_allows_real_rewrite(_isolated_workspace: Path) -> None:
    """Overwriting a large existing file with a tiny snippet is refused as a
    likely-accidental destructive edit (file untouched); overwriting it with
    genuinely comparable new content is allowed and actually persists."""
    workspace = _isolated_workspace
    target = workspace / "large.ts"
    original = "x" * 4200
    target.write_text(original, encoding="utf-8")

    blocked = UpdateFileAbility().run({"path": "large.ts", "content": "tiny"})
    assert blocked.status == "error"
    assert blocked.code == "destructive-partial-overwrite"
    assert target.read_text(encoding="utf-8") == original

    rewrite = "y" * 4300
    allowed = UpdateFileAbility().run({"path": "large.ts", "content": rewrite})
    assert allowed.status == "success"
    assert target.read_text(encoding="utf-8") == rewrite


@pytest.mark.skipif(not _HAS_DENO, reason="deno is not installed on this machine")
def test_run_script_executes_real_ts_and_captures_stdout(_isolated_workspace: Path) -> None:
    """run_script actually spawns the real Deno runtime against a real .ts
    file on disk and captures its real stdout — not a canned response."""
    workspace = _isolated_workspace
    (workspace / "hello.ts").write_text('console.log("chalie-test-marker-42");\n', encoding="utf-8")

    result = RunScriptAbility().run({"path": "hello.ts"})

    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    assert body["exit_code"] == 0
    stdout = body["stdout"]
    assert isinstance(stdout, str)
    assert "chalie-test-marker-42" in stdout


def test_run_script_refuses_non_ts_file(_isolated_workspace: Path) -> None:
    """A non-.ts file is refused before any subprocess is ever spawned — this
    must fail the same way even on a machine with no Deno installed, so it
    carries no skipif."""
    workspace = _isolated_workspace
    (workspace / "notes.txt").write_text("not a script", encoding="utf-8")

    result = RunScriptAbility().run({"path": "notes.txt"})

    assert result.status == "error"
    assert result.code == "unsupported-file-type"


@pytest.mark.skipif(not _HAS_DENO, reason="deno is not installed on this machine")
def test_run_script_cannot_read_file_outside_workspace_sandbox(_isolated_workspace: Path) -> None:
    """The real Deno sandbox denies filesystem access outside the
    --allow-read=<workspace> scope: a script cannot read a secret file living
    just outside the workspace root, even though nothing in Python enforces
    that — Deno's own permission model does. Run-faithful contract: the tool
    call itself succeeded (the process ran to completion), but the script's
    own exit code reports its failure and the secret content never reaches
    stdout."""
    workspace = _isolated_workspace
    outside_secret = workspace.parent / "outside_secret.txt"
    secret_value = "top-secret-value-9f3a"
    outside_secret.write_text(secret_value, encoding="utf-8")

    script = (
        "try {\n"
        f"  const text = await Deno.readTextFile({json.dumps(str(outside_secret))});\n"
        "  console.log(text);\n"
        "} catch (e) {\n"
        "  console.error(String(e));\n"
        "  Deno.exit(1);\n"
        "}\n"
    )
    (workspace / "peek.ts").write_text(script, encoding="utf-8")

    result = RunScriptAbility().run({"path": "peek.ts"})

    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    assert body["exit_code"] != 0
    stderr = body["stderr"]
    assert isinstance(stderr, str)
    assert stderr.strip() != ""
    stdout = body["stdout"]
    assert isinstance(stdout, str)
    assert secret_value not in stdout


@pytest.mark.skipif(not _HAS_DENO, reason="deno is not installed on this machine")
def test_run_script_rejects_pre_existing_symlink_escape(_isolated_workspace: Path) -> None:
    """A symlink planted inside the workspace before run_script is ever
    invoked, whose target resolves OUTSIDE the workspace, must be refused
    before Deno is ever spawned: Deno's scoped --allow-read/--allow-write
    flags follow an on-disk symlink without re-validating where it resolves,
    so run_script's own pre-flight walk of the workspace is the only thing
    standing between a planted symlink and real filesystem access outside
    the sandbox. The offending symlink's workspace-relative path must be
    named in the error so the model can find and remove it."""
    workspace = _isolated_workspace
    (workspace / "main.ts").write_text('console.log("should never run");\n', encoding="utf-8")

    outside_dir = workspace.parent / "outside"
    outside_dir.mkdir()
    outside_target = outside_dir / "secret.txt"
    outside_target.write_text("do not leak this", encoding="utf-8")
    os.symlink(outside_target, workspace / "planted_link")

    result = RunScriptAbility().run({"path": "main.ts"})

    assert result.status == "error"
    assert result.code == "symlink-escape"
    assert isinstance(result.body, str)
    assert "planted_link" in result.body

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RunScriptAbility — execute a TypeScript file from the code_agent workspace in a Deno sandbox.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Unlike the deleted code_eval (a stdin-piped snippet with zero permissions),
this runs an on-disk ``.ts`` file the model has already written into the
workspace via create_file/update_file, with permissions scoped to that same
workspace: ``--allow-read``/``--allow-write`` cover only the sandbox root, and
``--allow-net`` lets a script fetch data. ``--no-prompt`` turns any permission
it does not hold into a hard error instead of blocking on an unanswerable
interactive prompt.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult, truncate
from abilities._workspace import get_workspace_root, resolve_existing_file
from configs.enums.param_key import Keys

if TYPE_CHECKING:
    from typing import TypedDict

    class _TruncMeta(TypedDict, total=False):
        truncated: bool


logger = logging.getLogger(__name__)

# Clip each stream so a single run's output stays cheap to feed back into
# model context, reporting clipping uniformly via `meta truncated=true`
# instead of silently dropping output.
_MAX_OUTPUT_CHARS = 100 * 1024

_DENO_BIN = "deno"

# Hard wall-clock cap on a single execution, so a runaway script can never
# pin the delegate loop indefinitely.
_EXEC_TIMEOUT_S = 600


class RunScriptAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "run_script"

    def get_summary(self) -> str:
        return (
            "Run an existing TypeScript (.ts) file from the code_agent workspace "
            "in a sandboxed Deno runtime, with file access and network access "
            "scoped to the workspace. Write the script with create_file or "
            "update_file first, then run it here."
        )

    def get_examples(self) -> list[str]:
        return [
            "run the script.ts file I just wrote",
            "execute main.ts and show me the output",
            "run the data processing script",
            "test the function by running test.ts",
            "run the file that fetches and parses the API response",
            "execute the script that writes the report to disk",
            "run this TypeScript file to verify it works",
            "run the script and tell me if it errors",
        ]

    def get_search_tooltip(self) -> str:
        return "run a TypeScript file in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": (
                    "Path relative to the workspace root of the .ts file to run. "
                    "The file must already exist — create it first with "
                    "create_file or update_file."
                ),
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    # Result-contract messages — mirrors code_eval's actionable-error contract
    # so a run that cannot proceed always fails loudly instead of returning a
    # silent empty success the model would misread as 'done'.
    _ERR_NOT_TS = "Only .ts files can be run."
    _ERR_TIMEOUT = (
        "Your script was stopped because it ran longer than 10 minutes. "
        "This usually means an infinite loop or an operation that is too "
        "large — check your loop conditions or reduce the amount of work."
    )
    _ERR_NO_DENO = "The TypeScript sandbox runtime (Deno) is not installed, so the script could not be run."
    _ERR_CRASHED = "The script could not be run: the sandbox process exited unexpectedly without returning a result."
    _ERR_SYMLINK_ESCAPE = "The workspace contains a symlink that points outside it: {path}."
    _ERR_SYMLINK_UNRESOLVABLE = "The workspace contains a symlink that cannot be resolved: {path}."
    _ERR_WORKSPACE_UNREADABLE = "Containment could not be verified because {path} is unreadable, so the script was not run."

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))

        resolved, error = resolve_existing_file(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        if resolved.suffix != ".ts":
            return ToolResult.err(
                self._ERR_NOT_TS,
                code="unsupported-file-type",
                hint="only TypeScript (.ts) files can be run.",
            )

        return self._execute_with_cap(resolved)

    def _execute_with_cap(self, script_path: Path) -> ToolResult:
        """Run *script_path* in a fresh, workspace-scoped Deno process with a
        hard wall-clock cap. ``subprocess.run`` blocks until the process exits,
        so the sandbox is disposed the moment the script finishes, whether it
        succeeded or not. On timeout the process is killed and an actionable
        error is returned so the ACT loop never hangs forever.
        """
        if shutil.which(_DENO_BIN) is None:
            return ToolResult.err(
                self._ERR_NO_DENO,
                code="no-runtime",
                hint="install Deno so the TypeScript sandbox can execute scripts.",
            )

        workspace = get_workspace_root()

        symlink_error = _find_symlink_escape(workspace)
        if symlink_error is not None:
            return symlink_error

        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                [
                    _DENO_BIN,
                    "run",
                    "--no-config",
                    "--no-prompt",
                    f"--allow-read={workspace}",
                    f"--allow-write={workspace}",
                    "--allow-net",
                    str(script_path),
                ],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=_EXEC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.err(
                self._ERR_TIMEOUT,
                code="timeout",
                hint="reduce the amount of work or run on a smaller input.",
            )
        except OSError:
            return ToolResult.err(
                self._ERR_CRASHED,
                code="sandbox-crashed",
                hint="retry, or simplify the script so it runs within the sandbox.",
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        return self._assemble(completed.stdout, completed.stderr, completed.returncode, duration_ms)

    def _assemble(self, stdout: str, stderr: str, exit_code: int, duration_ms: int) -> ToolResult:
        """Clip each stream via the shared truncate primitive (``meta
        truncated=true`` when either was clipped) and return a branchable run
        result. Unlike code_eval, a clean exit with empty stdout is a valid
        outcome here — a script can legitimately succeed purely through
        file-system side effects (create_file/update_file calls it made)
        without printing anything, so no no-output guard is applied.
        """
        stdout, clipped_out = truncate(stdout, _MAX_OUTPUT_CHARS)
        stderr, clipped_err = truncate(stderr, _MAX_OUTPUT_CHARS)

        meta: "_TruncMeta" = {"truncated": True} if (clipped_out or clipped_err) else {}
        return ToolResult.ok(
            {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
            },
            **meta,
        )


def _find_symlink_escape(root: Path) -> ToolResult | None:
    """Return the refusal to run the script with, or ``None`` when *root*'s
    tree is clean.

    Deno's scoped ``--allow-read``/``--allow-write`` flags follow a symlink
    that already exists on disk without re-validating where it resolves, so a
    symlink planted inside the workspace before this call could hand the
    sandboxed script read or write access anywhere on the filesystem. Every
    other tool in the toolkit is already safe because ``resolve_in_root``
    resolves symlinks before its containment check; this walk gives the one
    tool that hands the workspace to an external process the same
    containment guarantee immediately before it does so.

    Two things can stop the walk from reaching a verdict, and each is a
    refusal in its own right rather than a silent pass:

    - A directory the walk cannot read at all. ``os.walk``'s default
      ``onerror`` swallows the ``OSError`` and just skips that subtree, which
      would let a symlink planted behind a locked-down directory escape
      unseen. The ``onerror`` callback below records the failure instead, so
      any unscannable directory fails the whole check closed.
    - A symlink whose target cannot be resolved at all — typically a
      self- or mutually-referential loop, which makes ``Path.resolve()``
      raise instead of returning a path. With nothing to test for
      containment, an unresolvable link is refused the same as an escaping
      one. The refusal states only that the link could not be resolved and
      never asserts a specific cause, because the ``except`` clause catches
      any resolution failure (an ``OSError`` as well as a loop's
      ``RuntimeError``), not loops alone — the likely fixes stay in the hint.
    """
    root_resolved = root.resolve()
    unreadable: list[str] = []

    def _record_unreadable(exc: OSError) -> None:
        unreadable.append(exc.filename or str(root))

    for dirpath, dirnames, filenames in os.walk(root, onerror=_record_unreadable, followlinks=False):
        for name in (*dirnames, *filenames):
            candidate = Path(dirpath) / name
            if not candidate.is_symlink():
                continue
            try:
                target = candidate.resolve()
            except (OSError, RuntimeError):
                return ToolResult.err(
                    RunScriptAbility._ERR_SYMLINK_UNRESOLVABLE.format(path=candidate.relative_to(root)),
                    code="symlink-escape",
                    hint="remove the broken or looping symlink so the workspace can be scanned safely.",
                )
            if target != root_resolved and not target.is_relative_to(root_resolved):
                return ToolResult.err(
                    RunScriptAbility._ERR_SYMLINK_ESCAPE.format(path=candidate.relative_to(root)),
                    code="symlink-escape",
                    hint="remove the symlink or repoint it so it resolves inside the workspace.",
                )

    if unreadable:
        try:
            unreadable_path = Path(unreadable[0]).relative_to(root)
        except ValueError:
            unreadable_path = Path(unreadable[0])
        return ToolResult.err(
            RunScriptAbility._ERR_WORKSPACE_UNREADABLE.format(path=unreadable_path),
            code="workspace-unreadable",
            hint="fix the directory's permissions so the workspace can be scanned before the script runs.",
        )

    return None

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

# Clip each stream to the same budget code_eval used, reporting clipping
# uniformly via `meta truncated=true` instead of silently dropping output.
_MAX_OUTPUT_CHARS = 100 * 1024

_DENO_BIN = "deno"

# Hard wall-clock cap on a single execution — same budget code_eval used.
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

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RunScriptAbility — execute a TypeScript file with Deno.

One of the inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Unlike the deleted code_eval (a stdin-piped snippet with zero permissions),
this runs an on-disk ``.ts`` file the model has already written into the
code_agent workspace (the default place the coding agent writes scripts),
with full permissions (``-A``): no read, write, or network restriction.
``--no-config`` only stops Deno from picking up an ambient
``deno.json``/``deno.jsonc`` so a run behaves the same regardless of what
else is on disk — it is not a permission boundary.

The script path is taken as an absolute path, so any ``.ts`` file on the
system can be executed — the code_agent workspace is just the conventional
home for scripts the coding agent produces. Optional command-line arguments
are appended to the invocation and reach the script as ``Deno.args``, so one
script can serve many inputs without being rewritten per run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult, truncate
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.run_script_params_bag import RunScriptParamsBag

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

_DENO_RUN_FLAGS: tuple[str, ...] = ("run", "--no-config", "-A")


class RunScriptAbility(Ability[RunScriptParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    # The typed input contract: the dispatch seam builds the bag via
    # RunScriptParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = RunScriptParamsBag

    def get_name(self) -> str:
        return "run_script"

    def get_summary(self) -> str:
        return (
            "Run an existing TypeScript (.ts) file with Deno, with full "
            "permissions (file, network, env, and every other capability). "
            "Scripts conventionally live in the code_agent workspace (the "
            "default place the coding agent writes them), but any absolute "
            ".ts path can be executed. Write the script with file_write "
            "first, then run it here. Optional args are passed to the "
            "script and readable inside it as Deno.args."
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
        return "run a TypeScript file with Deno"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": (
                    "Absolute path to the .ts file to run. The file must "
                    "already exist — create it first with file_write."
                ),
            },
            Keys.args: {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional command-line arguments passed to the script, "
                    "readable inside it as Deno.args."
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
    _ERR_NO_DENO = "The TypeScript runtime (Deno) is not installed, so the script could not be run."
    _ERR_CRASHED = "The script could not be run: the Deno process exited unexpectedly without returning a result."

    def run(self, params: RunScriptParamsBag) -> ToolResult:
        script_path = Path(params.path)

        if not script_path.is_absolute():
            return ToolResult.err(
                "The script path must be an absolute path.",
                code="invalid-path",
                hint="Pass an absolute path like /path/to/script.ts.",
            )

        if not script_path.exists():
            return ToolResult.err(
                f"{params.path} does not exist.",
                code="not-found",
                hint="use file_write to create the script first.",
            )

        if script_path.suffix != ".ts":
            return ToolResult.err(
                self._ERR_NOT_TS,
                code="unsupported-file-type",
                hint="only TypeScript (.ts) files can be run.",
            )

        return self._execute_with_cap(script_path, params.args)

    def _execute_with_cap(self, script_path: Path, script_args: list[str]) -> ToolResult:
        """Run *script_path* in a fresh Deno process with full permissions and
        a hard wall-clock cap. ``subprocess.run`` blocks until the process
        exits, so the process is gone the moment the script finishes, whether
        it succeeded or not. On timeout the process is killed and an
        actionable error is returned so the ACT loop never hangs forever.
        """
        if shutil.which(_DENO_BIN) is None:
            return ToolResult.err(
                self._ERR_NO_DENO,
                code="no-runtime",
                hint="install Deno so scripts can be run.",
            )

        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                [_DENO_BIN, *_DENO_RUN_FLAGS, str(script_path), *script_args],
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
                code="deno-crashed",
                hint="retry, or simplify the script so it runs cleanly.",
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        return self._assemble(completed.stdout, completed.stderr, completed.returncode, duration_ms)

    def _assemble(self, stdout: str, stderr: str, exit_code: int, duration_ms: int) -> ToolResult:
        """Clip each stream via the shared truncate primitive (``meta
        truncated=true`` when either was clipped) and return a branchable run
        result. Unlike code_eval, a clean exit with empty stdout is a valid
        outcome here — a script can legitimately succeed purely through
        file-system side effects without printing anything, so no no-output
        guard is applied.
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

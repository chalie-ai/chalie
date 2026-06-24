"""
CodeEvalAbility — TypeScript / JavaScript one-shot sandbox.

Runs model-supplied TypeScript (or JavaScript) in a Deno sandbox with zero
permissions: no network, no file system, no env, no subprocess — only pure
computation and ``console.log``. The sandbox is a fresh ``deno run`` process
fed the code on stdin; it exits (and is disposed) the moment the script
finishes, whether the code was correct or not.

Deno is the single-binary sandbox: built-in TypeScript compiler, no
package.json, no node_modules, permission-gated by default. ``--no-prompt``
turns a missing permission into a hard ``NotCapable`` error instead of
blocking on a prompt nobody can answer, so a script that reaches for the
network or disk fails loudly with the reason on stderr rather than hanging.

Execution uses ``subprocess.run`` with a wall-clock ``timeout``: on timeout
the process is killed and an actionable error is returned so the ACT loop
never hangs. stdout and stderr are captured separately so the model branches
on ``exit_code`` exactly as it would on ``node script.js``.
"""

import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult, truncate

if TYPE_CHECKING:
    from typing import TypedDict

    class _TruncMeta(TypedDict, total=False):
        truncated: bool

logger = logging.getLogger(__name__)

# Clip each stream to the same budget bash uses, reporting clipping uniformly via
# ``meta truncated=true`` instead of silently dropping output.
_MAX_OUTPUT_CHARS = 100 * 1024

# The Deno binary name resolved once at import; resolved lazily on first run so
# a missing binary surfaces as an actionable runtime error, not an import crash.
_DENO_BIN = "deno"


class CodeEvalAbility(Ability):
    # Action-less tool: the dispatcher's ACTION_REQUIRED pre-gate rejects a
    # missing/empty ``code`` as ``code=missing-params`` BEFORE the policy gate or
    # run(). The ``""`` key covers action-less tools (precedent: vision).
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.code,)}

    def get_name(self) -> str:
        return "code_eval"

    def get_summary(self) -> str:
        return (
            "Run TypeScript or JavaScript in a sandboxed Deno runtime to compute "
            "formulas, verify logic, and perform precise calculations. Use "
            "console.log to emit results."
        )

    def get_examples(self) -> list[str]:
        return [
            "calculate the exact monthly payment on a mortgage at 6.5% over 30 years",
            "verify this mathematical formula for me",
            "what is the compound interest on $10,000 at 4% for 15 years",
            "run this JavaScript snippet and tell me the output",
            "calculate the standard deviation of these numbers",
            "compute the fibonacci sequence up to the 20th term",
            "convert 37.5°C to Fahrenheit precisely",
            "check if this sorting algorithm is correct",
        ]

    def get_search_tooltip(self) -> str:
        return "TypeScript / JavaScript code execution"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.code: {
                "type": "string",
                "description": (
                    "TypeScript or JavaScript to execute. Use console.log to emit "
                    "results. The sandbox has zero permissions: no network, no "
                    "file system, no subprocess. Math, JSON, Array, and the rest "
                    "of the standard JS/TS built-in globals are available; "
                    "imports are blocked."
                ),
            },
        },
        "required": [Keys.code],
    }

    # Result-contract messages. The error strings are the actionable signals the
    # LLM receives in place of a silent empty success, which previously caused it
    # to retry the same call until the iteration wall.
    _ERR_NO_CODE = (
        "You need to provide code to be executed. "
        "Pass the TypeScript or JavaScript you want to run in the `code` parameter."
    )
    _ERR_NO_OUTPUT = (
        "Your code did not produce any output. "
        "Ensure you use `console.log` on whatever you want outputted / returned."
    )
    _ERR_TIMEOUT = (
        "Your code was stopped because it ran longer than 10 minutes. "
        "This usually means an infinite loop or an operation that is too "
        "large — check your loop conditions or reduce the amount of work."
    )
    _ERR_NO_DENO = (
        "The JavaScript sandbox runtime (Deno) is not installed, so the code "
        "could not be run."
    )
    _ERR_CRASHED = (
        "The code could not be run: the sandbox process exited unexpectedly "
        "without returning a result."
    )

    # Hard wall-clock cap on a single execution.
    _EXEC_TIMEOUT_S = 600

    def run(self, params: dict[str, object]) -> ToolResult:
        """Run the supplied TypeScript/JavaScript under a hard 10-minute cap and
        return a run result: anything that RAN comes back as ``ok`` with a
        branchable ``exit_code`` (exactly how ``deno run script.ts`` behaves), so
        the model can branch on it rather than parse prose. Errors are reserved
        for harness failures (missing code, no output, timeout, no runtime,
        sandbox crash) — never a silent empty success the LLM would misread as
        'done' and retry.

        Dispatched calls never reach the residue guard below: argument
        sanitisation strips a whitespace-only ``code`` to ``""`` before the
        ACTION_REQUIRED pre-gate, which rejects it. The guard covers direct
        callers the same way the pre-gate would have.
        """
        code = (cast(str, params.get(Keys.code)) or "").strip()
        if not code:
            return ToolResult.err(
                self._ERR_NO_CODE,
                code="missing-params",
                hint="pass the TypeScript or JavaScript you want to run in the `code` parameter.",
            )
        return self._execute_with_cap(code)

    def _execute_with_cap(self, code: str) -> ToolResult:
        """Run the sandboxed code in a fresh Deno process with a hard wall-clock
        cap, then assemble the ToolResult. ``subprocess.run`` blocks until the
        process exits — so the sandbox is disposed automatically the moment the
        script finishes, whether the code was correct or not. On timeout the
        process is killed and an actionable error is returned so the ACT loop
        never hangs forever.

        ``deno run --no-config --no-prompt -`` reads the code from stdin and runs
        it as TypeScript. ``--no-config`` ignores any local ``deno.json``;
        ``--no-prompt`` turns a missing permission into a hard ``NotCapable``
        error rather than blocking on an interactive prompt nobody can answer."""
        if shutil.which(_DENO_BIN) is None:
            return ToolResult.err(
                self._ERR_NO_DENO,
                code="no-runtime",
                hint="install Deno so the JavaScript sandbox can execute code.",
            )

        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
                [_DENO_BIN, "run", "--no-config", "--no-prompt", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=self._EXEC_TIMEOUT_S,
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
                hint="retry, or simplify the code so it runs within the sandbox.",
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        return self._assemble(
            completed.stdout, completed.stderr, completed.returncode, duration_ms
        )

    def _assemble(
        self, stdout: str, stderr: str, exit_code: int, duration_ms: int
    ) -> "ToolResult":
        """Clip each stream via the shared truncate primitive (``meta
        truncated=true`` when either was clipped), enforce the no-output
        guardrail (ran cleanly but printed nothing -> an actionable error, not a
        silent empty success), and otherwise return a branchable run result."""
        stdout, clipped_out = truncate(stdout, _MAX_OUTPUT_CHARS)
        stderr, clipped_err = truncate(stderr, _MAX_OUTPUT_CHARS)

        # Ran to completion with a clean exit but emitted nothing: the model would
        # read empty success as 'done' and retry the identical call until the
        # iteration wall. Surface it as an explicit, actionable error instead.
        if exit_code == 0 and not stdout:
            return ToolResult.err(
                self._ERR_NO_OUTPUT,
                code="no-output",
                hint="use console.log() on whatever you want returned.",
            )

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

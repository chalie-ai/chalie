"""
CodeEvalAbility — Restricted Python scratchpad.

Runs user-supplied Python in a RestrictedPython sandbox. No file I/O,
no subprocess, no imports. Pre-loaded safe modules: math, statistics,
json, decimal, fractions, itertools, functools, collections.

Execution happens in a separate (``spawn``) process with a hard 10-minute
wall-clock cap. The subprocess is the only way to force-kill arbitrary
CPU-bound code (e.g. ``while True``) — a thread cannot be interrupted, and
SIGALRM only fires on the main thread. ``spawn`` (not ``fork``) avoids
inheriting locks from the multithreaded server.
"""

import collections
import decimal
import fractions
import functools
import itertools
import json
import logging
import math
import multiprocessing
import statistics
import time
import traceback
from queue import Empty
from typing import ClassVar

from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.PrintCollector import PrintCollector

from abilities._ability import Ability
from abilities._result import ToolResult, truncate

logger = logging.getLogger(__name__)

# Clip each stream to the same budget bash uses, reporting clipping uniformly via
# ``meta truncated=true`` instead of silently dropping output.
_MAX_OUTPUT_CHARS = 100 * 1024

# Sandbox identity. Module-level so the picklable worker (``spawn``) and its
# helpers reference them without carrying a class instance across the boundary.
_FILENAME = "<scratchpad>"
_PRINT_VAR = "_print"


def _guarded_getattr(obj, name):
    """Block access to private/dunder attributes inside the sandbox."""
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' is not permitted in the sandbox")
    return getattr(obj, name)


class CodeEvalAbility(Ability):
    # Action-less tool: the dispatcher's ACTION_REQUIRED pre-gate rejects a
    # missing/empty ``code`` as ``code=missing-params`` BEFORE the policy gate or
    # run(). The ``""`` key covers action-less tools (precedent: vision).
    ACTION_REQUIRED: ClassVar[dict] = {"": ("code",)}

    def get_name(self) -> str:
        return "code_eval"

    def get_summary(self) -> str:
        return "Run Python code in a restricted sandbox to compute formulas, verify logic, and perform precise calculations."

    def get_examples(self) -> list[str]:
        return [
            "calculate the exact monthly payment on a mortgage at 6.5% over 30 years",
            "verify this mathematical formula for me",
            "what is the compound interest on $10,000 at 4% for 15 years",
            "run this Python snippet and tell me the output",
            "calculate the standard deviation of these numbers",
            "compute the fibonacci sequence up to the 20th term",
            "convert 37.5°C to Fahrenheit precisely",
            "check if this sorting algorithm is correct",
        ]

    def get_search_tooltip(self) -> str:
        return "Python code execution"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to emit results.",
            },
        },
        "required": ["code"],
    }

    # Result-contract messages. The error strings are the actionable signals the
    # LLM receives in place of a silent empty success, which previously caused it
    # to retry the same call until the iteration wall.
    _ERR_NO_CODE = (
        "You need to provide python code to be executed. "
        "Pass the Python you want to run in the `code` parameter."
    )
    _ERR_NO_OUTPUT = (
        "Your code did not produce any output. "
        "Ensure you use `print` on whatever you want outputted / returned"
    )
    _ERR_TIMEOUT = (
        "Your code was stopped because it ran longer than 10 minutes. "
        "This usually means an infinite loop or an operation that is too "
        "large — check your loop conditions or reduce the amount of work."
    )
    _ERR_CRASHED = (
        "The code could not be run: the sandbox process exited unexpectedly "
        "without returning a result."
    )

    # Hard wall-clock cap on a single execution, and how often the parent polls
    # the result queue while waiting (so an early crash is detected promptly
    # instead of waiting out the full cap).
    _EXEC_TIMEOUT_S = 600
    _POLL_INTERVAL_S = 2.0

    # Pre-built restricted globals — assembled once at import time.
    _RESTRICTED_GLOBALS: ClassVar[dict] = {
        **safe_globals,
        "__builtins__": safe_builtins,
        "_print_": PrintCollector,
        "_getattr_": _guarded_getattr,
        "_getiter_": iter,
        "_getitem_": lambda obj, key: obj[key],
        "math": math,
        "statistics": statistics,
        "json": json,
        "decimal": decimal,
        "fractions": fractions,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
    }

    def run(self, params: dict) -> ToolResult:
        """Run the supplied Python under a hard 10-minute cap and return a
        CPython-faithful run result: anything that RAN comes back as ``ok`` with a
        branchable ``exit_code`` (exactly how ``python script.py`` behaves), so the
        model can branch on it rather than parse prose. Errors are reserved for
        harness failures (missing code, no output, timeout, sandbox crash) — never
        a silent empty success the LLM would misread as 'done' and retry.

        Dispatched calls never reach the residue guard below: argument
        sanitisation strips a whitespace-only ``code`` to ``""`` before the
        ACTION_REQUIRED pre-gate, which rejects it. The guard covers direct
        callers the same way the pre-gate would have.
        """
        code = (params.get("code") or "").strip()
        if not code:
            return ToolResult.err(
                self._ERR_NO_CODE,
                code="missing-params",
                hint="pass the Python you want to run in the `code` parameter.",
            )
        return self._execute_with_cap(code)

    def _execute_with_cap(self, code: str) -> ToolResult:
        """Run the sandboxed code in a separate process with a hard wall-clock cap,
        then assemble the ToolResult in THIS (parent) process. A subprocess is the
        only way to force-kill arbitrary CPU-bound code (e.g. ``while True``) — a
        thread cannot be interrupted. On timeout the process is terminated and an
        actionable error is returned so the LLM can correct course instead of the
        ACT loop hanging forever. The worker only computes the raw streams; the
        parent owns ``duration_ms``, truncation, and the no-output guardrail."""
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_sandbox_worker, args=(code, result_queue), daemon=True,
        )
        started = time.monotonic()
        proc.start()

        deadline = started + self._EXEC_TIMEOUT_S
        result = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                # Read BEFORE join so a large (untruncated) result never
                # deadlocks the queue feeder thread against proc.join().
                result = result_queue.get(timeout=min(self._POLL_INTERVAL_S, remaining))
                break
            except Empty:
                if not proc.is_alive():
                    break  # finished or crashed without producing a result

        if result is not None:
            proc.join()
            duration_ms = int((time.monotonic() - started) * 1000)
            return self._assemble(result, duration_ms)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return ToolResult.err(
                self._ERR_TIMEOUT,
                code="timeout",
                hint="reduce the amount of work or run on a smaller input.",
            )
        proc.join()
        return ToolResult.err(
            self._ERR_CRASHED,
            code="sandbox-crashed",
            hint="retry, or simplify the code so it runs within the sandbox.",
        )

    def _assemble(self, raw: dict, duration_ms: int) -> ToolResult:
        """Turn the worker's raw ``{stdout, stderr, exit_code}`` dict into the
        sealed ToolResult: clip each stream via the shared truncate primitive
        (``meta truncated=true`` when either was clipped), enforce the no-output
        guardrail (ran cleanly but printed nothing → an actionable error, not a
        silent empty success), and otherwise return a branchable run result."""
        stdout, clipped_out = truncate(raw["stdout"], _MAX_OUTPUT_CHARS)
        stderr, clipped_err = truncate(raw["stderr"], _MAX_OUTPUT_CHARS)
        exit_code = raw["exit_code"]

        # Ran to completion with a clean exit but emitted nothing: the model would
        # read empty success as 'done' and retry the identical call until the
        # iteration wall. Surface it as an explicit, actionable error instead.
        if exit_code == 0 and not stdout:
            return ToolResult.err(
                self._ERR_NO_OUTPUT,
                code="no-output",
                hint="use print() on whatever you want returned.",
            )

        meta = {"truncated": True} if (clipped_out or clipped_err) else {}
        return ToolResult.ok(
            {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
            },
            **meta,
        )


def _compile_and_run(code: str) -> dict:
    """Compile and execute *code* in fresh restricted globals, returning a raw
    ``{stdout, stderr, exit_code}`` dict (no ToolResult — the parent assembles
    that). A syntax error or a runtime exception is a CPython-faithful
    ``exit_code`` 1 with the full traceback on ``stderr`` and any partial prints
    preserved on ``stdout``; a clean run is ``exit_code`` 0."""
    try:
        byte_code = compile_restricted(code, filename=_FILENAME, mode="exec")
    except SyntaxError:
        return {"stdout": "", "stderr": traceback.format_exc(), "exit_code": 1}

    # Fresh globals copy per call so state never leaks between executions.
    exec_globals = dict(CodeEvalAbility._RESTRICTED_GLOBALS)
    exec_locals: dict = {}

    try:
        exec(byte_code, exec_globals, exec_locals)
    except Exception:
        # Sandbox boundary: surface the full trace AND any partial print output
        # (stdout) separately so the model branches on exit_code rather than
        # parsing a single blob. The dispatcher logs the outcome; nothing swallowed.
        return {
            "stdout": _captured(exec_locals),
            "stderr": traceback.format_exc(),
            "exit_code": 1,
        }

    return {"stdout": _captured(exec_locals), "stderr": "", "exit_code": 0}


def _captured(exec_locals: dict) -> str:
    """Return text accumulated by the sandbox PrintCollector, or '' if the code
    never called print()."""
    collector = exec_locals.get(_PRINT_VAR)
    return collector() if collector is not None else ""


def _sandbox_worker(code: str, result_queue) -> None:
    """Entry point for the sandbox subprocess: compile and execute *code*, then
    put the raw ``{stdout, stderr, exit_code}`` dict on *result_queue*. Module-level
    so it is picklable under the ``spawn`` start method, and a plain dict (not a
    ToolResult) so the parent owns duration/truncation/guardrail assembly. Runs in
    its own process so a runaway loop can be force-terminated by the parent — a
    thread cannot be killed."""
    result_queue.put(_compile_and_run(code))

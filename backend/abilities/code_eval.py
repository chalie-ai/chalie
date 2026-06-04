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

logger = logging.getLogger(__name__)


def _guarded_getattr(obj, name):
    """Block access to private/dunder attributes inside the sandbox."""
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' is not permitted in the sandbox")
    return getattr(obj, name)


class CodeEvalAbility(Ability):
    NAME = "code_eval"
    SEARCH_TOOLTIP = "Python code execution"
    SUMMARY = "Run Python code in a restricted sandbox to compute formulas, verify logic, and perform precise calculations."
    EXAMPLES = [
        "calculate the exact monthly payment on a mortgage at 6.5% over 30 years",
        "verify this mathematical formula for me",
        "what is the compound interest on $10,000 at 4% for 15 years",
        "run this Python snippet and tell me the output",
        "calculate the standard deviation of these numbers",
        "compute the fibonacci sequence up to the 20th term",
        "convert 37.5°C to Fahrenheit precisely",
        "check if this sorting algorithm is correct",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to emit results.",
            },
        },
        "required": ["code"],
    }

    # Sandbox identity + result-contract messages. The error strings are the
    # actionable signals the LLM receives in place of a silent empty success,
    # which previously caused it to retry the same call until the iteration wall.
    _FILENAME = "<scratchpad>"
    _PRINT_VAR = "_print"
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

    def run(self, params: dict) -> dict:
        """Run the supplied Python under a hard 10-minute cap, returning its
        printed output or an actionable error — never a silent empty success
        that the LLM would misread as 'done' and retry."""
        code = (params.get("code") or "").strip()
        if not code:
            return self._error(self._ERR_NO_CODE)
        return self._execute_with_cap(code)

    def _execute_with_cap(self, code: str) -> dict:
        """Run the sandboxed code in a separate process with a hard wall-clock
        cap. A subprocess is the only way to force-kill arbitrary CPU-bound code
        (e.g. ``while True``) — a thread cannot be interrupted. On timeout the
        process is terminated and an actionable error is returned so the LLM can
        correct course instead of the ACT loop hanging forever."""
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_sandbox_worker, args=(code, result_queue), daemon=True,
        )
        proc.start()

        deadline = time.monotonic() + self._EXEC_TIMEOUT_S
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
            return result
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return self._error(self._ERR_TIMEOUT)
        proc.join()
        return self._error(self._ERR_CRASHED)

    def _compile_and_run(self, code: str) -> dict:
        """Compile and execute *code* in fresh restricted globals. Runs inside
        the sandbox subprocess; returns printed output, a full stack trace on
        failure, or a no-output error — never a silent empty success."""
        try:
            byte_code = compile_restricted(code, filename=self._FILENAME, mode="exec")
        except SyntaxError:
            return self._error(traceback.format_exc())
        return self._run(byte_code)

    def _run(self, byte_code) -> dict:
        """Execute compiled bytecode in fresh restricted globals, returning its
        printed output, a full stack trace on failure, or a no-output error."""
        # Fresh globals copy per call so state never leaks between executions.
        exec_globals = dict(self._RESTRICTED_GLOBALS)
        exec_locals: dict = {}

        try:
            exec(byte_code, exec_globals, exec_locals)
        except Exception:
            # Sandbox boundary: surface the full trace (and any partial output)
            # to the LLM. The dispatcher logs the error key, so it is not swallowed.
            return self._error(traceback.format_exc(), self._captured(exec_locals))

        captured = self._captured(exec_locals)
        if not captured:
            return self._error(self._ERR_NO_OUTPUT)
        return {"text": captured, "error": ""}

    def _captured(self, exec_locals: dict) -> str:
        """Return text accumulated by the sandbox PrintCollector, or '' if the
        code never called print()."""
        collector = exec_locals.get(self._PRINT_VAR)
        return collector() if collector is not None else ""

    def _error(self, message: str, captured: str = "") -> dict:
        """Build an error result, preserving any partial print output as text so
        the LLM keeps context alongside the failure."""
        return {"text": captured, "error": message}


def _sandbox_worker(code: str, result_queue) -> None:
    """Entry point for the sandbox subprocess: compile and execute *code*, then
    put the result dict on *result_queue*. Module-level so it is picklable under
    the ``spawn`` start method. Runs in its own process so a runaway loop can be
    force-terminated by the parent — a thread cannot be killed."""
    result_queue.put(CodeEvalAbility()._compile_and_run(code))

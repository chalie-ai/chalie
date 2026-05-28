"""
CodeEvalAbility — Restricted Python scratchpad.

Runs user-supplied Python in a RestrictedPython sandbox. No file I/O,
no subprocess, no imports. Pre-loaded safe modules: math, statistics,
json, decimal, fractions, itertools, functools, collections.
"""

import collections
import decimal
import fractions
import functools
import itertools
import json
import logging
import math
import statistics
import traceback
from typing import ClassVar

from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.PrintCollector import PrintCollector

from abilities._base import Ability

logger = logging.getLogger(__name__)


def _guarded_getattr(obj, name):
    """Block access to private/dunder attributes inside the sandbox."""
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' is not permitted in the sandbox")
    return getattr(obj, name)


class CodeEvalAbility(Ability):
    NAME = "code_eval"
    SEARCH_TOOLTIP = "Python code execution"
    POLICY_CATEGORY = "Code"
    POLICY_LABELS = {"": "Run sandboxed code"}
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
    TIMEOUT = 15

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

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        """Compile and run the supplied Python, returning its printed output or
        an actionable error — never a silent empty success that the LLM would
        misread as 'done' and retry."""
        code = (params.get("code") or "").strip()
        if not code:
            return self._error(self._ERR_NO_CODE)

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

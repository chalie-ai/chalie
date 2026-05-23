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
        code = (params.get("code") or "").strip()

        if not code:
            return {"error": "No code provided", "text": ""}

        try:
            byte_code = compile_restricted(code, filename="<scratchpad>", mode="exec")
        except SyntaxError as exc:
            return {"text": "", "error": f"Syntax error: {exc}"}

        # Fresh globals copy per call so state never leaks between executions.
        exec_globals = dict(self._RESTRICTED_GLOBALS)
        exec_locals: dict = {}

        try:
            exec(byte_code, exec_globals, exec_locals)
        except Exception as exc:
            collector = exec_locals.get("_print")
            captured = collector() if collector is not None else ""
            error_msg = f"{type(exc).__name__}: {exc}"
            text = f"{captured}{error_msg}".strip() if captured else error_msg
            return {"text": text, "error": error_msg}

        collector = exec_locals.get("_print")
        captured = collector() if collector is not None else ""
        return {"text": captured, "error": ""}

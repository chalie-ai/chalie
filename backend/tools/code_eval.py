"""
Code Eval Tool — restricted Python scratchpad.

Runs user-supplied Python in a RestrictedPython sandbox.  No file I/O,
no subprocess, no imports.  Pre-loaded safe modules: math, statistics,
json, decimal, fractions, itertools, functools, collections.
"""

import math
import statistics
import json
import decimal
import fractions
import itertools
import functools
import collections
import logging

from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.PrintCollector import PrintCollector

logger = logging.getLogger(__name__)


def _guarded_getattr(obj, name):
    """Block access to private/dunder attributes inside the sandbox."""
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' is not permitted in the sandbox")
    return getattr(obj, name)


# Pre-built restricted globals -- assembled once at import time.
_RESTRICTED_GLOBALS = {
    **safe_globals,
    "__builtins__": safe_builtins,
    # Print capture: compile_restricted rewrites print() -> _print_()
    "_print_": PrintCollector,
    # Attribute guard: block private/dunder names
    "_getattr_": _guarded_getattr,
    # Iteration and item access guards (required by RestrictedPython)
    "_getiter_": iter,
    "_getitem_": lambda obj, key: obj[key],
    # Safe modules -- pre-loaded, no import statement needed
    "math": math,
    "statistics": statistics,
    "json": json,
    "decimal": decimal,
    "fractions": fractions,
    "itertools": itertools,
    "functools": functools,
    "collections": collections,
}


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    code = (params.get("code") or "").strip()

    if not code:
        return {"error": "No code provided", "text": ""}

    # Compile with RestrictedPython -- catches unsafe constructs before execution.
    try:
        byte_code = compile_restricted(code, filename="<scratchpad>", mode="exec")
    except SyntaxError as exc:
        return {"text": "", "error": f"Syntax error: {exc}"}

    # Fresh globals copy per call so state never leaks between executions.
    exec_globals = dict(_RESTRICTED_GLOBALS)

    try:
        exec(byte_code, exec_globals)
    except Exception as exc:
        captured = str(exec_globals.get("_print", PrintCollector()))
        error_msg = f"{type(exc).__name__}: {exc}"
        text = f"{captured}\n{error_msg}".strip() if captured else error_msg
        return {"text": text, "error": error_msg}

    captured = str(exec_globals.get("_print", PrintCollector()))
    return {"text": captured, "error": ""}

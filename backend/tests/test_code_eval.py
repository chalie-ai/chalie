"""Unit tests for CodeEvalAbility — the sandbox result contract.

Every path must return a non-empty, actionable result: printed output on
success, a full stack trace on failure, and explicit errors for no-output and
no-code. A silent empty success previously caused the LLM to retry the same
call until the ACT-loop iteration wall (caught by the end-to-end scenario suite).
"""

import time

import pytest

from abilities.code_eval import CodeEvalAbility

pytestmark = pytest.mark.unit


def _run(code) -> dict:
    """Execute CodeEvalAbility with the given code param and return the dict."""
    params = {} if code is None else {"code": code}
    return CodeEvalAbility().run(params)


# ── 1. printed output is returned ─────────────────────────────────────


def test_printed_output_is_returned():
    result = _run("print(2 + 2)")
    assert result["text"] == "4\n"
    assert result["error"] == ""


def test_multiple_prints_all_captured():
    result = _run("print('a')\nprint('b')")
    assert result["text"] == "a\nb\n"
    assert result["error"] == ""


# ── 2. runtime errors return a full stack trace ───────────────────────


def test_runtime_error_returns_full_traceback():
    result = _run("print(1 / 0)")
    assert result["error"], "runtime error must populate the error key"
    assert "Traceback (most recent call last)" in result["error"]
    assert "ZeroDivisionError" in result["error"]


def test_runtime_error_preserves_partial_output():
    result = _run("print('before')\nraise ValueError('boom')")
    assert "ValueError" in result["error"]
    assert "boom" in result["error"]
    assert result["text"] == "before\n"


def test_syntax_error_returns_traceback():
    result = _run("print(")
    assert result["error"], "syntax error must populate the error key"
    assert "SyntaxError" in result["error"]


# ── 3. successful run with no output is an explicit error ─────────────


def test_no_output_returns_explicit_error():
    result = _run("x = 2 + 2")
    assert result["text"] == ""
    assert result["error"] == CodeEvalAbility._ERR_NO_OUTPUT
    assert "print" in result["error"]


# ── 4. missing code is an explicit error ──────────────────────────────


def test_missing_code_returns_explicit_error():
    for empty in (None, "", "   "):
        result = _run(empty)
        assert result["text"] == ""
        assert result["error"] == CodeEvalAbility._ERR_NO_CODE


# ── sandbox invariants still hold ─────────────────────────────────────


def test_safe_module_still_usable():
    # math is pre-loaded as a global; the sandbox blocks `import`, so it is
    # used directly. The result is emitted via print.
    result = _run("print(math.sqrt(16))")
    assert result["text"].strip() == "4.0"
    assert result["error"] == ""


def test_import_is_blocked_and_reported():
    # The sandbox forbids imports; the attempt must surface as a stack trace,
    # not a silent empty success.
    result = _run("import os\nprint(os.getcwd())")
    assert result["error"], "blocked import must populate the error key"
    assert result["text"] == ""


def test_no_path_returns_empty_string_loop_signal_is_gone():
    """Regression: a bare expression (no print) must NOT yield empty success."""
    result = _run("42")
    assert result["error"] == CodeEvalAbility._ERR_NO_OUTPUT


# ── hard wall-clock cap: runaway code is force-killed ─────────────────


def test_runaway_loop_is_terminated_with_timeout_error(monkeypatch):
    """A non-terminating loop is force-killed at the cap and returns the
    actionable timeout error — the ACT loop never hangs.

    The real subprocess is spawned and terminated; only the cap length is
    shortened (to keep the suite fast) so this exercises the genuine kill path.
    """
    monkeypatch.setattr(CodeEvalAbility, "_EXEC_TIMEOUT_S", 0.5)
    monkeypatch.setattr(CodeEvalAbility, "_POLL_INTERVAL_S", 0.1)

    start = time.monotonic()
    result = _run("while True:\n    pass")
    elapsed = time.monotonic() - start

    assert result["error"] == CodeEvalAbility._ERR_TIMEOUT
    assert result["text"] == ""
    assert elapsed < 30, "the runaway loop must be killed promptly, not run on"


def test_long_output_round_trips_through_subprocess():
    """A large (untruncated) result returns intact across the process boundary —
    proof the read-before-join path does not deadlock or truncate."""
    result = _run("print('x' * 200000)")
    assert result["error"] == ""
    assert result["text"].strip() == "x" * 200000

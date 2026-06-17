"""Unit tests for CodeEvalAbility — the sandbox result contract (TKT-917).

CPython-faithful: anything that RAN returns ``ok`` with a branchable
``exit_code`` (exactly how ``python script.py`` behaves) — clean exit 0 with
``stdout``, or exit 1 with the full traceback on ``stderr`` and any partial
prints preserved on ``stdout``. Errors are reserved for HARNESS failures:
missing code, no output, timeout, sandbox crash — each a stable kebab code, never
the banned ``code="error"``. A silent empty success previously caused the LLM to
retry the same call until the ACT-loop iteration wall (caught by the end-to-end
scenario suite), so the no-output guardrail is preserved as an explicit error.

``run()`` returns a ``ToolResult``: success carries the structured run result in
``body`` (``{stdout, stderr, exit_code, duration_ms}``); harness failures are
``status == "error"`` with a ``code``. These assert that contract directly.
"""

import time

import pytest

from abilities.code_eval import CodeEvalAbility

pytestmark = pytest.mark.unit


def _run(code):
    params = {} if code is None else {"code": code}
    return CodeEvalAbility().run(params)


# ── 1. printed output is a branchable run result ──────────────────────


def test_printed_output_is_returned():
    result = _run("print(2 + 2)")
    assert result.status == "success"
    assert result.body["stdout"] == "4\n"
    assert result.body["stderr"] == ""
    assert result.body["exit_code"] == 0
    assert isinstance(result.body["duration_ms"], int)
    assert result.body["duration_ms"] >= 0


def test_multiple_prints_all_captured():
    result = _run("print('a')\nprint('b')")
    assert result.status == "success"
    assert result.body["stdout"] == "a\nb\n"
    assert result.body["exit_code"] == 0


# ── 2. runtime errors are exit_code 1 with the full traceback on stderr ──


def test_runtime_error_returns_full_traceback():
    result = _run("print(1 / 0)")
    assert result.status == "success", "anything that RAN is a branchable success"
    assert result.body["exit_code"] == 1
    assert "Traceback (most recent call last)" in result.body["stderr"]
    assert "ZeroDivisionError" in result.body["stderr"]


def test_runtime_error_preserves_partial_output():
    result = _run("print('before')\nraise ValueError('boom')")
    assert result.status == "success"
    assert result.body["exit_code"] == 1
    assert "ValueError" in result.body["stderr"]
    assert "boom" in result.body["stderr"]
    # Partial print output is preserved on stdout, NOT folded into the traceback.
    assert result.body["stdout"] == "before\n"


def test_syntax_error_returns_traceback():
    result = _run("print(")
    assert result.status == "success", "syntax error is a branchable run result"
    assert result.body["exit_code"] == 1
    assert result.body["stdout"] == ""
    assert "SyntaxError" in result.body["stderr"]


# ── 3. successful run with no output is an explicit error ─────────────


def test_no_output_returns_explicit_error():
    result = _run("x = 2 + 2")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT
    assert "print" in result.hint


# ── 4. missing code is an explicit error ──────────────────────────────


def test_missing_code_returns_explicit_error():
    for empty in (None, "", "   "):
        result = _run(empty)
        assert result.status == "error"
        assert result.code == "missing-params"
        assert result.body == CodeEvalAbility._ERR_NO_CODE


# ── sandbox invariants still hold ─────────────────────────────────────


def test_safe_module_still_usable():
    # math is pre-loaded as a global; the sandbox blocks `import`, so it is
    # used directly. The result is emitted via print.
    result = _run("print(math.sqrt(16))")
    assert result.status == "success"
    assert result.body["exit_code"] == 0
    assert result.body["stdout"].strip() == "4.0"


def test_import_is_blocked_and_reported():
    # The sandbox forbids imports; the attempt must surface as a stack trace on
    # stderr with a non-zero exit, not a silent empty success.
    result = _run("import os\nprint(os.getcwd())")
    assert result.status == "success", "blocked import is a branchable run result"
    assert result.body["exit_code"] == 1
    assert result.body["stderr"]


def test_no_path_returns_empty_string_loop_signal_is_gone():
    """Regression: a bare expression (no print) must NOT yield empty success."""
    result = _run("42")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT


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

    assert result.status == "error"
    assert result.code == "timeout"
    assert result.body == CodeEvalAbility._ERR_TIMEOUT
    assert result.hint
    assert elapsed < 30, "the runaway loop must be killed promptly, not run on"


def test_long_output_round_trips_and_truncates_through_subprocess():
    """A large result returns across the process boundary (proof the
    read-before-join path returns rather than deadlocking) and is clipped to the
    shared cap with ``truncated=true`` meta — not silently dropped."""
    result = _run("print('x' * 200000)")
    assert result.status == "success"
    assert result.body["exit_code"] == 0
    assert result.meta.get("truncated") is True
    assert len(result.body["stdout"]) <= 100 * 1024
    assert result.body["stdout"].startswith("x")

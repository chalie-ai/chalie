"""Unit tests for CodeEvalAbility — the sandbox result contract.

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
from typing import Optional, cast

import pytest

from abilities.code_eval import CodeEvalAbility
from abilities._result import ToolResult

pytestmark = pytest.mark.unit


def _run(code: Optional[str]) -> ToolResult:
    params: dict[str, object] = {} if code is None else {"code": code}
    return CodeEvalAbility().run(params)


# ── 1. printed output is a branchable run result ──────────────────────


def test_printed_output_is_returned() -> None:
    result = _run("print(2 + 2)")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["stdout"] == "4\n"
    assert cast("dict[str, object]", result.body)["stderr"] == ""
    assert cast("dict[str, object]", result.body)["exit_code"] == 0
    assert isinstance(cast("dict[str, object]", result.body)["duration_ms"], int)
    assert cast(int, cast("dict[str, object]", result.body)["duration_ms"]) >= 0


def test_multiple_prints_all_captured() -> None:
    result = _run("print('a')\nprint('b')")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["stdout"] == "a\nb\n"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0


# ── 2. runtime errors are exit_code 1 with the full traceback on stderr ──


def test_runtime_error_returns_full_traceback() -> None:
    result = _run("print(1 / 0)")
    assert result.status == "success", "anything that RAN is a branchable success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert "Traceback (most recent call last)" in cast(str, cast("dict[str, object]", result.body)["stderr"])
    assert "ZeroDivisionError" in cast(str, cast("dict[str, object]", result.body)["stderr"])


def test_runtime_error_preserves_partial_output() -> None:
    result = _run("print('before')\nraise ValueError('boom')")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert "ValueError" in cast(str, cast("dict[str, object]", result.body)["stderr"])
    assert "boom" in cast(str, cast("dict[str, object]", result.body)["stderr"])
    # Partial print output is preserved on stdout, NOT folded into the traceback.
    assert cast("dict[str, object]", result.body)["stdout"] == "before\n"


def test_syntax_error_returns_traceback() -> None:
    result = _run("print(")
    assert result.status == "success", "syntax error is a branchable run result"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert cast("dict[str, object]", result.body)["stdout"] == ""
    assert "SyntaxError" in cast(str, cast("dict[str, object]", result.body)["stderr"])


# ── 3. successful run with no output is an explicit error ─────────────


def test_no_output_returns_explicit_error() -> None:
    result = _run("x = 2 + 2")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT
    assert "print" in cast(str, result.hint)


# ── 4. missing code is an explicit error ──────────────────────────────


def test_missing_code_returns_explicit_error() -> None:
    for empty in (None, "", "   "):
        result = _run(empty)
        assert result.status == "error"
        assert result.code == "missing-params"
        assert result.body == CodeEvalAbility._ERR_NO_CODE


# ── sandbox invariants still hold ─────────────────────────────────────


def test_safe_module_still_usable() -> None:
    # math is pre-loaded as a global; the sandbox blocks `import`, so it is
    # used directly. The result is emitted via print.
    result = _run("print(math.sqrt(16))")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0
    assert cast(str, cast("dict[str, object]", result.body)["stdout"]).strip() == "4.0"


def test_import_is_blocked_and_reported() -> None:
    # The sandbox forbids imports; the attempt must surface as a stack trace on
    # stderr with a non-zero exit, not a silent empty success.
    result = _run("import os\nprint(os.getcwd())")
    assert result.status == "success", "blocked import is a branchable run result"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert cast("dict[str, object]", result.body)["stderr"]


def test_no_path_returns_empty_string_loop_signal_is_gone() -> None:
    """Regression: a bare expression (no print) must NOT yield empty success."""
    result = _run("42")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT


# ── augmented assignment runs ( CE-1) ─────────────────────────


def test_augmented_assignment_runs() -> None:
    """Regression ( CE-1): RestrictedPython rewrites ``x += 1`` into
    ``x = _inplacevar_("+=", x, 1)``. The sandbox shipped no ``_inplacevar_`` in its
    globals, so EVERY augmented assignment died with
    ``NameError: name '_inplacevar_' is not defined`` — a ubiquitous Python idiom
    silently broken. Drive the real run() over a spread of in-place ops and the
    exact ``*=`` accumulator loop the ticket reproduced (25!), asserting the
    arithmetic actually executed."""
    result = _run("x = 0\nx += 5\nx *= 3\nx -= 1\nx //= 2\nprint(x)")
    assert result.status == "success"
    body = cast("dict[str, object]", result.body)
    assert body["exit_code"] == 0, cast("dict[str, object]", result.body)["stderr"]
    assert cast(str, body["stdout"]).strip() == "7"  # ((0+5)*3 - 1) // 2 == 7

    factorial = _run("result = 1\nfor i in range(1, 26):\n    result *= i\nprint(result)")
    assert factorial.status == "success"
    fbody = cast("dict[str, object]", factorial.body)
    assert fbody["exit_code"] == 0, fbody["stderr"]
    assert cast(str, fbody["stdout"]).strip() == "15511210043330985984000000"


# ── hard wall-clock cap: runaway code is force-killed ─────────────────


def test_runaway_loop_is_terminated_with_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_long_output_round_trips_and_truncates_through_subprocess() -> None:
    """A large result returns across the process boundary (proof the
    read-before-join path returns rather than deadlocking) and is clipped to the
    shared cap with ``truncated=true`` meta — not silently dropped."""
    result = _run("print('x' * 200000)")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0
    assert result.meta.get("truncated") is True
    assert len(cast(str, cast("dict[str, object]", result.body)["stdout"])) <= 100 * 1024
    assert cast(str, cast("dict[str, object]", result.body)["stdout"]).startswith("x")

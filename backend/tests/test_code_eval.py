"""Feature tests for CodeEvalAbility — the sandbox result contract.

Run-faithful: anything that RAN returns ``ok`` with a branchable ``exit_code``
(exactly how ``deno run script.ts`` behaves) — clean exit 0 with ``stdout``, or
exit 1 with the error on ``stderr`` and any partial prints preserved on
``stdout``. Errors are reserved for HARNESS failures: missing code, no output,
timeout, missing runtime, sandbox crash — each a stable kebab code, never the
banned ``code="error"``. A silent empty success previously caused the LLM to
retry the same call until the ACT-loop iteration wall, so the no-output
guardrail is preserved as an explicit error.

``run()`` returns a ``ToolResult``: success carries the structured run result in
``body`` (``{stdout, stderr, exit_code, duration_ms}``); harness failures are
``status == "error"`` with a ``code``. These assert that contract directly.

These exercise the REAL production hot path — the actual ``deno run``
subprocess with zero permissions — no mocks.
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
    result = _run("console.log(2 + 2)")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["stdout"] == "4\n"
    assert cast("dict[str, object]", result.body)["stderr"] == ""
    assert cast("dict[str, object]", result.body)["exit_code"] == 0
    assert isinstance(cast("dict[str, object]", result.body)["duration_ms"], int)
    assert cast(int, cast("dict[str, object]", result.body)["duration_ms"]) >= 0


def test_multiple_console_logs_all_captured() -> None:
    result = _run("console.log('a')\nconsole.log('b')")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["stdout"] == "a\nb\n"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0


def test_typescript_types_are_evaluated() -> None:
    result = _run("const x: number = 6 * 7\nconsole.log(x)")
    assert result.status == "success"
    assert cast(str, cast("dict[str, object]", result.body)["stdout"]).strip() == "42"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0


# ── 2. runtime errors are exit_code 1 with the error on stderr ──────────


def test_runtime_error_returns_full_traceback() -> None:
    result = _run("throw new Error('boom')")
    assert result.status == "success", "anything that RAN is a branchable success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert "boom" in cast(str, cast("dict[str, object]", result.body)["stderr"])


def test_runtime_error_preserves_partial_output() -> None:
    result = _run("console.log('before')\nthrow new Error('boom')")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert "boom" in cast(str, cast("dict[str, object]", result.body)["stderr"])
    # Partial print output is preserved on stdout, NOT folded into the error.
    assert cast("dict[str, object]", result.body)["stdout"] == "before\n"


def test_syntax_error_returns_traceback() -> None:
    result = _run("console.log(")
    assert result.status == "success", "syntax error is a branchable run result"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert cast("dict[str, object]", result.body)["stdout"] == ""
    assert cast("dict[str, object]", result.body)["stderr"]


# ── 3. successful run with no output is an explicit error ─────────────


def test_no_output_returns_explicit_error() -> None:
    result = _run("const x = 2 + 2")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT
    assert "console.log" in cast(str, result.hint)


# ── 4. missing code is an explicit error ──────────────────────────────


def test_missing_code_returns_explicit_error() -> None:
    for empty in (None, "", "   "):
        result = _run(empty)
        assert result.status == "error"
        assert result.code == "missing-params"
        assert result.body == CodeEvalAbility._ERR_NO_CODE


# ── sandbox invariants: zero permissions ─────────────────────────────


def test_network_access_is_blocked() -> None:
    # The sandbox forbids network access; the attempt must surface as a
    # non-zero exit with the permission denial on stderr, not silent success.
    result = _run('const r = await fetch("https://example.com"); console.log(r.status)')
    assert result.status == "success", "blocked network is a branchable run result"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert cast("dict[str, object]", result.body)["stderr"]


def test_file_write_is_blocked() -> None:
    result = _run('Deno.writeTextFileSync("/tmp/chalie-pwned.txt", "hack")')
    assert result.status == "success", "blocked file write is a branchable run result"
    assert cast("dict[str, object]", result.body)["exit_code"] == 1
    assert cast("dict[str, object]", result.body)["stderr"]


def test_bare_expression_is_no_output() -> None:
    """Regression: a bare expression (no console.log) must NOT yield empty success."""
    result = _run("42")
    assert result.status == "error"
    assert result.code == "no-output"
    assert result.body == CodeEvalAbility._ERR_NO_OUTPUT


# ── hard wall-clock cap: runaway code is force-killed ─────────────────


def test_runaway_loop_is_terminated_with_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-terminating loop is force-killed at the cap and returns the
    actionable timeout error — the ACT loop never hangs.

    The real subprocess is spawned and terminated; only the cap length is
    shortened (to keep the suite fast) so this exercises the genuine kill path.
    """
    monkeypatch.setattr(CodeEvalAbility, "_EXEC_TIMEOUT_S", 0.5)

    start = time.monotonic()
    result = _run("while (true) {}")
    elapsed = time.monotonic() - start

    assert result.status == "error"
    assert result.code == "timeout"
    assert result.body == CodeEvalAbility._ERR_TIMEOUT
    assert result.hint
    assert elapsed < 30, "the runaway loop must be killed promptly, not run on"


def test_long_output_round_trips_and_truncates_through_subprocess() -> None:
    """A large result returns across the process boundary and is clipped to the
    shared cap with ``truncated=true`` meta — not silently dropped."""
    result = _run("console.log('x'.repeat(200000))")
    assert result.status == "success"
    assert cast("dict[str, object]", result.body)["exit_code"] == 0
    assert result.meta.get("truncated") is True
    assert len(cast(str, cast("dict[str, object]", result.body)["stdout"])) <= 100 * 1024
    assert cast(str, cast("dict[str, object]", result.body)["stdout"]).startswith("x")


# ── missing runtime is an actionable error ─────────────────────────────


def test_missing_runtime_returns_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Deno is not on PATH, the tool returns a clear ``no-runtime`` error
    rather than crashing or silently failing."""
    import abilities.code_eval as ce

    monkeypatch.setattr(ce, "_DENO_BIN", "definitely-not-denooo")
    result = _run("console.log(1)")
    assert result.status == "error"
    assert result.code == "no-runtime"
    assert "Deno" in cast(str, result.body)
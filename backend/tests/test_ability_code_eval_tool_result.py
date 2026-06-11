# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the code_eval tool's ToolResult contract (TKT-917).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch("code_eval", params)`` chokepoint on the CHAT
channel against a real ``mp``-shaped context, the real ``AbilityRegistry``
resolution of the production ``CodeEvalAbility``, the real ``PolicyManager.wrap``
gate (code_eval seeds ``allow`` on chat in the db template — it is NOT in
``PolicyManager.INTERNAL``, so the real gate actually runs through to run()), the
real ``CodeEvalAbility.run`` driving a REAL ``spawn`` subprocess sandbox, and the
real ``ActTrail`` write.

TKT-917 migrates code_eval to the CPython-faithful contract: anything that RAN
returns ``ok`` with a branchable ``exit_code`` (exactly like ``python script.py``),
and ``code="error"`` is gone. Errors are reserved for harness failures
(missing-params / no-output / timeout / sandbox-crashed). Pins:

  * **Success envelope.** ``print(2+2)`` → ``status=success`` with a compact-JSON
    body ``{"stdout":"4\\n","stderr":"","exit_code":0,"duration_ms":<int>=0}``.
  * **Runtime exception.** ``print('before')`` then ``1/0`` → ``status=success``,
    ``exit_code`` 1, the ZeroDivisionError traceback in ``stderr``, and the partial
    ``before`` print preserved in ``stdout`` (NOT folded into one blob).
  * **Syntax error.** ``print(`` → ``exit_code`` 1, SyntaxError in ``stderr``,
    ``stdout`` empty.
  * **Missing code (pre-gate).** ``{}`` → the dispatcher ACTION_REQUIRED pre-gate
    rejects it as ``code=missing-params`` BEFORE run().
  * **Whitespace-only code.** ``"   "`` is normalised to empty by the production
    sanitize step, so the pre-gate rejects it as ``code=missing-params`` (run()'s
    own residue guard is pinned directly in ``test_code_eval.py``).
  * **No-output guardrail.** ``x = 2 + 2`` ran to completion but printed nothing →
    ``code=no-output`` with a print hint (the guardrail that killed a real
    retry-until-iteration-wall loop).
  * **Timeout.** ``while True: pass`` under a shortened cap → ``code=timeout`` with
    a hint, force-killed promptly.
  * **Truncation.** ``print('x'*200000)`` → ``status=success`` with
    ``truncated=true`` in the head and the stdout clipped to the cap.

OFFLINE-DETERMINISTIC: the sandbox subprocess is real and local — no network. The
success/exception/syntax/no-output/truncation bodies are compact JSON; errors are
prose with a kebab code in the head. ``code=error`` appears nowhere.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities.code_eval import CodeEvalAbility
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP, body, head, parse_body, seed_transcript

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor. code_eval seeds ``allow`` on chat in the db template, so the real gate
    passes through to the production run()."""
    return MP(seed_transcript(db, "chat", "run some python"), UserConfig({}))


def _head(rendered: str) -> str:
    return head(rendered, "code_eval")


def _body(rendered: str) -> str:
    """Extract the body between the open tag and the ``[end:code_eval]`` close."""
    return body(rendered, "code_eval")


def _body_json(rendered: str) -> dict:
    """Parse a success body (compact JSON) into a dict."""
    return parse_body(rendered, "code_eval")


# ── 1. success envelope: ran, printed, exit 0, branchable structured body ───────


def test_success_returns_structured_run_result(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "print(2 + 2)", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=success" in head
    assert "code=error" not in out

    body = _body_json(out)
    assert body["stdout"] == "4\n"
    assert body["stderr"] == ""
    assert body["exit_code"] == 0
    assert isinstance(body["duration_ms"], int)
    assert body["duration_ms"] >= 0


# ── 2. runtime exception: exit_code 1, traceback in stderr, partial stdout kept ──


def test_runtime_exception_is_branchable_success(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "print('before')\n1 / 0", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=success" in head, "anything that RAN is a branchable success"
    assert "code=error" not in out

    body = _body_json(out)
    assert body["exit_code"] == 1
    assert body["stdout"] == "before\n", "partial print preserved in stdout, not blobbed"
    assert "Traceback (most recent call last)" in body["stderr"]
    assert "ZeroDivisionError" in body["stderr"]


# ── 3. syntax error: exit_code 1, SyntaxError in stderr, empty stdout ───────────


def test_syntax_error_is_branchable_success(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "print(", "act_summary": "x"}
    )

    assert "status=success" in _head(out)
    assert "code=error" not in out

    body = _body_json(out)
    assert body["exit_code"] == 1
    assert body["stdout"] == ""
    assert "SyntaxError" in body["stderr"]


# ── 4. missing code param → dispatcher pre-gate missing-params, nothing runs ─────


def test_missing_code_is_pre_gate_missing_params(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch("code_eval", {"act_summary": "x"})

    head = _head(out)
    assert "status=error" in head
    assert "code=missing-params" in head
    assert "code=error" not in out
    assert "[end:code_eval]" in out


# ── 5. whitespace-only code → missing-params on the real dispatch path ──────────


def test_whitespace_only_code_is_missing_params(db, chat_mp):
    """A whitespace-only ``code`` is normalised to empty by the production
    ``_sanitize_llm_args`` step the dispatcher runs first, so the ACTION_REQUIRED
    pre-gate catches it as ``code=missing-params`` — the same actionable signal,
    never the banned ``code=error``. (run()'s own residue guard, which carries the
    print hint, is pinned directly in ``test_code_eval.py``.)"""
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "   ", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=missing-params" in head
    assert "code=error" not in out
    assert "code" in _body(out)
    assert "[end:code_eval]" in out


# ── 6. no-output guardrail: ran to completion but printed nothing ───────────────


def test_no_output_guardrail_is_explicit_error(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "x = 2 + 2", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=no-output" in head
    assert "code=error" not in out
    hint_lines = [ln for ln in out.splitlines() if ln.startswith("hint:")]
    assert hint_lines and "print" in hint_lines[0]


# ── 7. timeout: runaway loop force-killed promptly with an actionable hint ───────


def test_runaway_loop_is_timeout_error(db, chat_mp, monkeypatch):
    """A non-terminating loop is force-killed at the cap and returns the actionable
    timeout error — the ACT loop never hangs. Only the cap length is shortened so
    this still exercises the genuine kill path through the real subprocess."""
    import time

    monkeypatch.setattr(CodeEvalAbility, "_EXEC_TIMEOUT_S", 0.5)
    monkeypatch.setattr(CodeEvalAbility, "_POLL_INTERVAL_S", 0.1)

    start = time.monotonic()
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "while True:\n    pass", "act_summary": "x"}
    )
    elapsed = time.monotonic() - start

    head = _head(out)
    assert "status=error" in head
    assert "code=timeout" in head
    assert "code=error" not in out
    assert any(ln.startswith("hint:") for ln in out.splitlines())
    assert elapsed < 30, "the runaway loop must be killed promptly, not run on"


# ── 8. truncation: large stdout clipped to the cap, truncated=true, no legacy ────


def test_large_output_is_truncated(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "print('x' * 200000)", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=success" in head
    assert "truncated=true" in head
    assert "code=error" not in out

    body = _body_json(out)
    assert body["exit_code"] == 0
    # Clipped to the shared 100KB cap — far below the 200000 emitted.
    assert len(body["stdout"]) <= 100 * 1024


# ── 9. an error dispatch still records the rendered envelope on the act-trail ────


def test_error_dispatch_writes_act_trail(db, chat_mp):
    ToolDispatcher(chat_mp).dispatch(
        "code_eval", {"code": "x = 2 + 2", "act_summary": "x"}
    )

    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any(
        "[code_eval(status=error, code=no-output" in row["result"] for row in trail
    )


# ── 10. registry metadata + frozen single-param schema ──────────────────────────


def test_code_eval_registered_with_metadata():
    ability = AbilityRegistry.get("code_eval")
    assert ability.get_name() == "code_eval"
    assert ability.get_summary()
    assert ability.get_examples()
    assert ability.get_search_tooltip()

    schema = ability.get_parameters()
    assert schema["required"] == ["code"]
    assert list(schema["properties"]) == ["code"]

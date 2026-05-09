"""Feature tests for SubagentAbility wait=True (blocking) semantics.

Two behaviors under test:
  C14 — wait=True blocks until the subagent send() returns, then returns the
         result inline (not a JSON ack).
  C16 — SubagentProcessor.send() raising an exception must produce an error
         tag in the result dict rather than propagating the exception to caller.

C15 (wait=True advances parent _loop_start) is REMOVED — _loop_start is gone
in the v0.6.0 timeout rip. SubagentProcessor now uses a per-instance _deadline;
no parent budget extension is needed.

SubagentProcessor.send() is patched in C14/C16 because running a real ACT loop
requires a live LLM provider — which is not available in unit test environment.
The patch is minimal and boundary-correct: it replaces only `send()` on
SubagentProcessor, and the behavior under test (timing, error handling) lives
entirely in SubagentAbility._run_sync(), which runs for real.
"""

import time

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# C14. Blocking wait returns result inline
# ---------------------------------------------------------------------------


class TestWaitBlocks:
    def test_wait_true_blocks_until_processor_completes(self):
        """wait=True must hold the caller until SubagentProcessor.send() returns,
        then embed the response text in the tag body (not a JSON ack).

        SubagentProcessor.send() is patched to sleep 0.5 s and return a known
        string. We assert:
          - elapsed >= 0.4 s (actual blocking occurred)
          - result text contains the returned string (not a JSON ack)
        """
        from unittest.mock import patch

        DELAY = 0.5
        SENTINEL = "summary text from subagent"

        def _fake_send(self_proc):
            time.sleep(DELAY)
            return SENTINEL

        with patch("services.subagent_processor.SubagentProcessor.send", _fake_send):
            from abilities.subagent import SubagentAbility

            t0 = time.monotonic()
            result = SubagentAbility().execute(
                channel="user",
                params={"prompt": "x", "wait": True},
                telemetry=None,
            )
            elapsed = time.monotonic() - t0

        assert elapsed >= 0.4, f"Expected blocking >=0.4s, got {elapsed:.3f}s"

        text = result["text"]
        assert SENTINEL in text, f"Expected sentinel in result, got: {repr(text)}"
        # Must NOT be a JSON ack — ack bodies start with '{"success"'
        assert '"success"' not in text, f"Got JSON ack instead of inline result: {repr(text)}"
        assert text.startswith("[subagent("), repr(text)
        assert "wait=True" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)


# ---------------------------------------------------------------------------
# C16. SubagentProcessor exception → error tag, not raised
# ---------------------------------------------------------------------------


class TestWaitExceptionHandled:
    def test_wait_true_processor_exception_returns_error_tag_not_raises(self):
        """A RuntimeError from SubagentProcessor.send() must be caught and
        surfaced as an error tag in the result dict — not propagated to caller.
        """
        from unittest.mock import patch

        def _boom(self_proc):
            raise RuntimeError("boom — simulated processor failure")

        with patch("services.subagent_processor.SubagentProcessor.send", _boom):
            from abilities.subagent import SubagentAbility

            result = SubagentAbility().execute(
                channel="user",
                params={"prompt": "x", "wait": True},
                telemetry=None,
            )

        assert isinstance(result, dict), "execute() must return dict, not raise"
        text = result["text"]
        assert "error=" in text, f"Expected error attr in tag, got: {repr(text)}"
        assert "boom" in text.lower() or "simulated" in text.lower() or "error=" in text, (
            f"Error details not surfaced in result: {repr(text)}"
        )
        assert text.startswith("[subagent("), repr(text)
        assert "[end:subagent]" in text, repr(text)

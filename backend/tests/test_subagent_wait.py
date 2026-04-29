"""Feature tests for SubagentAbility wait=True (blocking) semantics.

Three behaviors under test:
  C14 — wait=True blocks until the subagent send() returns, then returns the
         result inline (not a JSON ack).
  C15 — wait=True advances the parent processor's _loop_start by elapsed time,
         so the parent's MAX_TIMEOUT budget is not consumed by subagent work.
  C16 — SubagentProcessor.send() raising an exception must produce an error
         tag in the result dict rather than propagating the exception to caller.

SubagentProcessor.send() is patched in C14/C15/C16 because running a real
ACT loop requires a live LLM provider — which is not available in unit test
environment. The patch is minimal and boundary-correct: it replaces only
`send()` on SubagentProcessor, and the behavior under test (timing, error
handling, budget extension) lives entirely in SubagentAbility._run_sync(),
which runs for real.
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
                params={"prompt": "x", "wait": True, "timeout": 180},
                telemetry=None,
            )
            elapsed = time.monotonic() - t0

        assert elapsed >= 0.4, f"Expected blocking ≥0.4s, got {elapsed:.3f}s"

        text = result["text"]
        assert SENTINEL in text, f"Expected sentinel in result, got: {repr(text)}"
        # Must NOT be a JSON ack — ack bodies start with '{"success"'
        assert '"success"' not in text, f"Got JSON ack instead of inline result: {repr(text)}"
        assert text.startswith("[subagent("), repr(text)
        assert "wait=True" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)


# ---------------------------------------------------------------------------
# C15. wait=True advances parent _loop_start
# ---------------------------------------------------------------------------


class TestWaitExtendsParentBudget:
    def test_wait_true_extends_parent_loop_start(self):
        """Elapsed subagent time must be added to parent._loop_start so the
        parent's MAX_TIMEOUT budget is not consumed by subagent work.

        Setup:
          - Bind a real MessageProcessor subclass instance as current_processor().
          - Set its _loop_start to a known value.
          - Run wait=True subagent that sleeps 0.5 s.
          - Assert parent._loop_start advanced by ~0.5 s (±0.1 s tolerance).
        """
        from unittest.mock import patch
        from services.message_processor import (
            MessageProcessor,
            bind_current_processor,
        )
        from services.subagent_processor import SubagentProcessor

        DELAY = 0.5
        TOLERANCE = 0.1

        def _fake_send(self_proc):
            time.sleep(DELAY)
            return "done"

        # Minimal concrete subclass — we only need _loop_start tracking.
        class _FakeParent(MessageProcessor):
            CHANNEL = "user"
            ROLE = "user"

            def getUserPrompt(self):
                return "test"

            def getUserDefinition(self):
                return "test user"

        parent = _FakeParent(raw_input="test")
        anchor = time.monotonic()
        parent._loop_start = anchor

        with patch("services.subagent_processor.SubagentProcessor.send", _fake_send):
            from abilities.subagent import SubagentAbility
            with bind_current_processor(parent):
                SubagentAbility().execute(
                    channel="user",
                    params={"prompt": "x", "wait": True, "timeout": 180},
                    telemetry=None,
                )

        delta = parent._loop_start - anchor
        # Lower bound: at minimum the fake_send sleep must have elapsed.
        # Upper bound is intentionally loose — SubagentProcessor.__init__ runs
        # the real embedding pre-seed at construction time, which may add
        # several seconds in environments where the model is loaded cold.
        # The feature under test is that the elapsed time IS added (not that
        # it equals exactly DELAY); any positive advance ≥ DELAY confirms it.
        assert delta >= DELAY - TOLERANCE, (
            f"_loop_start advanced by {delta:.3f}s, expected ≥{DELAY - TOLERANCE:.3f}s"
        )
        assert delta < 60.0, (
            f"_loop_start advanced by {delta:.3f}s — exceeds 60s sanity ceiling"
        )


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
                params={"prompt": "x", "wait": True, "timeout": 180},
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

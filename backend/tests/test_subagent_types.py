"""Feature tests for the v0.6.0 subagent async-return flow, SubagentReturnProcessor,
and pending_steers drain.

Covers:
  - _deliver_envelope Case A: mid-ACT parent appends to _pending_steers
  - _deliver_envelope Case B: idle parent spawns a daemon thread via _spawn_return_processor
  - SubagentReturnProcessor inherits UserMessageProcessor with ROLE=subagent_return
  - _pending_steers drained into _act_trail by UMP.getUserPrompt()
"""

import threading

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# T1. _deliver_envelope Case A — mid-ACT parent gets envelope in _pending_steers
# ---------------------------------------------------------------------------


class TestAsyncSteerCaseA:
    def test_async_steer_appends_to_parent_pending_steers(self):
        """When parent is a UserMessageProcessor, _deliver_envelope appends the
        envelope to parent._pending_steers (not directly to _act_trail)."""
        from abilities.subagent import _deliver_envelope
        from services.user_message_processor import UserMessageProcessor

        # Minimal UMP subclass — overrides the abstract methods with no-ops.
        class _StubUMP(UserMessageProcessor):
            def getUserPrompt(self):
                return "test"

            def getUserDefinition(self):
                return "test user"

        parent = _StubUMP(raw_input="test")
        # Simulate mid-ACT state: iteration > 0
        parent._current_iteration = 2

        envelope = "[subagent.complete(type=general_purpose)]\nresult\n[end:subagent.complete]"
        _deliver_envelope(envelope, parent)

        assert envelope in parent._pending_steers, (
            f"Envelope not found in _pending_steers: {parent._pending_steers}"
        )
        # Must NOT have gone directly to _act_trail
        assert envelope not in parent._act_trail, (
            "Envelope was placed in _act_trail directly (should be _pending_steers)"
        )


# ---------------------------------------------------------------------------
# T2. _deliver_envelope Case B — idle parent spawns daemon thread
# ---------------------------------------------------------------------------


class TestAsyncSteerCaseB:
    def test_spawn_return_processor_returns_daemon_thread(self):
        """_spawn_return_processor() returns a daemon thread named 'subagent-return'
        whose target is callable. We assert spawn mechanics — not the thread's
        execution (that requires a live LLM + DB and is covered by nightly scenarios).
        """
        from abilities.subagent import _spawn_return_processor

        envelope = "[subagent.complete(type=web_surfer)]\nbody\n[end:subagent.complete]"

        # Capture threads spawned during this call
        before_names = {t.name for t in threading.enumerate()}
        _spawn_return_processor(envelope)
        after_threads = {t for t in threading.enumerate() if t.name not in before_names}

        spawned = [t for t in after_threads if t.name == "subagent-return"]
        assert len(spawned) == 1, (
            f"Expected exactly 1 'subagent-return' thread, found: {[t.name for t in after_threads]}"
        )
        assert spawned[0].daemon is True, (
            "subagent-return thread must be a daemon so it does not block process exit"
        )
        assert callable(spawned[0]._target), (
            "subagent-return thread must have a callable target"
        )

    def test_deliver_envelope_with_none_parent_routes_to_case_b(self):
        """_deliver_envelope(envelope, parent_ref=None) takes Case B path — the
        non-UMP branch — and does not raise. The spawn mechanics are verified by
        test_spawn_return_processor_returns_daemon_thread above."""
        from abilities.subagent import _deliver_envelope

        envelope = "[subagent.complete(type=general_purpose)]\nbody\n[end:subagent.complete]"
        # Must not raise; daemon thread is fire-and-forget
        _deliver_envelope(envelope, parent_ref=None)


# ---------------------------------------------------------------------------
# T3. SubagentReturnProcessor inherits UserMessageProcessor
# ---------------------------------------------------------------------------


class TestSubagentReturnProcessor:
    def test_subagent_return_processor_inherits_user_message_processor(self):
        """SubagentReturnProcessor is a UserMessageProcessor subclass with
        ROLE='subagent_return' and inherits CHANNEL='user'."""
        from services.user_message_processor import (
            SubagentReturnProcessor,
            UserMessageProcessor,
        )

        assert issubclass(SubagentReturnProcessor, UserMessageProcessor), (
            "SubagentReturnProcessor must inherit UserMessageProcessor"
        )

        assert SubagentReturnProcessor.ROLE == "subagent_return", (
            f"Expected ROLE='subagent_return', got '{SubagentReturnProcessor.ROLE}'"
        )
        assert SubagentReturnProcessor.CHANNEL == "user", (
            f"Expected CHANNEL='user' (inherited), got '{SubagentReturnProcessor.CHANNEL}'"
        )

    def test_subagent_return_processor_can_be_instantiated(self):
        """SubagentReturnProcessor can be constructed with raw_input only."""
        from services.user_message_processor import SubagentReturnProcessor

        proc = SubagentReturnProcessor(
            raw_input="[subagent.complete(type=general_purpose)]\nresult\n[end:subagent.complete]"
        )
        assert proc.ROLE == "subagent_return"
        assert proc.CHANNEL == "user"


# ---------------------------------------------------------------------------
# T4. _pending_steers drained into _act_trail by UMP.getUserPrompt()
# ---------------------------------------------------------------------------


class TestPendingSteersDrain:
    def test_pending_steers_drained_into_act_trail_by_get_user_prompt(self, db):
        """_pending_steers accumulated between iterations are drained into
        _act_trail by UMP.getUserPrompt(), then _pending_steers is cleared.

        getUserPrompt() reads real services (data_graph, world_state, etc.) against
        a real in-memory SQLite DB. The drain fires at step 6 of 7 regardless of
        what the prior steps return — we let them run against empty tables and assert
        the drain state at the end.
        """
        from services.user_message_processor import UserMessageProcessor

        class _StubUMP(UserMessageProcessor):
            def getUserDefinition(self):
                # Override DB read with a fast static fallback so the test
                # does not depend on data_graph having a user_summary row.
                return "Test user."

        proc = _StubUMP(raw_input="test prompt")

        env_a = "[subagent.complete(type=web_surfer)]\nresult a\n[end:subagent.complete]"
        env_b = "[subagent.complete(type=summariser)]\nresult b\n[end:subagent.complete]"
        proc._pending_steers = [env_a, env_b]

        # Run getUserPrompt() against real services. It may raise after the
        # drain (e.g. if some later step has a dependency we have not seeded),
        # but the drain itself is unconditional — assert state in a finally block.
        try:
            proc.getUserPrompt()
        except Exception:
            pass

        assert env_a in proc._act_trail, (
            f"env_a not in _act_trail after getUserPrompt(). "
            f"_act_trail={proc._act_trail}, _pending_steers={proc._pending_steers}"
        )
        assert env_b in proc._act_trail, (
            f"env_b not in _act_trail after getUserPrompt(). "
            f"_act_trail={proc._act_trail}, _pending_steers={proc._pending_steers}"
        )
        assert proc._pending_steers == [], (
            f"_pending_steers not cleared after drain: {proc._pending_steers}"
        )

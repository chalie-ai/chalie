"""Feature tests for subagent async-return flow and _pending_steers drain.

Covers:
  - _pending_steers drained into _act_trail by UMP.get_user_prompt() (north star
    invariant: steers accumulated mid-ACT must be visible to the next iteration)
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# T4. _pending_steers drained into _act_trail by UMP.get_user_prompt()
# ---------------------------------------------------------------------------


class TestPendingSteersDrain:
    def test_pending_steers_drained_into_act_trail_by_get_user_prompt(self, db):
        """_pending_steers accumulated between iterations are drained into
        _act_trail by UMP.get_user_prompt(), then _pending_steers is cleared.

        get_user_prompt() reads real services (data_graph, world_state, etc.) against
        a real in-memory SQLite DB. The drain fires at step 6 of 7 regardless of
        what the prior steps return — we let them run against empty tables and assert
        the drain state at the end.
        """
        from services.user_message_processor import UserMessageProcessor

        class _StubUMP(UserMessageProcessor):
            def get_user_definition(self):
                # Override DB read with a fast static fallback so the test
                # does not depend on data_graph having a user_summary row.
                return "Test user."

        proc = _StubUMP(raw_input="test prompt")

        env_a = "[subagent.complete(type=web_surfer)]\nresult a\n[end:subagent.complete]"
        env_b = "[subagent.complete(type=summariser)]\nresult b\n[end:subagent.complete]"
        proc._pending_steers = [env_a, env_b]

        # Run get_user_prompt() against real services. It may raise after the
        # drain (e.g. if some later step has a dependency we have not seeded),
        # but the drain itself is unconditional — assert state in a finally block.
        try:
            proc.get_user_prompt()
        except Exception:
            pass

        assert env_a in proc._act_trail, (
            f"env_a not in _act_trail after get_user_prompt(). "
            f"_act_trail={proc._act_trail}, _pending_steers={proc._pending_steers}"
        )
        assert env_b in proc._act_trail, (
            f"env_b not in _act_trail after get_user_prompt(). "
            f"_act_trail={proc._act_trail}, _pending_steers={proc._pending_steers}"
        )
        assert proc._pending_steers == [], (
            f"_pending_steers not cleared after drain: {proc._pending_steers}"
        )

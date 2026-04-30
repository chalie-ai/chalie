"""Feature tests for SubagentProcessor construction and deadline semantics.

Covers: per-instance _deadline calculation (type default, override, override-
smaller-than-default), and per-instance ALWAYS_AVAILABLE wiring from agent_type.
"""

import time

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# B1. Per-instance _deadline from agent_type
# ---------------------------------------------------------------------------


class TestDeadlineSemantics:
    def test_deadline_set_from_type_default_timeout(self):
        """SubagentProcessor with default agent_type sets _deadline to
        time.time() + general_purpose default_timeout (1800s), within 1s."""
        from abilities.subagent import SUBAGENT_TYPES
        from services.subagent_processor import SubagentProcessor

        t_before = time.time()
        proc = SubagentProcessor(raw_input="x")  # default agent_type=general_purpose
        t_after = time.time()

        expected = SUBAGENT_TYPES["general_purpose"]["default_timeout"]
        # _deadline should be ~t_before + expected
        assert proc._deadline >= t_before + expected - 1, (
            f"deadline {proc._deadline:.1f} < t_before + {expected} - 1"
        )
        assert proc._deadline <= t_after + expected + 1, (
            f"deadline {proc._deadline:.1f} > t_after + {expected} + 1"
        )

    def test_deadline_set_from_max_timeout_override(self):
        """max_timeout_override=300 must win over the type's default_timeout."""
        from services.subagent_processor import SubagentProcessor

        t_before = time.time()
        proc = SubagentProcessor(raw_input="x", max_timeout_override=300)

        delta = proc._deadline - t_before
        assert abs(delta - 300) <= 1.0, (
            f"deadline delta={delta:.3f}s, expected ~300s"
        )

    def test_deadline_override_smaller_than_type_default_honoured(self):
        """A small override (60s) on a type with 3600s default must use 60s."""
        from services.subagent_processor import SubagentProcessor

        t_before = time.time()
        proc = SubagentProcessor(
            raw_input="x", agent_type="web_surfer", max_timeout_override=60
        )
        delta = proc._deadline - t_before
        assert abs(delta - 60) <= 1.0, (
            f"deadline delta={delta:.3f}s, expected ~60s for override=60"
        )


# ---------------------------------------------------------------------------
# B3. Per-instance ALWAYS_AVAILABLE wired from agent_type
# ---------------------------------------------------------------------------


class TestProcessorWiring:
    def test_per_instance_always_available_is_set_from_agent_type(self):
        """After construction, ALWAYS_AVAILABLE on the instance must equal
        the type's native_tools (not the class-level empty list)."""
        from abilities.subagent import SUBAGENT_TYPES
        from services.subagent_processor import SubagentProcessor

        for agent_type, entry in SUBAGENT_TYPES.items():
            proc = SubagentProcessor(raw_input="x", agent_type=agent_type)
            # Instance attribute shadows the class attribute
            assert proc.ALWAYS_AVAILABLE == entry["native_tools"], (
                f"{agent_type}: instance ALWAYS_AVAILABLE {proc.ALWAYS_AVAILABLE} "
                f"!= native_tools {entry['native_tools']}"
            )

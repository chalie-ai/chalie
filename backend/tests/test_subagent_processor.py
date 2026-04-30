"""Feature tests for SubagentProcessor construction, deadline semantics,
blocklist enforcement, and class constants.

The pre-seed (_preseeded_tools) tests are REMOVED — pre-seeding was deleted in
the v0.6.0 subagent rework. The _max_timeout_seconds property tests are
REPLACED by _deadline semantics tests: instantiating SubagentProcessor with a
given agent_type sets _deadline to roughly time.time() + type_default_timeout.
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
# B2. Blocklist constant
# ---------------------------------------------------------------------------


class TestBlocklistConstant:
    def test_blocklist_constant_includes_subagent_save_graph_detect_pattern(self):
        from services.subagent_processor import _BLOCKED

        assert isinstance(_BLOCKED, frozenset)
        assert "subagent" in _BLOCKED
        assert "save_graph" in _BLOCKED
        assert "detect_pattern" in _BLOCKED


# ---------------------------------------------------------------------------
# B3. Class constants
# ---------------------------------------------------------------------------


class TestProcessorClassConstants:
    def test_processor_class_constants(self):
        """CHANNEL, ROLE, MAX_ITERATIONS present. MAX_TIMEOUT is gone.
        ALWAYS_AVAILABLE is [] at class level (set per-instance in __init__)."""
        from services.subagent_processor import SubagentProcessor, _BLOCKED

        assert SubagentProcessor.CHANNEL == "subagent"
        assert SubagentProcessor.ROLE == "subagent"
        assert SubagentProcessor.MAX_ITERATIONS == 50

        # MAX_TIMEOUT class constant must be GONE (ripped in §14.1)
        assert not hasattr(SubagentProcessor, "MAX_TIMEOUT"), (
            "MAX_TIMEOUT class constant should have been removed in the timeout rip"
        )

        # ALWAYS_AVAILABLE at class level must be [] — populated per-instance
        assert SubagentProcessor.ALWAYS_AVAILABLE == [], (
            f"Class-level ALWAYS_AVAILABLE should be [] (set per-instance), "
            f"got: {SubagentProcessor.ALWAYS_AVAILABLE}"
        )

        discoverable = SubagentProcessor.DISCOVERABLE
        expected_tools = [
            "browser", "code_eval", "document", "list", "memory", "news",
            "programming_docs_search", "read", "review_tool_calls",
            "schedule", "search", "weather",
        ]
        for name in expected_tools:
            assert name in discoverable, f"'{name}' missing from DISCOVERABLE"

        # No blocklisted names must appear in DISCOVERABLE
        for blocked in _BLOCKED:
            assert blocked not in discoverable, (
                f"Blocklisted '{blocked}' found in DISCOVERABLE"
            )

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

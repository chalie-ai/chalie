"""Feature tests for SubagentProcessor construction and deadline semantics.

Covers: per-instance _deadline calculation (override-wins-over-type-default
is the invariant that protects wait=true from exceeding the 300s cap), and
per-instance ALWAYS_AVAILABLE wiring from agent_type.
"""

import time

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# B1. Override timeout wins over type default
#
# The critical invariant: wait=true in SubagentAbility caps at 300s via
# max_timeout_override=min(type_default, 300). If the processor ignored the
# override and used the type default (3600s for web_surfer), the parent ACT
# iteration would block for up to an hour.
# ---------------------------------------------------------------------------


class TestDeadlineSemantics:
    def test_max_timeout_override_wins_over_type_default(self):
        """max_timeout_override=300 must produce a _deadline of t_now + 300,
        even for web_surfer (type default = 3600s).

        This guards the wait=true cap: SubagentAbility calls
        SubagentProcessor(..., max_timeout_override=min(type_default, 300))
        so the parent ACT iteration never blocks past 5 min."""
        from services.subagent_processor import SubagentProcessor

        t_before = time.time()
        proc = SubagentProcessor(
            raw_input="x", agent_type="web_surfer", max_timeout_override=300
        )
        t_after = time.time()

        delta = proc._deadline - t_before
        assert 299 <= delta <= t_after - t_before + 301, (
            f"deadline delta={delta:.3f}s, expected ~300s (override should win over "
            f"web_surfer default of 3600s)"
        )


# ---------------------------------------------------------------------------
# B3. Per-instance ALWAYS_AVAILABLE wired from agent_type
# ---------------------------------------------------------------------------


class TestProcessorWiring:
    def test_per_instance_always_available_is_set_from_agent_type(self):
        """After construction, ALWAYS_AVAILABLE on the instance must equal
        the type's native_tools (not the class-level empty list).

        This ensures each subagent type gets only its declared tool surface —
        a web_surfer must not have the summariser's tool set and vice versa."""
        from abilities.subagent import SUBAGENT_TYPES
        from services.subagent_processor import SubagentProcessor

        for agent_type, entry in SUBAGENT_TYPES.items():
            proc = SubagentProcessor(raw_input="x", agent_type=agent_type)
            assert proc.ALWAYS_AVAILABLE == entry["native_tools"], (
                f"{agent_type}: instance ALWAYS_AVAILABLE {proc.ALWAYS_AVAILABLE} "
                f"!= native_tools {entry['native_tools']}"
            )

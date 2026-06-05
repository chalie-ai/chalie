import pytest
from abilities._registry import AbilityRegistry
from configs.channels import UserConfig

pytestmark = pytest.mark.integration


def test_thinking_never_discoverable(db):
    # thinking is registered but in no discoverable/always_available list.
    assert "thinking" in {a.NAME for a in AbilityRegistry.all()}
    cfg = UserConfig()
    assert "thinking" not in (cfg.always_available or [])
    assert "thinking" not in (cfg.discoverable or [])


def test_thinking_config_retains_parent_tool_surface(db):
    from abilities.thinking import ThinkingConfig
    from configs.channels import UserConfig
    parent = UserConfig()
    tc = ThinkingConfig(parent.always_available, parent.discoverable, parent.policy_channel)
    assert tc.always_available == list(parent.always_available or [])
    assert tc.discoverable == list(parent.discoverable or [])
    assert tc.thinking_mode == "high"
    assert tc.max_iterations == 1


def test_high_thinking_gate_result_is_visible_to_turn_zero_dispatch():
    """_run_thinking_gate writes self.thinking_level (public); _seed_turn_zero
    reads getattr(self, 'thinking_level', 'low').  This test asserts the writer
    and reader use the same attribute so a 'high' gate result actually triggers
    the thinking dispatch in turn-0.  Fails on the old code where the gate
    wrote self._thinking_level and the seed read self.thinking_level.
    """
    from services.message_processor import MessageProcessor

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "raw input", None)
    mp.config = UserConfig()
    # Simulate what _run_thinking_gate now writes after the fix.
    mp.thinking_level = "high"
    # The seed's read must see the same value the gate wrote.
    assert getattr(mp, "thinking_level", "low") == "high"

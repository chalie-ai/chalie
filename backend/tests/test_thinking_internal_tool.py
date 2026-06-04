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

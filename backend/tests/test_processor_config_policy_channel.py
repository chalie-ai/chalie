"""ProcessorConfig.policy_channel enum + derived usage_class (PolicyManager redesign)."""
import pytest

from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


def _make(policy_channel):
    return ProcessorConfig(
        channel="user", role="user", policy_channel=policy_channel,
        build_user_prompt=lambda mp: "", build_user_definition=lambda mp: "",
        build_system_prompt=lambda mp: "", always_available=[], discoverable=[],
        blocked=frozenset(), max_iterations=None, skip_transcript=False,
        skip_input_row=False, suppress_history=False, broadcast_to=None,
        memory_seed=False, post_turn=None,
    )


def test_enum_has_four_values():
    vals = {c.value for c in ProcessorConfig.POLICY_CHANNEL}
    assert vals == {"chat", "subagent", "subconscious", "external_agent"}


def test_usage_class_derives_from_policy_channel():
    assert _make(ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS).usage_class == "subconscious"
    assert _make(ProcessorConfig.POLICY_CHANNEL.CHAT).usage_class == "chat"

"""ProcessorConfig.policy_channel enum + derived usage_class (PolicyManager redesign)."""
import pytest

from configs.enums.policy_channel import PolicyChannel
from tests.helpers import StubProcessorConfig

pytestmark = pytest.mark.unit


def _make(policy_channel: PolicyChannel) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel="user", role="user", policy_channel=policy_channel,
        always_available=[],
        skip_transcript=False,
        skip_input_row=False, suppress_history=False, broadcast_to=None,
        memory_seed=False,
    )


def test_enum_has_three_values() -> None:
    vals = {c.value for c in PolicyChannel}
    assert vals == {"chat", "subconscious", "external_agent"}


def test_usage_class_derives_from_policy_channel() -> None:
    assert _make(PolicyChannel.SUBCONSCIOUS).usage_class == "subconscious"
    assert _make(PolicyChannel.CHAT).usage_class == "chat"

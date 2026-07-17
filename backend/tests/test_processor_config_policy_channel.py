"""ProcessorConfig.policy_channel enum (PolicyManager redesign)."""
import pytest

from configs.enums.policy_channel import PolicyChannel

pytestmark = pytest.mark.unit


def test_enum_has_three_values() -> None:
    vals = {c.value for c in PolicyChannel}
    assert vals == {"chat", "subconscious", "external_agent"}

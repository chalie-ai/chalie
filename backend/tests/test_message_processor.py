
import pytest


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_config(
    *,
    channel="dmn",
    role="proactive_thought",
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
):

    from services.processor_config import ProcessorConfig
    from tests.helpers import StubProcessorConfig

    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
        build_user_prompt=lambda _mp: "user body",
        build_user_definition=lambda _mp: "user definition",
        build_system_prompt=lambda _mp: "system",
        always_available=[],
        skip_transcript=skip_transcript,
        skip_input_row=skip_input_row,
        suppress_history=suppress_history,
        broadcast_to=broadcast_to,
        memory_seed=memory_seed,
    )


# ---------------------------------------------------------------------------
# suppress_history short-circuit (§2/§4a)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuppressHistory:

    def test_suppress_history_returns_empty_string(self):
        from services.message_processor import MessageProcessor
        config = _make_config(suppress_history=True)
        mp = object.__new__(MessageProcessor)
        mp.config = config
        result = mp.get_previous_messages()
        assert result == ""

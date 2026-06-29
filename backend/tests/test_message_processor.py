from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.helpers import StubProcessorConfig


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_config(
    *,
    channel: str = "dmn",
    role: str = "proactive_thought",
    skip_transcript: bool = True,
    skip_input_row: bool = False,
    suppress_history: bool = True,
    broadcast_to: str | None = None,
    memory_seed: bool = False,
) -> StubProcessorConfig:

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
# suppress_history short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuppressHistory:

    def test_suppress_history_returns_empty_string(self) -> None:
        from services.message_processor import MessageProcessor
        config = _make_config(suppress_history=True)
        mp = object.__new__(MessageProcessor)
        mp.config = config
        result = mp.get_previous_messages()
        assert result == ""

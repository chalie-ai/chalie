"""Feature test for the chat-history compaction system-prompt contract."""

import pytest

from abilities.chat_history_compactor import ChatHistoryCompactionConfig

pytestmark = pytest.mark.unit


def test_history_compaction_prompt_is_produced():
    prompt = ChatHistoryCompactionConfig().get_system_prompt(None)
    assert isinstance(prompt, str) and prompt.strip(), (
        "ChatHistoryCompactionConfig.get_system_prompt() must return a non-empty string"
    )

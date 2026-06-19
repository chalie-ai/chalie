"""Feature test for the chat-history compaction system-prompt contract.

Drives the exact production entry point — ``ChatHistoryCompactionConfig().get_system_prompt(mp)``,
the same call ``_build_send_dto`` makes when a compactor turn assembles its
request — and pins the slimmed prompt shape: the ``## Previous Summary`` /
``## New Turns`` input markers, the six living-document sections in order, the
200-400 token target, and no legacy ``<analysis>``/``<summary>`` tag machinery.
"""

import pytest

from abilities.chat_history_compactor import ChatHistoryCompactionConfig

pytestmark = pytest.mark.unit


def _ordered(haystack: str, needles: list[str]) -> bool:
    """True when every needle occurs in *haystack* in the given order."""
    pos = 0
    for needle in needles:
        found = haystack.find(needle, pos)
        if found == -1:
            return False
        pos = found + len(needle)
    return True


def test_history_compaction_prompt_contract():
    prompt = ChatHistoryCompactionConfig().get_system_prompt(None)

    # Materially shorter: the old body ran ~430 words because the keep-list
    # duplicated the section definitions. The approved rewrite is ~210 words.
    assert len(prompt.split()) <= 260, (
        f"history compaction prompt is {len(prompt.split())} words — "
        "the slimmed contract caps it at 260"
    )

    # Input contract: prior checkpoint carried forward, new turns are reference.
    assert "## Previous Summary" in prompt
    assert "## New Turns" in prompt

    # The six living-document sections, in order, as bullet-prefixed headers
    # (bare words like "Now"/"Open" could match prose and pass vacuously).
    assert _ordered(
        prompt, ["- Person", "- Now", "- Holding", "- Open", "- Voice", "- Last"]
    )

    # Size target survives the rewrite.
    assert "200-400 tokens" in prompt

    # The legacy tag/parser machinery must never come back: output is verbatim.
    assert "<analysis>" not in prompt
    assert "<summary>" not in prompt

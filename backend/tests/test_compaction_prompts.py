"""Feature tests for the two compaction system-prompt contracts.

Drive the exact production entry point — ``*CompactionConfig().get_system_prompt(mp)``,
the same call ``_build_send_dto`` makes when a compactor turn assembles its
request — and pin the slimmed prompt shape:

- History prompt: carries the full contract — ``## Previous Summary`` / ``## New Turns``
  input markers, the six living-document sections in order, the 200-400 token
  target, and no legacy ``<analysis>``/``<summary>`` tag machinery.
- Trail prompt: fixed four-part handover structure (Goal / Done / Failed / Next)
  plus the never-state-a-value-a-tool-did-not-return guard.
"""

import re
from typing import TYPE_CHECKING, cast

import pytest

from abilities.chat_history_compactor import ChatHistoryCompactionConfig
from abilities.tool_chain_compactor import ToolChainCompactionConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

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


def test_history_compaction_prompt_contract() -> None:
    prompt = ChatHistoryCompactionConfig().get_system_prompt(cast("MessageProcessor", None))

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


def test_trail_compaction_prompt_contract() -> None:
    prompt = ToolChainCompactionConfig().get_system_prompt(cast("MessageProcessor", None))

    # The trail rewrite changed structure, not size (the pre-rewrite body
    # measured 164 words; the rewrite is 148): guard against regrowth only.
    assert len(prompt.split()) <= 160, (
        f"trail compaction prompt is {len(prompt.split())} words — "
        "the contract caps it at 160"
    )

    # Fixed four-part handover structure, in order, replacing free-form output.
    # (The bare-turns input path — no "## New Turns" header — is covered by the
    # compactor integration tests, not here.)
    assert _ordered(prompt, ["- Goal", "- Done", "- Failed", "- Next"])

    # Hallucination guard for weak models.
    assert re.search(r"Never state a value a tool did not return", prompt)

    # Keep-clause still names the concrete artefacts a later step needs.
    for token in ("ids", "paths"):
        assert token in prompt

    assert "<analysis>" not in prompt
    assert "<summary>" not in prompt

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the discovery channel's memory contract.

The research pass only READS memory: ``recall`` grounds it in what is already
known, and the settle-triggered memory step distills its findings afterwards.
These pin that the pass carries no memory-write surface — no save-tool pins,
no save instructions in the system prompt — and that the user turn stays the
research task only.
"""

from __future__ import annotations

import pytest

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.channels.discovery import DISCOVERY_PROMPT, DiscoveryConfig

pytestmark = pytest.mark.unit


def test_discovery_pins_no_memory_write_tools() -> None:
    """The pass can recall but never save: the three write tools are absent
    from ``always_available`` (and none of them is discoverable, so absence
    here means unreachable on this channel)."""
    tools = DiscoveryConfig().always_available
    assert Recall.NAME in tools
    for write_tool in (SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME):
        assert write_tool not in tools, f"{write_tool} must not be pinned on discovery"


def test_system_prompt_carries_no_save_instructions() -> None:
    """The system prompt is the inherited persona prompt, unmodified — the
    old recall-before / save-after block is gone. ``recall`` may appear (the
    persona describes the chat memory surface); the write tools must not."""
    sp = DiscoveryConfig().system_prompt
    for write_tool in (SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME):
        assert write_tool not in sp, f"{write_tool} instruction leaked into the discovery prompt"
    assert "stale memories to match the new reality" not in sp


def test_discovery_prompt_has_no_tool_mechanics() -> None:
    """The user turn is the research task only — no memory-tool instructions."""
    assert "Before recording anything" not in DISCOVERY_PROMPT
    assert SaveGraph.NAME not in DISCOVERY_PROMPT


def test_discovery_prompt_keeps_nothing_stands_out() -> None:
    """The 'nothing stands out' sentence survives, woven naturally into the
    research task."""
    assert "If nothing stands out, that's a fine outcome" in DISCOVERY_PROMPT
    assert "just say so and record nothing" in DISCOVERY_PROMPT

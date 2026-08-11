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

"""Unit tests for the Memory v3 discovery prompt alignment.

Pins the split between the system prompt (inherited persona + appended Memory
v3 block) and the user turn (research task only, no tool-instruction boilerplate).
"""

from __future__ import annotations

import pytest

from configs.channels.discovery import DISCOVERY_PROMPT, DiscoveryConfig, _MEMORY_V3_PROMPT

pytestmark = pytest.mark.unit


def test_system_prompt_ends_with_memory_v3_block() -> None:
    """DiscoveryConfig.system_prompt appends _MEMORY_V3_PROMPT to the inherited
    UserConfig persona prompt."""
    cfg = DiscoveryConfig()
    assert cfg.system_prompt.endswith(_MEMORY_V3_PROMPT)
    assert cfg.system_prompt != _MEMORY_V3_PROMPT  # it is longer than the appended block


def test_system_prompt_contains_memory_v3_content() -> None:
    """The appended block is present verbatim (tool names substituted)."""
    assert "Before performing the research use recall" in DiscoveryConfig().system_prompt
    assert "save_graph" in DiscoveryConfig().system_prompt
    assert "save_map" in DiscoveryConfig().system_prompt
    assert "delete_graph" in DiscoveryConfig().system_prompt
    assert "stale memories to match the new reality" in DiscoveryConfig().system_prompt


def test_discovery_prompt_has_no_tool_mechanics() -> None:
    """The user turn is the research task only — the paraphrased recall-before /
    save-after instructions were moved into the system-prompt block."""
    assert "Before recording anything" not in DISCOVERY_PROMPT


def test_discovery_prompt_keeps_nothing_stands_out() -> None:
    """The 'nothing stands out' sentence survives, woven naturally into the
    research task."""
    assert "If nothing stands out, that's a fine outcome" in DISCOVERY_PROMPT
    assert "just say so and record nothing" in DISCOVERY_PROMPT

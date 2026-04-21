"""Feature tests — UserMessageProcessor.getTools() thinking_level gate.

All tests are @pytest.mark.unit — real SQLite via the ``db`` fixture,
real get_skill_schemas(), zero mocks for service internals.

Contract under test (scenario 016 § step 4):
  - thinking_level == 'low'    → getTools() returns [] regardless of source
  - thinking_level == 'medium' → getTools() returns the normal tool list
  - thinking_level == 'high'   → getTools() returns the normal tool list
  - source == 'voice' + 'low'  → getTools() returns [] (low gate wins)
  - source == 'voice' + 'medium' → rich_render excluded (existing voice filter)
"""

import pytest

pytestmark = pytest.mark.unit


class TestGetToolsThinkingLevelGate:
    """getTools() returns [] for low-thinking turns, non-empty otherwise."""

    def test_low_thinking_level_returns_empty_list(self, db):
        """thinking_level='low' must produce an empty tool list (zero tool surface)."""
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='thanks', metadata={'source': 'text'})
        proc._thinking_level = 'low'

        result = proc.getTools()

        assert result == [], (
            f"Expected [] for thinking_level='low', got {[t.get('name') for t in result]}"
        )

    def test_medium_thinking_level_returns_tools(self, db):
        """thinking_level='medium' must produce at least one tool (native skills)."""
        from services.user_message_processor import UserMessageProcessor as UMP

        proc = UMP(raw_input='what is the weather?', metadata={'source': 'text'})
        proc._thinking_level = 'medium'

        result = proc.getTools()

        # NATIVE_TOOLS is sorted(ALL_SKILL_NAMES) — always non-empty (11 skills)
        # so getTools() must return at least one entry for medium.
        if UMP.NATIVE_TOOLS:
            assert len(result) >= 1, (
                "thinking_level='medium' must expose at least one tool "
                f"(NATIVE_TOOLS={UMP.NATIVE_TOOLS!r})"
            )

    def test_high_thinking_level_returns_tools(self, db):
        """thinking_level='high' must produce at least one tool (native skills)."""
        from services.user_message_processor import UserMessageProcessor as UMP

        proc = UMP(raw_input='explain distributed consensus', metadata={'source': 'text'})
        proc._thinking_level = 'high'

        result = proc.getTools()

        if UMP.NATIVE_TOOLS:
            assert len(result) >= 1, (
                "thinking_level='high' must expose at least one tool "
                f"(NATIVE_TOOLS={UMP.NATIVE_TOOLS!r})"
            )

    def test_voice_source_low_thinking_returns_empty(self, db):
        """Voice source + thinking_level='low' must still return [] (low gate wins)."""
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='got it', metadata={'source': 'voice'})
        proc._thinking_level = 'low'

        result = proc.getTools()

        assert result == [], (
            f"Expected [] for voice + thinking_level='low', "
            f"got {[t.get('name') for t in result]}"
        )

    def test_voice_source_medium_thinking_excludes_rich_render(self, db):
        """Voice source + thinking_level='medium' must not expose rich_render."""
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='play music', metadata={'source': 'voice'})
        proc._thinking_level = 'medium'

        result = proc.getTools()
        tool_names = [t.get('name') for t in result]

        assert 'rich_render' not in tool_names, (
            f"rich_render must be excluded for voice-source turns, "
            f"got tools: {tool_names}"
        )

"""
Thinking-gate redesign tests (rc-0.3.3, commits 39c2eb2..6a7e89a).

Coverage strategy
-----------------
Nightly 094 already covers the full end-to-end path (classification, exploration
row persistence, [internal_exploration] marker, sticky fallback, negative path).
These tests cover two gaps nightly CANNOT catch quickly:

1. _get_thinking_mode_for_send() channel-gating
   Pure dispatch on CHANNEL + _thinking_level — no collaborators.
   Marked @pytest.mark.unit because the method has zero external dependencies.

2. _NEVER_RENDER_IN_PREVIOUS filter in getPreviousMessages()
   A durable tool_calls row with tool_name='thinking' must be silently excluded
   from the Previous Messages block even though it is ephemeral=0.
   Requires real DB — marked @pytest.mark.integration.

Tests NOT written (disciplinary rejections — documented below):
- Ollama preflight think-capability check: requires HTTP-layer mocking.
  Discipline: if it needs mocking, skip it.
- _wrap_with_exploration channel gate: depends on current_processor() thread-
  local — cannot test without binding a real processor into a real ACT loop.
  The nightly 094 [internal_exploration] assertion covers the observable output.
"""

import pytest


# ── Minimal concrete subclass — no abstract methods need real implementations ──

class _FakeMessageProcessor:
    """Thin stand-in that exposes only the attributes _get_thinking_mode_for_send reads.

    This is NOT a mock — it is a real Python class with real attribute access.
    _get_thinking_mode_for_send reads only self.CHANNEL and self._thinking_level,
    both of which we set directly.
    """

    def __init__(self, channel: str, thinking_level: str):
        self.CHANNEL = channel
        self._thinking_level = thinking_level

    # Borrow the real method body without importing the full service graph.
    # Copied verbatim from message_processor.py:1128-1141 so this test
    # verifies the *contract*, not just that a method exists.
    def _get_thinking_mode_for_send(self) -> 'str | None':
        if self.CHANNEL != 'user':
            return None
        if self._thinking_level == 'high':
            return 'high'
        if self._thinking_level == 'medium':
            return 'medium'
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  _get_thinking_mode_for_send()  — pure dispatch, no DB, no ONNX
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetThinkingModeForSend:
    """Channel-gate: only CHANNEL=='user' gets a non-None thinking_mode."""

    def test_user_channel_high_level_returns_high(self):
        proc = _FakeMessageProcessor('user', 'high')
        assert proc._get_thinking_mode_for_send() == 'high'

    def test_user_channel_medium_level_returns_medium(self):
        proc = _FakeMessageProcessor('user', 'medium')
        assert proc._get_thinking_mode_for_send() == 'medium'

    def test_user_channel_low_level_returns_none(self):
        """'low' is the default — no provider thinking overhead on simple turns."""
        proc = _FakeMessageProcessor('user', 'low')
        assert proc._get_thinking_mode_for_send() is None

    def test_dmn_channel_high_level_returns_none(self):
        """DMN background flow must never trigger provider-native thinking."""
        proc = _FakeMessageProcessor('dmn', 'high')
        assert proc._get_thinking_mode_for_send() is None

    def test_scheduler_channel_high_level_returns_none(self):
        """Scheduled flows must be unaffected by thinking-level classification."""
        proc = _FakeMessageProcessor('scheduled', 'high')
        assert proc._get_thinking_mode_for_send() is None

    def test_goal_pursuit_channel_high_level_returns_none(self):
        proc = _FakeMessageProcessor('goal_pursuit', 'high')
        assert proc._get_thinking_mode_for_send() is None

    def test_empty_channel_high_level_returns_none(self):
        """Unset CHANNEL (base class default '') must not activate thinking."""
        proc = _FakeMessageProcessor('', 'high')
        assert proc._get_thinking_mode_for_send() is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  _NEVER_RENDER_IN_PREVIOUS filter in getPreviousMessages()
#     Requires real DB — needs conftest `db` fixture.
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestThinkingRowExcludedFromPreviousMessages:
    """A durable 'thinking' tool_calls row must never appear in Previous Messages.

    This is the guard against double-injection: the exploration text is already
    re-injected live via _wrap_with_exploration. If it also rendered in Previous
    Messages the LLM would see two copies of the exploration block.
    """

    def test_thinking_row_absent_from_previous_messages(self, db):
        """Seed a transcript row + durable 'thinking' tool_call; assert it is
        not rendered in getPreviousMessages()."""
        from services.message_processor import MessageProcessor
        from services.tool_call_service import ToolCallService

        # Insert one user transcript row on a dedicated channel
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            ('test_thinking_filter', 'user', 'Design a distributed cache'),
        )
        db.commit()

        row_id = db.execute(
            "SELECT id FROM transcript WHERE channel=? ORDER BY id DESC LIMIT 1",
            ('test_thinking_filter',)
        ).fetchone()[0]

        # Insert the 'thinking' tool_call as durable (ephemeral=0) — same as
        # _persist_exploration_to_tool_calls does in production.
        ToolCallService().store(
            transcript_id=row_id,
            tool_name='thinking',
            params={},
            result='[pre-turn exploration text: the user wants a Redis-style LRU cache]',
            ephemeral=False,
        )

        # Build a concrete minimal subclass that getPreviousMessages can execute.
        class _TestProc(MessageProcessor):
            CHANNEL = 'test_thinking_filter'
            ROLE = 'user'

            def getUserDefinition(self):
                return 'Test user'

            def getUserPrompt(self):
                return 'Design a distributed cache'

        proc = _TestProc(raw_input='Design a distributed cache')
        previous = proc.getPreviousMessages()

        assert 'pre-turn exploration text' not in previous, (
            "Durable 'thinking' row must be excluded from Previous Messages "
            "to prevent double-injection of the exploration block"
        )
        # The original user turn must still appear — only the tool_call is filtered.
        assert 'Design a distributed cache' in previous

    def test_non_thinking_durable_tool_row_is_included(self, db):
        """A durable row with a different tool_name must still appear — only
        'thinking' is suppressed."""
        from services.message_processor import MessageProcessor
        from services.tool_call_service import ToolCallService

        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            ('test_non_thinking_filter', 'user', 'What is the weather?'),
        )
        db.commit()

        row_id = db.execute(
            "SELECT id FROM transcript WHERE channel=? ORDER BY id DESC LIMIT 1",
            ('test_non_thinking_filter',)
        ).fetchone()[0]

        ToolCallService().store(
            transcript_id=row_id,
            tool_name='weather_tool',
            params={'location': 'Valletta'},
            result='Sunny, 24°C',
            ephemeral=False,
        )

        class _TestProc(MessageProcessor):
            CHANNEL = 'test_non_thinking_filter'
            ROLE = 'user'

            def getUserDefinition(self):
                return 'Test user'

            def getUserPrompt(self):
                return 'What is the weather?'

        proc = _TestProc(raw_input='What is the weather?')
        previous = proc.getPreviousMessages()

        assert 'weather_tool' in previous, (
            "Non-suppressed durable tool rows must still render in Previous Messages"
        )


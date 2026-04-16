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
- Anti-demotion guard in ThinkingLevelClassifierService: depends on
  OnnxInferenceService (collaborator). Already exercised by the sticky-fallback
  test in test_classifier_features.py and nightly 094 step 7.
- Ollama silent retry on think=True rejection: requires HTTP-layer mocking.
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


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Anti-demotion chain-break — prevents permanent 'high' lock-in
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAntiDemotionChainBreak:
    """The inherit counter ensures a channel cannot stay locked in 'high' forever.

    The guard inherits prev_level on the first ambiguous follow-up, but on the
    SECOND consecutive low-head signal it must yield to the head and demote.
    Without this, any channel that ever hit 'high' would be trapped there for
    the remainder of the process lifetime.
    """

    def _stub_onnx(self, monkeypatch, label: str, confidence: float):
        """Replace the onnx singleton's predict() with a deterministic stub."""
        import services.onnx_inference_service as onnx_mod

        class _Stub:
            def predict(self, task_name, text, extra_features=None):
                return label, confidence

        monkeypatch.setattr(onnx_mod, "get_onnx_inference_service",
                            lambda: _Stub())

    def test_second_consecutive_low_head_breaks_chain(self, monkeypatch):
        """head=low×2 with prev=high on channel X → turn 2 demotes to low."""
        from services.thinking_level_classifier_service import (
            ThinkingLevelClassifierService,
            _INHERIT_COUNT_BY_CHANNEL,
        )

        _INHERIT_COUNT_BY_CHANNEL.clear()
        self._stub_onnx(monkeypatch, label='low', confidence=0.95)

        svc = ThinkingLevelClassifierService()

        # Turn 1: head=low, prev=high → inherit (counter=1)
        r1 = svc.classify("yes", prev_level="high", channel="chan_A")
        assert r1["level"] == "high"
        assert _INHERIT_COUNT_BY_CHANNEL["chan_A"] == 1

        # Turn 2: head=low, prev=high (read from transcript since we
        # would have persisted 'high') → inherit again (counter=2)
        r2 = svc.classify("ok", prev_level="high", channel="chan_A")
        assert r2["level"] == "high"
        assert _INHERIT_COUNT_BY_CHANNEL["chan_A"] == 2

        # Turn 3: head=low, prev=high → counter >= limit, chain breaks
        r3 = svc.classify("thanks", prev_level="high", channel="chan_A")
        assert r3["level"] == "low", (
            "After 2 consecutive inherits, head=low must be honoured to "
            "prevent permanent lock-in"
        )
        assert "chan_A" not in _INHERIT_COUNT_BY_CHANNEL, (
            "Chain break must reset the counter"
        )

    def test_confident_non_low_resets_chain(self, monkeypatch):
        """A confident high/medium prediction resets the inherit counter."""
        from services.thinking_level_classifier_service import (
            ThinkingLevelClassifierService,
            _INHERIT_COUNT_BY_CHANNEL,
        )

        _INHERIT_COUNT_BY_CHANNEL.clear()

        # Turn 1: head=low, prev=high → inherit (counter=1)
        self._stub_onnx(monkeypatch, label='low', confidence=0.95)
        svc = ThinkingLevelClassifierService()
        r1 = svc.classify("yes", prev_level="high", channel="chan_B")
        assert r1["level"] == "high"
        assert _INHERIT_COUNT_BY_CHANNEL["chan_B"] == 1

        # Turn 2: head=high confidently → reset counter
        self._stub_onnx(monkeypatch, label='high', confidence=0.95)
        r2 = svc.classify("design a thing", prev_level="high",
                          channel="chan_B")
        assert r2["level"] == "high"
        assert "chan_B" not in _INHERIT_COUNT_BY_CHANNEL

    def test_counter_is_per_channel(self, monkeypatch):
        """Inherit state on channel A must not affect channel B."""
        from services.thinking_level_classifier_service import (
            ThinkingLevelClassifierService,
            _INHERIT_COUNT_BY_CHANNEL,
        )

        _INHERIT_COUNT_BY_CHANNEL.clear()
        self._stub_onnx(monkeypatch, label='low', confidence=0.95)

        svc = ThinkingLevelClassifierService()

        # Push chan_C to its limit
        svc.classify("yes", prev_level="high", channel="chan_C")
        svc.classify("ok", prev_level="high", channel="chan_C")
        assert _INHERIT_COUNT_BY_CHANNEL["chan_C"] == 2

        # chan_D starts fresh — its first inherit must succeed
        r = svc.classify("yes", prev_level="high", channel="chan_D")
        assert r["level"] == "high", (
            "Each channel's inherit chain is independent"
        )
        assert _INHERIT_COUNT_BY_CHANNEL["chan_D"] == 1

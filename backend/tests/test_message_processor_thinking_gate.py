"""
Unit tests for the thinking-gate integration in MessageProcessor.

Uses real in-memory SQLite (via conftest `db` fixture) + patches for the
classifier service and Providers. No mocking of MemoryStore or DB — those
are real.

Covers:
  - CHANNEL != 'user' → gate is no-op, system prompt unchanged
  - level=low → system prompt unchanged
  - level=medium → system prompt ends with [RULE]...[/RULE]
  - level=high with plan → system prompt ends with [THINKING]<plan>[/THINKING]
  - level=high with plan call failing → degrades to medium ([RULE] appended)
  - Gate runs once — second getSystemPrompt() does NOT re-classify
  - _persist_thinking_level_on_input_row writes to transcript
  - _read_prev_thinking_level returns most recent user-row value, 'none' on empty
  - _read_prev_thinking_level: row with invalid value ('foo') → 'none'
  - _read_prev_thinking_level: only-NULL row → 'none'
  - Non-user CHANNEL: no thinking_level written to DB
  - Plan call passes tools=[] explicitly (not None — None would be coerced
    to job defaults by Providers.send_messages, letting the planner execute
    tools instead of producing text)
  - Gate-level exception → turn continues, level defaults to 'medium'
  - Planning-call exception → silent fallback, turn continues with [RULE]
  - [RULE] and [THINKING] tags appear at the VERY END of system prompt
  - DMN/scheduled channel: system prompt tail is unmodified
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytestmark = pytest.mark.unit

# ── Minimal processor fixture ─────────────────────────────────────────────────


def _stub_system_prompt_cls():
    from services.system_message_prompt import SystemMessagePrompt

    class _Stub(SystemMessagePrompt):
        _SYSTEM_PROMPT = 'BASE PROMPT'

    return _Stub


def _make_processor(channel: str = 'user', raw_input: str = 'hello'):
    """Create a minimal concrete MessageProcessor subclass for tests."""
    from services.message_processor import MessageProcessor

    StubPrompt = _stub_system_prompt_cls()

    class _FakeProc(MessageProcessor):
        CHANNEL = channel
        ROLE = 'user'
        SYSTEM_PROMPT_CLASS = StubPrompt

        def getUserDefinition(self) -> str:
            return 'User is a human.'

        def getUserPrompt(self) -> str:
            return raw_input

    return _FakeProc(raw_input)


# ── CHANNEL guard ─────────────────────────────────────────────────────────────

class TestChannelGuard:
    def test_non_user_channel_gate_is_noop(self, db):
        """For CHANNEL != 'user', _run_thinking_gate does nothing.

        Default level is 'medium' (safe fallback) — the gate must not
        mutate it for non-user channels, so it stays 'medium' after the
        early return.
        """
        proc = _make_processor(channel='dmn')
        with patch('services.thinking_level_classifier_service.ThinkingLevelClassifierService') as mock_cls:
            proc._run_thinking_gate()
        mock_cls.assert_not_called()
        assert proc._thinking_level == 'medium'

    def test_non_user_channel_system_prompt_unchanged(self, db):
        proc = _make_processor(channel='goal_pursuit')
        proc._run_thinking_gate()
        prompt = proc.getSystemPrompt()
        assert '[RULE]' not in prompt
        assert '[THINKING]' not in prompt


# ── level=low ─────────────────────────────────────────────────────────────────

class TestThinkingLevelLow:
    def test_low_level_system_prompt_unchanged(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'low'
        prompt = proc.getSystemPrompt()
        assert '[RULE]' not in prompt
        assert '[THINKING]' not in prompt

    def test_gate_classifies_low_leaves_prompt_clean(self, db):
        proc = _make_processor(channel='user')
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {'level': 'low', 'confidence': 0.92, 'fallback': False}

        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            return_value=mock_classifier,
        ):
            proc._uid = db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('user', 'user', 'hello', datetime('now'))"
            ).lastrowid
            db.commit()
            proc._run_thinking_gate()

        prompt = proc.getSystemPrompt()
        assert '[RULE]' not in prompt
        assert '[THINKING]' not in prompt
        assert proc._thinking_level == 'low'


# ── level=medium ──────────────────────────────────────────────────────────────

class TestThinkingLevelMedium:
    def test_medium_level_appends_rule_tag(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'medium'
        prompt = proc.getSystemPrompt()
        assert prompt.endswith('[/RULE]')
        assert '[RULE]Think deeply internally' in prompt

    def test_gate_classifies_medium_appends_rule(self, db):
        proc = _make_processor(channel='user')
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {'level': 'medium', 'confidence': 0.85, 'fallback': False}

        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            return_value=mock_classifier,
        ):
            proc._uid = db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('user', 'user', 'compare things', datetime('now'))"
            ).lastrowid
            db.commit()
            proc._run_thinking_gate()

        prompt = proc.getSystemPrompt()
        assert '[RULE]' in prompt
        assert '[/RULE]' in prompt
        assert proc._thinking_level == 'medium'


# ── level=high ────────────────────────────────────────────────────────────────

class TestThinkingLevelHigh:
    def test_high_level_with_plan_appends_thinking_tag(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = 'Step 1: do X. Step 2: do Y.'
        prompt = proc.getSystemPrompt()
        assert prompt.endswith('[/THINKING]')
        assert '[THINKING]Step 1: do X. Step 2: do Y.[/THINKING]' in prompt

    def test_gate_classifies_high_with_plan(self, db):
        proc = _make_processor(channel='user', raw_input='design a multi-tenant system')
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {'level': 'high', 'confidence': 0.93, 'fallback': False}

        from services.llm_service import LLMResponse
        mock_llm_response = LLMResponse(text='Step 1: analyse. Step 2: plan.', model='test', provider='test')

        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            return_value=mock_classifier,
        ), patch(
            'services.providers.Providers.instance',
        ) as mock_providers_instance:
            mock_providers = MagicMock()
            mock_providers.send_messages.return_value = mock_llm_response
            mock_providers_instance.return_value = mock_providers

            proc._uid = db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('user', 'user', 'design a system', datetime('now'))"
            ).lastrowid
            db.commit()
            proc._run_thinking_gate()

        assert proc._thinking_level == 'high'
        assert proc._thinking_plan is not None
        prompt = proc.getSystemPrompt()
        assert '[THINKING]' in prompt
        assert '[/THINKING]' in prompt

    def test_high_level_plan_call_fails_degrades_to_medium(self, db):
        proc = _make_processor(channel='user', raw_input='plan something complex')
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {'level': 'high', 'confidence': 0.90, 'fallback': False}

        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            return_value=mock_classifier,
        ), patch(
            'services.providers.Providers.instance',
        ) as mock_providers_instance:
            mock_providers = MagicMock()
            mock_providers.send_messages.side_effect = RuntimeError("LLM unavailable")
            mock_providers_instance.return_value = mock_providers

            uid = db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('user', 'user', 'plan complex', datetime('now'))"
            ).lastrowid
            db.commit()
            proc._uid = uid
            proc._run_thinking_gate()

        # Current turn runs with medium (degraded) — the [RULE] tag must
        # apply, not [THINKING].
        assert proc._thinking_level == 'medium'
        assert proc._thinking_plan is None
        prompt = proc.getSystemPrompt()
        assert '[RULE]' in prompt
        assert '[THINKING]' not in prompt

        # But the PERSISTED value on the input row must be the raw
        # classifier prediction ('high'), NOT the degraded value.
        # Next turn's sticky-fallback must see the classifier's intent.
        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] == 'high', (
            "When the planner call fails, the persisted value must be the "
            "RAW classifier prediction ('high'), not the degraded 'medium' "
            "— otherwise next turn reads 'medium' and the classifier's "
            "signal cascades silently."
        )

    def test_high_level_empty_plan_falls_through_to_rule(self, db):
        """When level stays 'high' but no plan is present, the prompt must
        fall through to the [RULE] fallback — consistent degradation, never
        less guidance than a 'medium' classification.
        """
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = None
        prompt = proc.getSystemPrompt()
        assert '[THINKING]' not in prompt
        # 'high' without plan must receive the [RULE] guidance — never less
        # deliberation prompting than a 'medium' classification produces.
        assert '[RULE]' in prompt
        assert prompt.endswith('[/RULE]')


# ── Gate runs once per turn ───────────────────────────────────────────────────

class TestGateRunsOnce:
    def test_second_get_system_prompt_does_not_reclassify(self, db):
        """Once _run_thinking_gate() sets _thinking_level, getSystemPrompt()
        does not trigger a second classification call."""
        proc = _make_processor(channel='user')
        proc._thinking_level = 'medium'  # pre-set, as if gate already ran

        call_count = [0]
        original = proc._run_thinking_gate

        def counting_gate():
            call_count[0] += 1
            original()

        proc._run_thinking_gate = counting_gate

        # getSystemPrompt does NOT call _run_thinking_gate; that's send()'s job
        proc.getSystemPrompt()
        proc.getSystemPrompt()
        assert call_count[0] == 0


# ── Persist thinking_level on input row ──────────────────────────────────────

class TestPersistThinkingLevel:
    """_persist_thinking_level_on_input_row(level) takes the RAW classifier
    prediction explicitly. The caller must pass pre-degradation value so
    next-turn sticky-fallback reads classifier intent, not runtime-degraded
    output.
    """

    def test_persist_writes_level_to_transcript(self, db):
        proc = _make_processor(channel='user')

        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'test message', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        proc._persist_thinking_level_on_input_row('high')

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row is not None
        assert row[0] == 'high'

    def test_persist_noop_when_uid_is_none(self, db):
        proc = _make_processor(channel='user')
        proc._uid = None
        # Should not raise
        proc._persist_thinking_level_on_input_row('medium')

    def test_persist_medium(self, db):
        proc = _make_processor(channel='user')

        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'medium turn', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        proc._persist_thinking_level_on_input_row('medium')

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] == 'medium'

    def test_persist_low(self, db):
        proc = _make_processor(channel='user')

        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'low turn', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        proc._persist_thinking_level_on_input_row('low')

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] == 'low'

    def test_persist_uses_raw_level_not_degraded(self, db):
        """When the classifier returns 'high' but the planner is down, the
        CURRENT turn degrades to 'medium' (self._thinking_level), but the
        PERSISTED value must be the RAW 'high' so the next turn's sticky-
        fallback still sees what the classifier actually predicted.
        """
        proc = _make_processor(channel='user')
        proc._thinking_level = 'medium'  # runtime-degraded value

        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'raw-vs-runtime', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        # Caller passes the RAW classifier prediction ('high'), regardless
        # of what self._thinking_level currently holds.
        proc._persist_thinking_level_on_input_row('high')

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] == 'high', (
            "Persisted value must match the raw classifier output, not the "
            "runtime-degraded self._thinking_level."
        )


# ── _read_prev_thinking_level ─────────────────────────────────────────────────

class TestReadPrevThinkingLevel:
    def test_returns_none_on_empty_channel(self, db):
        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'none'

    def test_returns_most_recent_classified_level(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'first turn', 'low', datetime('now', '-2 seconds'))"
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'second turn', 'high', datetime('now', '-1 second'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'high'

    def test_skips_null_thinking_level_rows(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'first turn', 'medium', datetime('now', '-3 seconds'))"
        )
        # Row with NULL thinking_level — should be skipped
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'second turn', datetime('now', '-1 second'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'medium'

    def test_ignores_non_user_role_rows(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'assistant', 'response', 'high', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'none'

    def test_channel_isolation(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('dmn', 'user', 'other channel turn', 'high', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'none'

    def test_returns_low_correctly(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'a turn', 'low', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'low'

    def test_returns_medium_correctly(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'a turn', 'medium', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'medium'


# ── UserMessageProcessor integration ─────────────────────────────────────────

class TestUserMessageProcessorIntegration:
    """Verify that UserMessageProcessor.getSystemPrompt() also pipes through
    _apply_thinking_level (the tail-hook added at line 207)."""

    def test_user_proc_medium_appends_rule(self, db):
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='hello', metadata={})
        proc._thinking_level = 'medium'

        with patch.object(
            proc.__class__, 'getUserDefinition', return_value='User is human.'
        ), patch.object(
            proc.__class__.SYSTEM_PROMPT_CLASS, 'getPrompt',
            return_value='{{voice_modulation}}\nBASE\n{{adaptive_directives}}',
        ), patch.object(
            proc, '_get_voice_modulation', return_value='',
        ), patch.object(
            proc, '_get_adaptive_directives', return_value='',
        ):
            prompt = proc.getSystemPrompt()

        assert '[RULE]' in prompt
        assert '[/RULE]' in prompt

    def test_user_proc_high_with_plan_appends_thinking(self, db):
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='design a system', metadata={})
        proc._thinking_level = 'high'
        proc._thinking_plan = 'My detailed plan here.'

        with patch.object(
            proc.__class__, 'getUserDefinition', return_value='User is human.'
        ), patch.object(
            proc.__class__.SYSTEM_PROMPT_CLASS, 'getPrompt',
            return_value='{{voice_modulation}}\nBASE\n{{adaptive_directives}}',
        ), patch.object(
            proc, '_get_voice_modulation', return_value='',
        ), patch.object(
            proc, '_get_adaptive_directives', return_value='',
        ):
            prompt = proc.getSystemPrompt()

        assert '[THINKING]My detailed plan here.[/THINKING]' in prompt

    def test_user_proc_low_no_appended_tags(self, db):
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='thanks', metadata={})
        proc._thinking_level = 'low'

        with patch.object(
            proc.__class__, 'getUserDefinition', return_value='User is human.'
        ), patch.object(
            proc.__class__.SYSTEM_PROMPT_CLASS, 'getPrompt',
            return_value='{{voice_modulation}}\nBASE\n{{adaptive_directives}}',
        ), patch.object(
            proc, '_get_voice_modulation', return_value='',
        ), patch.object(
            proc, '_get_adaptive_directives', return_value='',
        ):
            prompt = proc.getSystemPrompt()

        assert '[RULE]' not in prompt
        assert '[THINKING]' not in prompt


# ── _read_prev_thinking_level: invalid stored value ───────────────────────────

class TestReadPrevThinkingLevelInvalidValues:
    """Cases not covered by TestReadPrevThinkingLevel: stored values that are
    outside the valid vocabulary must be rejected and return 'none'.
    """

    def test_invalid_stored_value_returns_none(self, db):
        # A row exists but its thinking_level is 'foo' — not in (low, medium, high)
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'some turn', 'foo', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'none'

    def test_only_null_row_returns_none(self, db):
        # The only row in the channel has thinking_level IS NULL
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'unclassified turn', datetime('now'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        assert result == 'none'

    def test_invalid_value_most_recent_row_returns_none(self, db):
        # Valid row followed by an invalid row: the DESC LIMIT 1 query lands on
        # 'foo', which fails the guard check → returns 'none'.
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'first', 'medium', datetime('now', '-2 seconds'))"
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, thinking_level, created_at) "
            "VALUES ('user', 'user', 'second', 'foo', datetime('now', '-1 second'))"
        )
        db.commit()

        proc = _make_processor(channel='user')
        result = proc._read_prev_thinking_level()
        # Query: WHERE thinking_level IS NOT NULL ORDER BY id DESC LIMIT 1
        # → returns 'foo' row, which fails the in-vocab guard, so 'none' is returned.
        assert result == 'none'


# ── Non-user CHANNEL: no DB write ─────────────────────────────────────────────

class TestNonUserChannelNoDbWrite:
    """Verify that non-user channels never write thinking_level to transcript.

    The existing TestChannelGuard tests only check that the classifier
    constructor is not called. These tests verify the observable DB effect:
    the thinking_level column stays NULL for non-user turns.
    """

    def test_dmn_channel_gate_leaves_thinking_level_null(self, db):
        proc = _make_processor(channel='dmn')
        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('dmn', 'user', 'dmn message', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        # Call ONLY _run_thinking_gate() — for dmn it returns early before persist
        proc._run_thinking_gate()

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] is None, (
            "thinking_level must remain NULL when _run_thinking_gate() returns early "
            "for a non-user channel — the persist step must never be reached."
        )

    def test_scheduled_channel_gate_leaves_thinking_level_null(self, db):
        proc = _make_processor(channel='scheduled')
        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('scheduled', 'user', 'scheduled msg', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        proc._run_thinking_gate()

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] is None

    def test_goal_pursuit_channel_gate_leaves_thinking_level_null(self, db):
        proc = _make_processor(channel='goal_pursuit')
        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'user', 'goal msg', datetime('now'))"
        ).lastrowid
        db.commit()

        proc._uid = uid
        proc._run_thinking_gate()

        row = db.execute(
            "SELECT thinking_level FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] is None


# ── Plan call passes tools=[] (explicit empty — not None) ────────────────────

class TestPlanCallToolsEmpty:
    """The planning LLM call must pass tools=[] (explicit empty list), NOT
    tools=None. Providers.send_messages treats tools=None as "use job
    defaults" and falls back to the full innate-skill catalog, which would
    let the planner actually execute tools mid-plan. tools=[] is the only
    way to guarantee the planner produces plain text.
    """

    def test_generate_thinking_plan_passes_tools_empty_list(self, db):
        proc = _make_processor(channel='user', raw_input='design an auth system')

        from services.llm_service import LLMResponse
        captured_kwargs = {}

        def capture_send(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return LLMResponse(text='Step 1: analyse. Step 2: plan.', model='test', provider='test')

        with patch('services.providers.Providers.instance') as mock_pi:
            mock_providers = MagicMock()
            mock_providers.send_messages.side_effect = capture_send
            mock_pi.return_value = mock_providers

            proc._generate_thinking_plan()

        assert 'tools' in captured_kwargs, (
            "_generate_thinking_plan() must pass an explicit tools kwarg to "
            "Providers.send_messages(). Omitting it lets the provider fall back "
            "to whatever defaults are configured for the job."
        )
        assert captured_kwargs['tools'] == [], (
            f"Expected tools=[] (explicit empty list), got {captured_kwargs['tools']!r}. "
            "tools=None would be coerced to the job's default tool list by "
            "Providers.send_messages — the planner could then call tools."
        )
        assert captured_kwargs['tools'] is not None, (
            "tools must NOT be None — Providers.send_messages treats None as "
            "'resolve job defaults'. Only an explicit [] guarantees zero tools."
        )


class TestPlanCallUserBodyComposition:
    """The planner user message must include a native-tool catalog + a
    compact Previous Messages block, per the user's spec ("Analyse the
    tools, prompt and chat history you have been provided"). Tools go in
    as TEXT — not via the tools= kwarg — so the planner can reason ABOUT
    them without the ability to call them.
    """

    def _capture_messages(self, proc):
        """Run _generate_thinking_plan and capture the messages passed."""
        from services.llm_service import LLMResponse
        captured = {}

        def capture_send(system, messages, **kwargs):
            captured['system'] = system
            captured['messages'] = messages
            captured['kwargs'] = kwargs
            return LLMResponse(text='plan', model='test', provider='test')

        with patch('services.providers.Providers.instance') as mock_pi:
            mock_providers = MagicMock()
            mock_providers.send_messages.side_effect = capture_send
            mock_pi.return_value = mock_providers
            proc._generate_thinking_plan()
        return captured

    def test_plan_body_includes_native_tool_catalog(self, db):
        """The planner user message must include an '### Available Tools'
        section rendered from NATIVE_TOOLS — not passed via tools= kwarg.
        """
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import SystemMessagePrompt

        class _StubPrompt(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'BASE'

        class _Proc(MessageProcessor):
            CHANNEL = 'user'
            ROLE = 'user'
            SYSTEM_PROMPT_CLASS = _StubPrompt
            NATIVE_TOOLS = ['memory']  # use a real innate skill name

            def getUserDefinition(self):
                return 'user'

            def getUserPrompt(self):
                return 'x'

        proc = _Proc('design a system')
        captured = self._capture_messages(proc)

        content = captured['messages'][0]['content']
        assert '### Available Tools' in content, (
            "Planner body must render a '### Available Tools' section — "
            "the planner needs to reason about available tools as TEXT."
        )
        # The native 'memory' skill should appear in the catalog.
        assert 'memory' in content

    def test_plan_body_includes_current_user_message(self, db):
        proc = _make_processor(channel='user', raw_input='compare A and B')
        captured = self._capture_messages(proc)

        content = captured['messages'][0]['content']
        assert '### Current User Message' in content
        assert 'compare A and B' in content

    def test_plan_body_includes_verbatim_spec_prompt(self, db):
        """The '### Your Task' section must contain the user's spec prompt
        verbatim (the planning instructions the user explicitly dictated).
        """
        proc = _make_processor(channel='user', raw_input='x')
        captured = self._capture_messages(proc)

        content = captured['messages'][0]['content']
        assert '### Your Task' in content
        assert 'DO NOT MAKE TOOL CALLS' in content
        assert 'externalise your thoughts' in content.lower()

    def test_plan_body_includes_previous_messages_when_present(self, db):
        """When the transcript has prior rows, the planner body must
        include a '### Previous Messages' section so the LLM can consider
        the chat history (per the spec).
        """
        # Seed prior transcript rows.
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'earlier turn', datetime('now', '-2 seconds'))"
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'assistant', 'earlier reply', datetime('now', '-1 second'))"
        )
        db.commit()

        proc = _make_processor(channel='user', raw_input='follow-up')
        captured = self._capture_messages(proc)

        content = captured['messages'][0]['content']
        assert '### Previous Messages' in content
        assert 'earlier turn' in content
        assert 'earlier reply' in content

    def test_plan_body_omits_previous_messages_when_empty(self, db):
        """No transcript → no empty '### Previous Messages' section
        (don't waste tokens on an empty header)."""
        proc = _make_processor(channel='user', raw_input='first turn')
        captured = self._capture_messages(proc)

        content = captured['messages'][0]['content']
        assert '### Previous Messages' not in content


# ── Gate-level exception → turn continues ─────────────────────────────────────

class TestGateLevelException:
    """If something goes wrong in _run_thinking_gate() at a level above the
    classifier (e.g., the import fails, a DB read raises an unexpected error),
    the gate must NOT propagate the exception — the turn must continue.
    """

    def test_gate_exception_does_not_propagate(self, db):
        proc = _make_processor(channel='user', raw_input='hello')

        uid = db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'hello', datetime('now'))"
        ).lastrowid
        db.commit()
        proc._uid = uid

        # Patch ThinkingLevelClassifierService to raise on construction
        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            side_effect=RuntimeError("failed to load ONNX head"),
        ):
            # Must not raise
            proc._run_thinking_gate()

        assert proc._thinking_level == 'medium'
        assert proc._thinking_plan is None

    def test_gate_exception_system_prompt_still_works(self, db):
        proc = _make_processor(channel='user', raw_input='hello')

        with patch(
            'services.thinking_level_classifier_service.ThinkingLevelClassifierService',
            side_effect=RuntimeError("failed to load ONNX head"),
        ):
            proc._run_thinking_gate()

        # getSystemPrompt() must return a usable string — no crash
        prompt = proc.getSystemPrompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Gate exception defaulted to 'medium', so [RULE] tag should be present
        assert '[RULE]' in prompt


# ── Tags appear at VERY END of system prompt ──────────────────────────────────

class TestTagsAtEndOfPrompt:
    """[RULE] and [THINKING] tags must be at the very end of the system prompt,
    after all other content. This preserves the KV-cache stable prefix above.
    """

    def test_rule_tag_is_last_content(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'medium'
        prompt = proc.getSystemPrompt()
        assert prompt.endswith('[/RULE]')
        rule_pos = prompt.index('[RULE]')
        assert prompt[rule_pos - 2:rule_pos] == '\n\n', (
            "[RULE] tag must be separated from prompt body by double newline"
        )

    def test_thinking_tag_is_last_content(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = 'Analyse, then plan, then execute.'
        prompt = proc.getSystemPrompt()
        assert prompt.endswith('[/THINKING]')
        thinking_pos = prompt.index('[THINKING]')
        assert prompt[thinking_pos - 2:thinking_pos] == '\n\n', (
            "[THINKING] tag must be separated from prompt body by double newline"
        )

    def test_no_content_after_rule_tag(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'medium'
        prompt = proc.getSystemPrompt()
        close_tag_pos = prompt.rindex('[/RULE]')
        trailing = prompt[close_tag_pos + len('[/RULE]'):]
        assert trailing == '', (
            f"Nothing should follow [/RULE] in the system prompt, got: {trailing!r}"
        )

    def test_no_content_after_thinking_tag(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = 'Plan here.'
        prompt = proc.getSystemPrompt()
        close_tag_pos = prompt.rindex('[/THINKING]')
        trailing = prompt[close_tag_pos + len('[/THINKING]'):]
        assert trailing == '', (
            f"Nothing should follow [/THINKING] in the system prompt, got: {trailing!r}"
        )


# ── apply_thinking_level on high+None plan: falls through to [RULE] ──────────

class TestHighLevelNoPlanBehavior:
    """When level='high' but _thinking_plan is None (e.g. plan call failed
    and the level wasn't degraded, or a future code path leaves it in that
    state), _apply_thinking_level must fall through to the [RULE] fallback
    — never silently drop to "no guidance", which would give LESS
    deliberation prompting than a 'medium' classification.
    """

    def test_high_no_plan_no_thinking_tag_appended(self, db):
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = None
        prompt = proc.getSystemPrompt()
        assert '[THINKING]' not in prompt

    def test_high_no_plan_falls_through_to_rule(self, db):
        # level='high', plan=None — must emit the [RULE] fallback, NOT a
        # bare prompt. Consistent degradation with 'medium'.
        proc = _make_processor(channel='user')
        proc._thinking_level = 'high'
        proc._thinking_plan = None
        prompt = proc.getSystemPrompt()
        assert '[RULE]' in prompt
        assert '[/RULE]' in prompt
        assert prompt.endswith('[/RULE]')

    def test_high_no_plan_prompt_equals_medium_prompt(self, db):
        # With level='medium', prompt ends with [RULE]...[/RULE].
        # With level='high' + plan=None, we fall through to the same [RULE]
        # branch, so the output must be identical.
        proc_medium = _make_processor(channel='user')
        proc_medium._thinking_level = 'medium'
        prompt_medium = proc_medium.getSystemPrompt()

        proc_high_no_plan = _make_processor(channel='user')
        proc_high_no_plan._thinking_level = 'high'
        proc_high_no_plan._thinking_plan = None
        prompt_high_no_plan = proc_high_no_plan.getSystemPrompt()

        assert prompt_medium == prompt_high_no_plan

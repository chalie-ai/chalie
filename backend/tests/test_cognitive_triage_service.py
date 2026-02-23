"""Unit tests for CognitiveTriageService."""
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


@pytest.fixture
def triage_context():
    """Build a default TriageContext for testing."""
    # Import inline to avoid requiring the actual service to exist yet
    # We'll define a minimal context dict and patch the dataclass
    return {
        'context_warmth': 0.5,
        'memory_confidence': 0.5,
        'working_memory_turns': 2,
        'gist_count': 3,
        'fact_count': 5,
        'previous_mode': 'RESPOND',
        'previous_tools': [],
        'tool_summaries': '## Information Retrieval\n- duckduckgo_search: Search the web',
        'working_memory_summary': 'U: hello | A: hi there',
    }


class TestSocialFilter:
    """Test the regex-based social fast filter."""

    def test_empty_text_returns_ignore(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("")
        assert result is not None
        assert result.mode == 'IGNORE'
        assert result.branch == 'social'
        assert result.fast_filtered is True

    def test_whitespace_returns_ignore(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("   ")
        assert result is not None
        assert result.mode == 'IGNORE'

    def test_greeting_returns_acknowledge(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        for greeting in ["hey", "Hi there", "hello!", "Good morning"]:
            result = svc._social_filter(greeting)
            assert result is not None, f"Expected social result for '{greeting}'"
            assert result.mode == 'ACKNOWLEDGE'
            assert result.branch == 'social'

    def test_greeting_with_question_passes_through(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("hey, what's the weather today?")
        assert result is None  # Should NOT be filtered — has a real question

    def test_positive_feedback_returns_acknowledge(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        for feedback in ["thanks!", "thank you", "great", "that works"]:
            result = svc._social_filter(feedback)
            assert result is not None, f"Expected social result for '{feedback}'"
            assert result.mode == 'ACKNOWLEDGE'

    def test_cancel_returns_cancel(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        for cancel in ["never mind", "cancel", "forget it", "don't bother"]:
            result = svc._social_filter(cancel)
            assert result is not None, f"Expected social result for '{cancel}'"
            assert result.mode == 'CANCEL'

    def test_self_resolved_returns_acknowledge(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("all good, found it")
        assert result is not None
        assert result.mode == 'ACKNOWLEDGE'

    def test_normal_question_returns_none(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("what's the weather in London?")
        assert result is None  # Not social, should go to LLM

    def test_factual_question_returns_none(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = svc._social_filter("search for best Python tutorials")
        assert result is None


class TestSelfEvalRules:
    """Test deterministic self-eval guardrail rules."""

    def _make_result(self, branch, mode, tools=None, **kwargs):
        from services.cognitive_triage_service import TriageResult
        defaults = dict(
            confidence_internal=0.5,
            confidence_tool_need=0.5,
            freshness_risk=0.3,
            decision_entropy=0.0,
            reasoning='test',
            triage_time_ms=0.0,
            fast_filtered=False,
            self_eval_override=False,
            self_eval_reason='',
        )
        defaults.update(kwargs)
        return TriageResult(
            branch=branch, mode=mode, tools=tools or [], skills=[], **defaults
        )

    def _make_context(self, **kwargs):
        from services.cognitive_triage_service import TriageContext
        defaults = dict(
            context_warmth=0.5,
            memory_confidence=0.5,
            working_memory_turns=2,
            gist_count=3,
            fact_count=5,
            previous_mode='RESPOND',
            previous_tools=[],
            tool_summaries='## Info\n- search: searches things',
            working_memory_summary='',
        )
        defaults.update(kwargs)
        return TriageContext(**defaults)

    def test_rule1_act_without_tools_downgrades_to_respond(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        # Empty tool_summaries so _pick_default_tool returns '' → triggers downgrade
        ctx = self._make_context(tool_summaries='')
        result = svc._self_evaluate(result, "search for something", ctx)
        assert result.branch == 'respond'
        assert result.mode == 'RESPOND'
        assert result.self_eval_override is True
        assert result.self_eval_reason == 'act_without_tools'

    def test_rule1_act_with_tools_not_modified(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=['duckduckgo_search'])
        ctx = self._make_context()
        result = svc._self_evaluate(result, "search for something", ctx)
        assert result.branch == 'act'
        assert result.self_eval_override is False

    def test_rule2_act_failsafe_escalates_respond_to_act(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('respond', 'RESPOND', freshness_risk=0.8)
        ctx = self._make_context(memory_confidence=0.1, tool_summaries='## Info\n- search: web search')
        # "What is the current Bitcoin price?" — factual question
        result = svc._self_evaluate(result, "What is the current Bitcoin price?", ctx)
        assert result.branch == 'act'
        assert result.self_eval_reason == 'act_failsafe'

    def test_rule2_no_escalation_without_tools(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('respond', 'RESPOND', freshness_risk=0.8)
        ctx = self._make_context(memory_confidence=0.1, tool_summaries='')  # No tools
        result = svc._self_evaluate(result, "What is the current Bitcoin price?", ctx)
        assert result.branch == 'respond'  # No tools available, can't escalate

    def test_rule3_social_with_question_becomes_respond(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('social', 'ACKNOWLEDGE')
        ctx = self._make_context()
        result = svc._self_evaluate(result, "hey what time is it?", ctx)
        assert result.branch == 'respond'
        assert result.self_eval_reason == 'social_with_question'

    def test_rule4_anti_oscillation_same_tool(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result(
            'act', 'ACT', tools=['duckduckgo_search'],
            confidence_tool_need=0.5
        )
        ctx = self._make_context(
            previous_mode='ACT',
            previous_tools=['duckduckgo_search'],
        )
        result = svc._self_evaluate(result, "search again", ctx)
        # confidence_tool_need was 0.5, after *0.7 = 0.35, which is < 0.4
        assert result.branch == 'respond'
        assert result.self_eval_reason == 'anti_oscillation_same_tool'

    def test_rule4_anti_oscillation_different_tool_not_suppressed(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result(
            'act', 'ACT', tools=['weather'],
            confidence_tool_need=0.8
        )
        ctx = self._make_context(
            previous_mode='ACT',
            previous_tools=['duckduckgo_search'],  # Different tool
        )
        result = svc._self_evaluate(result, "check weather", ctx)
        assert result.branch == 'act'  # Not suppressed — different tool

    def test_heuristic_fallback_question_gives_respond(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        from services.cognitive_triage_service import TriageContext
        ctx = TriageContext(
            context_warmth=0.5, memory_confidence=0.5,
            working_memory_turns=0, gist_count=0, fact_count=0,
            previous_mode='RESPOND', previous_tools=[],
            tool_summaries='', working_memory_summary='',
        )
        result = svc._heuristic_fallback("What is quantum computing?", ctx)
        assert result.branch == 'respond'

    def test_heuristic_fallback_command_gives_act_when_tools_available(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        from services.cognitive_triage_service import TriageContext
        ctx = TriageContext(
            context_warmth=0.5, memory_confidence=0.5,
            working_memory_turns=0, gist_count=0, fact_count=0,
            previous_mode='RESPOND', previous_tools=[],
            tool_summaries='## Info\n- search: web search',
            working_memory_summary='',
        )
        result = svc._heuristic_fallback("search for Python tutorials", ctx)
        assert result.branch == 'act'


class TestTriageFull:
    """Integration-style tests for the full triage flow (with mocked LLM)."""

    def _make_context(self, **kwargs):
        from services.cognitive_triage_service import TriageContext
        defaults = dict(
            context_warmth=0.5, memory_confidence=0.5,
            working_memory_turns=2, gist_count=2, fact_count=3,
            previous_mode='RESPOND', previous_tools=[],
            tool_summaries='## Info\n- duckduckgo_search: web search',
            working_memory_summary='U: hi | A: hello',
        )
        defaults.update(kwargs)
        return TriageContext(**defaults)

    def test_triage_greeting_fast_path(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("hey!", ctx)
        assert result.branch == 'social'
        assert result.fast_filtered is True
        assert result.triage_time_ms >= 0

    def test_triage_cancel_fast_path(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("never mind", ctx)
        assert result.branch == 'social'
        assert result.mode == 'CANCEL'
        assert result.fast_filtered is True

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_llm_respond_branch(self, mock_get_llm):
        """LLM returns RESPOND → should get respond branch."""
        import json
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text=json.dumps({
            'mode': 'RESPOND',
            'tools': [],
            'confidence_internal': 0.8,
            'confidence_tool_need': 0.1,
            'freshness_risk': 0.0,
            'reasoning': 'Memory is sufficient',
        }))
        mock_get_llm.return_value = mock_llm

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("What is Python?", ctx)
        assert result.branch == 'respond'
        assert result.mode == 'RESPOND'
        assert result.fast_filtered is False

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_llm_act_branch_with_tools(self, mock_get_llm):
        """LLM returns ACT with tools → should get act branch."""
        import json
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text=json.dumps({
            'mode': 'ACT',
            'tools': ['duckduckgo_search'],
            'confidence_internal': 0.1,
            'confidence_tool_need': 0.9,
            'freshness_risk': 0.7,
            'reasoning': 'Need current data',
        }))
        mock_get_llm.return_value = mock_llm

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("What's the current Bitcoin price?", ctx)
        assert result.branch == 'act'
        assert 'duckduckgo_search' in result.tools

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_llm_act_no_tools_self_eval_downgrades(self, mock_get_llm):
        """LLM returns ACT with no tools → self-eval downgrades to respond."""
        import json
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text=json.dumps({
            'mode': 'ACT',
            'tools': [],
            'confidence_internal': 0.2,
            'confidence_tool_need': 0.8,
            'freshness_risk': 0.5,
            'reasoning': 'Needs tools but forgot to list them',
        }))
        mock_get_llm.return_value = mock_llm

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        # Empty tool_summaries so _pick_default_tool returns '' → downgrade fires
        ctx = self._make_context(tool_summaries='')
        result = svc.triage("do something external", ctx)
        assert result.branch == 'respond'  # Downgraded by self-eval
        assert result.self_eval_override is True
        assert result.self_eval_reason == 'act_without_tools'

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_llm_timeout_fallback(self, mock_get_llm):
        """LLM raises exception → heuristic fallback is used."""
        mock_get_llm.return_value = MagicMock(
            send_message=MagicMock(side_effect=TimeoutError("LLM timeout"))
        )

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("search for Python tutorials", ctx)
        # Should fall back gracefully, not raise
        assert result.branch in ('act', 'respond', 'clarify', 'social')
        assert result.triage_time_ms >= 0

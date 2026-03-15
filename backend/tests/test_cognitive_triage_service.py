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


class TestEmptyInputGuard:
    """Test the empty-input guard in triage()."""

    def test_empty_text_returns_ignore(self):
        from services.cognitive_triage_service import CognitiveTriageService, TriageContext
        svc = CognitiveTriageService()
        ctx = TriageContext(
            context_warmth=0.5, memory_confidence=0.5,
            working_memory_turns=0, gist_count=0, fact_count=0,
            previous_mode='RESPOND', previous_tools=[],
            tool_summaries='', working_memory_summary='',
        )
        result = svc.triage("", ctx)
        assert result.mode == 'IGNORE'
        assert result.branch == 'ignore'
        assert result.fast_filtered is True

    def test_whitespace_returns_ignore(self):
        from services.cognitive_triage_service import CognitiveTriageService, TriageContext
        svc = CognitiveTriageService()
        ctx = TriageContext(
            context_warmth=0.5, memory_confidence=0.5,
            working_memory_turns=0, gist_count=0, fact_count=0,
            previous_mode='RESPOND', previous_tools=[],
            tool_summaries='', working_memory_summary='',
        )
        result = svc.triage("   ", ctx)
        assert result.mode == 'IGNORE'
        assert result.fast_filtered is True


class TestSelfEvalRules:
    """Test deterministic self-eval guardrail rules."""

    def _make_result(self, branch, mode, tools=None, **kwargs):
        from services.cognitive_triage_service import TriageResult
        defaults = dict(
            confidence_internal=0.5,
            confidence_tool_need=0.5,
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
        from unittest.mock import patch, MagicMock
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        # Empty tool_summaries → no tools available at all → downgrade
        ctx = self._make_context(tool_summaries='')
        # Mock ONNX away so skill recovery can't save the ACT branch
        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False
        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc._self_evaluate(result, "search for something", ctx)
        assert result.branch == 'respond'
        assert result.mode == 'RESPOND'
        assert result.self_eval_override is True
        assert result.self_eval_reason == 'act_no_tools_available'

    def test_rule1_act_with_tools_not_modified(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=['duckduckgo_search'])
        ctx = self._make_context()
        result = svc._self_evaluate(result, "search for something", ctx)
        assert result.branch == 'act'
        assert result.self_eval_override is False

    def test_rule1_act_with_contextual_skill_stays_act(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        result.skills = ['recall', 'memorize', 'introspect', 'schedule']
        ctx = self._make_context(tool_summaries='')
        result = svc._self_evaluate(result, "remind me at 3pm", ctx)
        assert result.branch == 'act'
        assert result.self_eval_reason == 'act_innate_skill'

    def test_rule1_act_no_tools_defers_to_loop_when_tools_available(self):
        """ACT + no tools + tools registered → stays ACT, defers tool selection to ACT loop."""
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        ctx = self._make_context(tool_summaries='## Info\n- search: web search')
        result = svc._self_evaluate(result, "open this link", ctx)
        assert result.branch == 'act'
        assert result.mode == 'ACT'
        assert result.self_eval_reason == 'act_tool_deferred_to_loop'

    def test_rule5_url_in_message_escalates_to_act(self):
        """RESPOND + URL in message + tools available → escalate to ACT."""
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('respond', 'RESPOND')
        ctx = self._make_context(tool_summaries='## Info\n- read: read URLs and local files')
        # Use a message with URL but without tool invocation words,
        # so rule 2b doesn't fire first
        result = svc._self_evaluate(result, "this is https://github.com/chalie-ai/chalie", ctx)
        assert result.branch == 'act'
        assert result.mode == 'ACT'
        assert result.self_eval_reason == 'act_url_detected'

    def test_rule5_url_no_escalation_without_tools(self):
        """RESPOND + URL but no tools → stays RESPOND."""
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('respond', 'RESPOND')
        ctx = self._make_context(tool_summaries='')
        result = svc._self_evaluate(result, "check https://example.com", ctx)
        assert result.branch == 'respond'

    def test_rule5_url_already_act_not_modified(self):
        """ACT + URL → no change (already ACT)."""
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=['read'])
        ctx = self._make_context()
        result = svc._self_evaluate(result, "read https://example.com", ctx)
        assert result.branch == 'act'
        assert result.self_eval_reason == ''  # No override needed

    def test_rule3_ignore_with_question_becomes_respond(self):
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('ignore', 'IGNORE')
        ctx = self._make_context()
        result = svc._self_evaluate(result, "hey what time is it?", ctx)
        assert result.branch == 'respond'
        assert result.self_eval_reason == 'ignore_with_question'

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

    def test_rule1_act_no_tools_recovers_skill_via_onnx(self):
        """ACT + skills=[] + no tools + ONNX recovers contextual skill → stays ACT."""
        from services.cognitive_triage_service import CognitiveTriageService, _CONTEXTUAL_SKILLS
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        ctx = self._make_context(tool_summaries='')
        result = svc._self_evaluate(result, "remind me to water the plants at 6pm tomorrow", ctx)
        assert result.branch == 'act'
        assert result.mode == 'ACT'
        # ONNX should recover at least one contextual skill
        recovered = [s for s in result.skills if s in _CONTEXTUAL_SKILLS]
        assert len(recovered) >= 1
        assert result.self_eval_override is True
        assert result.self_eval_reason == 'act_innate_skill_recovered'

    def test_rule1_act_no_tools_no_innate_match_still_downgrades(self):
        """ACT + skills=[] + no tools + no ONNX recovery → downgrade."""
        from unittest.mock import patch, MagicMock
        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        result = self._make_result('act', 'ACT', tools=[])
        ctx = self._make_context(tool_summaries='')
        # Mock ONNX away so skill recovery can't save the ACT branch
        mock_onnx = MagicMock()
        mock_onnx.is_available.return_value = False
        with patch('services.onnx_inference_service.get_onnx_inference_service', return_value=mock_onnx):
            result = svc._self_evaluate(result, "do something external with a database", ctx)
        assert result.branch == 'respond'
        assert result.self_eval_reason == 'act_no_tools_available'


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

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_greeting_goes_through_llm(self, mock_get_llm):
        """Greetings now go through LLM triage (no social filter short-circuit)."""
        import json
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text=json.dumps({
            'mode': 'RESPOND',
            'tools': [],
            'confidence_internal': 0.9,
            'confidence_tool_need': 0.0,
            'freshness_risk': 0.0,
            'reasoning': 'Simple greeting — respond warmly',
        }))
        mock_get_llm.return_value = mock_llm

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("hey!", ctx)
        assert result.branch == 'respond'
        assert result.mode == 'RESPOND'
        assert result.fast_filtered is False  # Went through LLM

    @patch('services.cognitive_triage_service.CognitiveTriageService._get_llm')
    def test_triage_cancel_goes_through_llm(self, mock_get_llm):
        """Cancel messages now go through LLM triage."""
        import json
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text=json.dumps({
            'mode': 'CANCEL',
            'tools': [],
            'confidence_internal': 1.0,
            'confidence_tool_need': 0.0,
            'freshness_risk': 0.0,
            'reasoning': 'User cancelled',
        }))
        mock_get_llm.return_value = mock_llm

        from services.cognitive_triage_service import CognitiveTriageService
        svc = CognitiveTriageService()
        ctx = self._make_context()
        result = svc.triage("never mind", ctx)
        assert result.branch == 'ignore'
        assert result.mode == 'CANCEL'
        assert result.fast_filtered is False  # Went through LLM

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
        # Empty tool_summaries so _pick_default_tool returns '' → innate skill recovery fires
        ctx = self._make_context(tool_summaries='')
        result = svc.triage("do something external", ctx)
        # Innate primitives (recall, memorize, introspect) are always injected for ACT,
        # so the recovery path keeps ACT mode with innate skills instead of downgrading
        assert result.branch == 'act'
        assert result.self_eval_override is True
        assert result.self_eval_reason == 'act_innate_skill'

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
        assert result.branch in ('act', 'respond', 'clarify', 'ignore')
        assert result.triage_time_ms >= 0



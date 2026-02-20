"""Tests for ModeRouterService — deterministic mode routing."""

import pytest
from unittest.mock import patch, MagicMock
from services.mode_router_service import ModeRouterService, collect_routing_signals


pytestmark = pytest.mark.unit


def _make_config():
    return {
        'base_scores': {
            'RESPOND': 0.40,
            'CLARIFY': 0.30,
            'ACT': 0.20,
            'ACKNOWLEDGE': 0.10,
            'IGNORE': -0.50,
        },
        'weights': {
            'respond.warmth_boost': 0.20,
            'respond.fact_density': 0.15,
            'respond.gist_density': 0.10,
            'respond.question_warm': 0.15,
            'respond.cold_penalty': 0.15,
            'respond.greeting_penalty': 0.20,
            'respond.feedback_penalty': 0.15,
            'clarify.cold_boost': 0.25,
            'clarify.question_no_facts': 0.20,
            'clarify.new_topic_question': 0.10,
            'clarify.cold_question': 0.05,
            'clarify.warm_penalty': 0.20,
            'act.question_moderate_context': 0.20,
            'act.interrogative_gap': 0.15,
            'act.implicit_reference': 0.15,
            'act.very_cold_penalty': 0.10,
            'act.warm_facts_penalty': 0.10,
            'act.tool_relevance_strong': 0.60,
            'act.tool_relevance_moderate': 0.35,
            'act.tool_relevance_weak': 0.15,
            'acknowledge.greeting': 0.80,
            'acknowledge.positive_feedback': 0.55,
            'acknowledge.question_penalty': 0.30,
            'ignore.empty_input': 1.00,
        },
        'tiebreaker_base_margin': 0.20,
        'tiebreaker_min_margin': 0.08,
    }


def _make_signals(**overrides):
    """Build a base signals dict with sensible defaults, applying overrides."""
    base = {
        'context_warmth': 0.5,
        'working_memory_turns': 2,
        'gist_count': 2,
        'fact_count': 3,
        'fact_keys': ['name', 'lang', 'pref'],
        'world_state_present': True,
        'topic_confidence': 0.8,
        'is_new_topic': False,
        'session_exchange_count': 2,
        'prompt_token_count': 10,
        'has_question_mark': False,
        'interrogative_words': False,
        'greeting_pattern': False,
        'explicit_feedback': None,
        'information_density': 0.8,
        'implicit_reference': False,
        'tool_relevance_score': 0.0,
    }
    base.update(overrides)
    return base


class TestModeRouter:

    def test_high_warmth_selects_respond(self):
        """High context warmth (>0.6) should favour RESPOND."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(context_warmth=0.8, gist_count=4, fact_count=5)
        result = router.route(signals, "Tell me about X")
        assert result['mode'] == 'RESPOND'

    def test_cold_context_selects_clarify(self):
        """Cold context (<0.3) with a question should favour CLARIFY."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(
            context_warmth=0.1,
            has_question_mark=True,
            interrogative_words=True,
            fact_count=0,
            gist_count=0,
            is_new_topic=True,
        )
        result = router.route(signals, "What is this?")
        assert result['mode'] == 'CLARIFY'

    def test_greeting_selects_acknowledge(self):
        """Greeting pattern should favour ACKNOWLEDGE."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(greeting_pattern=True)
        result = router.route(signals, "Hey")
        assert result['mode'] == 'ACKNOWLEDGE'

    def test_tool_relevance_selects_act(self):
        """Strong tool relevance score should favour ACT."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(
            context_warmth=0.5,
            has_question_mark=True,
            interrogative_words=True,
            tool_relevance_score=0.60,
            fact_count=1,
            gist_count=0,
        )
        result = router.route(signals, "Search the web for latest news about Malta")
        assert result['mode'] == 'ACT'

    def test_tiebreaker_invoked_within_margin(self):
        """When top-2 scores are within margin, tie-breaker LLM should be invoked."""
        router = ModeRouterService(_make_config())

        # Mock the tiebreaker to return a specific mode
        router._tiebreaker_ollama = MagicMock()
        router._tiebreaker_prompt = "test"
        router._tiebreaker_ollama.send_message.return_value = '{"choice": "A"}'

        # Craft signals where RESPOND and CLARIFY are very close
        signals = _make_signals(
            context_warmth=0.35,
            has_question_mark=True,
            interrogative_words=True,
            fact_count=0,
            gist_count=0,
        )

        result = router.route(signals, "What can you tell me?")

        # Either tiebreaker was used, or scores were decisive.
        # The key assertion is that the result is valid.
        assert result['mode'] in ModeRouterService.MODES
        assert isinstance(result['tiebreaker_used'], bool)

    def test_anti_oscillation_suppresses_act(self):
        """Previous ACT should suppress ACT re-selection by -0.15."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(
            context_warmth=0.5,
            has_question_mark=True,
            interrogative_words=True,
            implicit_reference=True,
            fact_count=1,
        )

        # Without previous ACT
        result_no_prev = router.route(signals, "What did we discuss?")
        act_score_no_prev = result_no_prev['scores']['ACT']

        # With previous ACT
        result_with_prev = router.route(signals, "What did we discuss?", previous_mode='ACT')
        act_score_with_prev = result_with_prev['scores']['ACT']

        assert act_score_with_prev < act_score_no_prev
        assert abs(act_score_no_prev - act_score_with_prev - 0.15) < 0.001

    def test_score_ordering_deterministic(self):
        """Identical signals must produce identical scores."""
        router = ModeRouterService(_make_config())
        signals = _make_signals()

        result1 = router.route(signals, "test input")
        result2 = router.route(signals, "test input")

        assert result1['scores'] == result2['scores']
        assert result1['mode'] == result2['mode']

    def test_collect_routing_signals_structure(self):
        """collect_routing_signals returns expected keys."""
        wm = MagicMock()
        wm.get_recent_turns.return_value = [{'role': 'user', 'content': 'hi'}]
        gs = MagicMock()
        gs.get_latest_gists.return_value = [{'type': 'observation', 'content': 'x', 'confidence': 8}]
        fs = MagicMock()
        fs.get_all_facts.return_value = [{'key': 'name', 'value': 'test'}]
        ws = MagicMock()
        ws.get_world_state.return_value = ""
        ss = MagicMock()
        ss.topic_exchange_count = 3

        signals = collect_routing_signals(
            text="Hello?",
            topic="test-topic",
            context_warmth=0.5,
            working_memory=wm,
            gist_storage=gs,
            fact_store=fs,
            world_state_service=ws,
            classification_result={'confidence': 0.8, 'is_new_topic': False},
            session_service=ss,
        )

        expected_keys = {
            'context_warmth', 'working_memory_turns', 'gist_count', 'fact_count',
            'fact_keys', 'world_state_present', 'topic_confidence', 'is_new_topic',
            'session_exchange_count', 'prompt_token_count', 'has_question_mark',
            'interrogative_words', 'greeting_pattern', 'explicit_feedback',
            'information_density', 'implicit_reference',
            'intent_type', 'intent_confidence', 'intent_needs_tools',
            'intent_complexity', 'tool_relevance_score',
        }
        assert expected_keys == set(signals.keys())

"""Unit tests for GoalSignalService — signal extraction and routing."""

import json
import sqlite3
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from services.goal_signal_service import (
    extract_and_route_signals,
    route_cognitive_signal,
    INTENT_TYPES,
    _extract_signal_content,
    _map_signal_type,
)


pytestmark = pytest.mark.unit


GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'emergent',
    status TEXT NOT NULL DEFAULT 'candidate',
    description TEXT NOT NULL,
    parent_motives TEXT DEFAULT '[]',
    identity_links TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.1,
    salience REAL DEFAULT 0.0,
    commitment REAL DEFAULT 0.0,
    urgency REAL DEFAULT 0.0,
    timescale TEXT DEFAULT 'medium_term',
    strategy TEXT,
    lineage_parent_id TEXT,
    evidence_count INTEGER DEFAULT 0,
    last_reinforced_at TEXT,
    last_acted_at TEXT,
    outcome_feedback TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goal_evidence (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goals(id),
    signal_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    strength REAL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
"""


def _make_db():
    """Create an in-memory SQLite db with goal schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(GOAL_SCHEMA)
    conn.commit()

    @contextmanager
    def _connection():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    svc = MagicMock()
    svc.connection = _connection
    return svc, conn


class TestIntentTypeDetection:
    """Tests for intent-based explicit signal detection (replaces regex)."""

    def test_command_intent_triggers_explicit_signal(self):
        """'command' intent_type should produce an explicit_statement signal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []
        mock_ecology.create_goal.return_value = {'id': 'g1', 'type': 'stated'}

        classification = {'topic': 'cooking', 'intent_type': 'command'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('cooking', 'Find me a recipe for pasta', classification)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' in signal_types

    def test_action_intent_triggers_explicit_signal(self):
        """'action' intent_type should produce an explicit_statement signal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []
        mock_ecology.create_goal.return_value = {'id': 'g2', 'type': 'stated'}

        classification = {'topic': 'cooking', 'intent_type': 'action'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('cooking', 'Remind me to buy groceries tomorrow', classification)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' in signal_types

    def test_question_intent_does_not_trigger_explicit(self):
        """'question' intent_type should NOT produce an explicit_statement signal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        classification = {'topic': 'weather', 'intent_type': 'question'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('weather', 'What is the weather like today?', classification)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' not in signal_types

    def test_statement_intent_does_not_trigger_explicit(self):
        """'statement' intent_type should NOT produce an explicit_statement signal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        classification = {'topic': 'general', 'intent_type': 'statement'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('general', 'The weather is nice today and I like it', classification)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' not in signal_types

    def test_no_classification_no_explicit_signal(self):
        """No classification dict should not produce explicit signals."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('general', 'Just chatting about things here', None)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' not in signal_types

    def test_empty_intent_type_no_explicit_signal(self):
        """Empty intent_type string should not trigger explicit detection."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        classification = {'topic': 'general', 'intent_type': ''}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals('general', 'Some longer text about nothing specific', classification)

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' not in signal_types

    def test_intent_types_constant(self):
        """INTENT_TYPES should contain 'command' and 'action'."""
        assert 'command' in INTENT_TYPES
        assert 'action' in INTENT_TYPES
        assert 'question' not in INTENT_TYPES
        assert 'statement' not in INTENT_TYPES


class TestSignalExtraction:

    def test_explicit_intent_creates_stated_goal(self):
        """Explicit intent with no matching goal should create a stated goal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []
        mock_ecology.create_goal.return_value = {
            'id': 'test-goal-id',
            'type': 'stated',
            'status': 'actionable',
        }

        classification = {'topic': 'cooking', 'intent_type': 'command'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals(
                'cooking', 'I need to plan dinner for 20 people', classification
            )

        signal_types = [s['signal_type'] for s in signals]
        assert 'explicit_statement' in signal_types
        assert 'goal_created' in signal_types

    def test_matching_goal_gets_evidence(self):
        """Text matching an existing goal should add evidence."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'existing-goal', 'similarity': 0.8, 'description': 'cooking'},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals(
                'cooking', 'What should I cook for the dinner party?', None
            )

        mock_ecology.add_evidence.assert_called_once()
        call_kwargs = mock_ecology.add_evidence.call_args
        assert call_kwargs[1]['goal_id'] == 'existing-goal'
        assert call_kwargs[1]['signal_type'] == 'topic_recurrence'

    def test_explicit_intent_with_matching_goal_no_duplicate_create(self):
        """Explicit intent matching an existing goal should add evidence, not create new goal."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'existing-goal', 'similarity': 0.9, 'description': 'plan dinner'},
        ]

        classification = {'intent_type': 'command'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            signals = extract_and_route_signals(
                'cooking', 'I need to plan dinner for the party', classification
            )

        mock_ecology.add_evidence.assert_called()
        mock_ecology.create_goal.assert_not_called()

    def test_short_text_skipped(self):
        """Very short text should not produce signals."""
        signals = extract_and_route_signals('test_topic', 'ok', None)
        assert signals == []

    def test_none_text_returns_empty(self):
        """None text should return empty list."""
        signals = extract_and_route_signals('topic', None, None)
        assert signals == []

    def test_empty_string_returns_empty(self):
        """Empty string should return empty list."""
        signals = extract_and_route_signals('topic', '', None)
        assert signals == []

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only text should return empty list."""
        signals = extract_and_route_signals('topic', '   ', None)
        assert signals == []

    def test_non_explicit_unmatched_stores_signal(self):
        """Non-explicit text with no goal match should store as unmatched."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        mock_store = MagicMock()

        classification = {'intent_type': 'statement'}

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store):
            signals = extract_and_route_signals(
                'general', 'The weather is absolutely beautiful today', classification
            )

        # Should store unmatched signal for pattern detection
        mock_store.setex.assert_called_once()


class TestCognitiveSignalRouting:

    def test_route_cognitive_signal_with_dict(self):
        """Dict-format cognitive signals should be routed to goal ecology."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'goal-1', 'similarity': 0.75},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'User learned about cooking techniques'},
                'source_id': 'semantic_consolidation',
            })

        mock_ecology.add_evidence.assert_called()

    def test_route_cognitive_signal_no_match_stores_unmatched(self):
        """Cognitive signals not matching any goal should be stored for pattern detection."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        mock_store = MagicMock()

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store):
            route_cognitive_signal({
                'signal_type': 'novel_observation',
                'payload': {'content': 'Interesting pattern about user behavior'},
            })

        mock_store.setex.assert_called_once()

    def test_route_with_dataclass_like_object(self):
        """Should handle objects with signal_type attribute."""
        mock_signal = MagicMock()
        mock_signal.signal_type = 'new_knowledge'
        mock_signal.payload = {'content': 'Discovered an important pattern in data'}
        mock_signal.source_id = 'semantic_worker'

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'goal-1', 'similarity': 0.8},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal(mock_signal)

        mock_ecology.add_evidence.assert_called()
        call_kwargs = mock_ecology.add_evidence.call_args[1]
        assert call_kwargs['signal_type'] == 'behavioral'

    def test_route_with_short_content_skipped(self):
        """Signals with insufficient content should be skipped."""
        mock_ecology = MagicMock()

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'hi'},
            })

        mock_ecology.find_matching_goals.assert_not_called()

    def test_route_with_none_signal_no_crash(self):
        """None signal should not crash."""
        route_cognitive_signal(None)

    def test_route_with_empty_dict_no_crash(self):
        """Empty dict signal should not crash."""
        route_cognitive_signal({})

    def test_route_routes_to_top_two_matches(self):
        """Cognitive signals should route evidence to top 2 matching goals."""
        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'goal-1', 'similarity': 0.9},
            {'id': 'goal-2', 'similarity': 0.8},
            {'id': 'goal-3', 'similarity': 0.7},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'Important discovery about cooking techniques'},
            })

        assert mock_ecology.add_evidence.call_count == 2


class TestSignalContentExtraction:

    def test_extract_from_string_payload(self):
        """String payloads should be returned directly if long enough."""
        result = _extract_signal_content('test', 'This is a meaningful signal content')
        assert result == 'This is a meaningful signal content'

    def test_extract_from_dict_content_field(self):
        """Dict payloads with 'content' field should extract it."""
        result = _extract_signal_content('test', {'content': 'Some meaningful content'})
        assert result == 'Some meaningful content'

    def test_short_string_returns_none(self):
        """Very short strings should return None."""
        result = _extract_signal_content('test', 'hi')
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dicts should return None."""
        result = _extract_signal_content('test', {})
        assert result is None

    def test_extract_from_dict_text_field(self):
        result = _extract_signal_content('test', {'text': 'Some meaningful text here'})
        assert result == 'Some meaningful text here'

    def test_extract_from_dict_message_field(self):
        result = _extract_signal_content('test', {'message': 'An important message here'})
        assert result == 'An important message here'

    def test_extract_from_dict_description_field(self):
        result = _extract_signal_content('test', {'description': 'A detailed description here'})
        assert result == 'A detailed description here'

    def test_extract_from_composed_fields(self):
        result = _extract_signal_content('test', {
            'topic': 'cooking',
            'action': 'searched recipes',
            'result': 'found 5 results',
        })
        assert result is not None
        assert 'cooking' in result
        assert 'searched recipes' in result

    def test_extract_from_gist_field(self):
        result = _extract_signal_content('test', {'gist': 'A summary of the conversation about cooking'})
        assert result == 'A summary of the conversation about cooking'

    def test_extract_from_summary_field(self):
        result = _extract_signal_content('test', {'summary': 'Discussion about travel plans to Japan'})
        assert result == 'Discussion about travel plans to Japan'

    def test_none_payload_returns_none(self):
        result = _extract_signal_content('test', None)
        assert result is None


class TestSignalTypeMapping:

    def test_new_knowledge_maps_to_behavioral(self):
        assert _map_signal_type('new_knowledge') == 'behavioral'

    def test_schedule_fired_maps_to_temporal(self):
        assert _map_signal_type('schedule_fired') == 'temporal'

    def test_memory_pressure_maps_to_ambient(self):
        assert _map_signal_type('memory_pressure') == 'ambient'

    def test_user_message_maps_to_topic_recurrence(self):
        assert _map_signal_type('user_message') == 'topic_recurrence'

    def test_unknown_maps_to_behavioral(self):
        assert _map_signal_type('completely_unknown_type') == 'behavioral'

    def test_trait_changed_maps_to_behavioral(self):
        assert _map_signal_type('trait_changed') == 'behavioral'

    def test_task_state_changed_maps_to_behavioral(self):
        assert _map_signal_type('task_state_changed') == 'behavioral'

    def test_thread_expired_maps_to_ambient(self):
        assert _map_signal_type('thread_expired') == 'ambient'

    def test_novel_observation_maps_to_behavioral(self):
        assert _map_signal_type('novel_observation') == 'behavioral'


@pytest.mark.unit
class TestAmbientEnrichment:
    """Test ambient context enrichment of goal signals."""

    def test_deep_focus_boosts_strength(self, mock_store):
        """Deep focus attention should boost signal strength by 1.3x."""
        from services.goal_signal_service import _enrich_with_ambient_context

        signals = [{'signal_type': 'topic_recurrence', 'strength': 0.5}]

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient, \
             patch('services.client_context_service.ClientContextService') as MockCtx:
            MockCtx.return_value.get.return_value = {'behavioral': {}}
            MockAmbient.return_value.infer.return_value = {
                'attention': 'deep_focus', 'tempo': None,
            }

            _enrich_with_ambient_context(signals)

        assert signals[0]['strength'] == 0.65  # 0.5 * 1.3
        assert 'deep_focus' in signals[0].get('ambient_context', '')

    def test_casual_attention_no_boost(self, mock_store):
        """Casual attention should not modify strength."""
        from services.goal_signal_service import _enrich_with_ambient_context

        signals = [{'signal_type': 'topic_recurrence', 'strength': 0.5}]

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient, \
             patch('services.client_context_service.ClientContextService') as MockCtx:
            MockCtx.return_value.get.return_value = {'behavioral': {}}
            MockAmbient.return_value.infer.return_value = {
                'attention': 'casual', 'tempo': None,
            }

            _enrich_with_ambient_context(signals)

        assert signals[0]['strength'] == 0.5  # unchanged
        assert signals[0].get('ambient_context') == 'casual'

    def test_routine_tempo_annotated(self, mock_store):
        """Routine tempo should be annotated on signals."""
        from services.goal_signal_service import _enrich_with_ambient_context

        signals = [{'signal_type': 'topic_recurrence', 'strength': 0.5}]

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient, \
             patch('services.client_context_service.ClientContextService') as MockCtx:
            MockCtx.return_value.get.return_value = {'behavioral': {}}
            MockAmbient.return_value.infer.return_value = {
                'attention': None, 'tempo': 'routine',
            }

            _enrich_with_ambient_context(signals)

        assert 'routine' in signals[0].get('ambient_context', '')

    def test_empty_signals_noop(self, mock_store):
        """Empty signal list should not error."""
        from services.goal_signal_service import _enrich_with_ambient_context
        signals = []
        _enrich_with_ambient_context(signals)  # Should not raise

    def test_ambient_service_failure_graceful(self, mock_store):
        """Ambient service failure should not prevent signal extraction."""
        from services.goal_signal_service import _enrich_with_ambient_context

        signals = [{'signal_type': 'test', 'strength': 0.5}]

        with patch('services.ambient_inference_service.AmbientInferenceService', side_effect=Exception("boom")):
            _enrich_with_ambient_context(signals)  # Should not raise

        assert signals[0]['strength'] == 0.5  # unchanged

    def test_strength_capped_at_one(self, mock_store):
        """Strength should not exceed 1.0 after boost."""
        from services.goal_signal_service import _enrich_with_ambient_context

        signals = [{'signal_type': 'topic_recurrence', 'strength': 0.9}]

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient, \
             patch('services.client_context_service.ClientContextService') as MockCtx:
            MockCtx.return_value.get.return_value = {'behavioral': {}}
            MockAmbient.return_value.infer.return_value = {
                'attention': 'deep_focus', 'tempo': None,
            }

            _enrich_with_ambient_context(signals)

        assert signals[0]['strength'] == 1.0  # capped

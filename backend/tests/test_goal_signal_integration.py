"""
Integration tests for goal signal pipeline wiring.

Verifies that extract_and_route_signals and route_cognitive_signal are callable
with the exact signatures and data shapes used by digest_worker and
semantic_consolidation_worker.
"""

import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


class TestExtractAndRouteSignalsCallable:
    """Verify extract_and_route_signals can be called with digest_worker's shapes."""

    def test_called_with_topic_text_classification(self):
        """Function accepts (topic, text, classification) as digest_worker uses it."""
        from services.goal_signal_service import extract_and_route_signals

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        classification = {
            'topic': 'cooking',
            'confidence': 8,
            'similar_topic': '',
            'topic_update': '',
            'context_warmth': 0.5,
            'intent_type': 'command',
        }

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            result = extract_and_route_signals(
                'cooking',
                'Please help me plan dinner for the week',
                classification,
            )

        assert isinstance(result, list)

    def test_called_with_none_classification(self):
        """Function accepts None classification (default value)."""
        from services.goal_signal_service import extract_and_route_signals

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            result = extract_and_route_signals('general', 'Some longer message text here', None)

        assert isinstance(result, list)

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list without crashing."""
        from services.goal_signal_service import extract_and_route_signals

        result = extract_and_route_signals('topic', '', None)
        assert result == []

    def test_short_text_returns_empty_list(self):
        """Short text (under MIN_TEXT_LENGTH) should return empty list."""
        from services.goal_signal_service import extract_and_route_signals

        result = extract_and_route_signals('topic', 'ok', None)
        assert result == []

    def test_none_text_returns_empty_list(self):
        """None text should return empty list without crashing."""
        from services.goal_signal_service import extract_and_route_signals

        result = extract_and_route_signals('topic', None, None)
        assert result == []

    def test_intent_type_from_merged_classification(self):
        """Intent type in classification dict should trigger explicit_statement signal."""
        from services.goal_signal_service import extract_and_route_signals

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []
        mock_ecology.create_goal.return_value = {'id': 'g1', 'type': 'stated'}

        # Simulate digest_worker's merged classification dict
        merged_classification = {
            'topic': 'tasks',
            'confidence': 9,
            'intent_type': 'action',
        }

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            result = extract_and_route_signals(
                'tasks',
                'Set a reminder to call dentist tomorrow morning',
                merged_classification,
            )

        signal_types = [s['signal_type'] for s in result]
        assert 'explicit_statement' in signal_types


class TestRouteCognitiveSignalCallable:
    """Verify route_cognitive_signal can be called with semantic_consolidation_worker's shapes."""

    def test_called_with_new_knowledge_dict(self):
        """Function accepts dict with signal_type, payload, source_id."""
        from services.goal_signal_service import route_cognitive_signal

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = []
        mock_store = MagicMock()

        signal = {
            'signal_type': 'new_knowledge',
            'payload': {'content': 'User has deep interest in machine learning and AI research'},
            'source_id': 'semantic_consolidation:machine_learning',
        }

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store):
            route_cognitive_signal(signal)  # Should not raise

    def test_called_with_matching_goal_routes_evidence(self):
        """When a matching goal exists, evidence is added."""
        from services.goal_signal_service import route_cognitive_signal

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.return_value = [
            {'id': 'goal-1', 'similarity': 0.78},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'User regularly studies programming languages and tools'},
                'source_id': 'semantic_consolidation:programming',
            })

        mock_ecology.add_evidence.assert_called()

    def test_called_with_short_concept_content_skipped(self):
        """Concepts with very short content should be silently skipped."""
        from services.goal_signal_service import route_cognitive_signal

        mock_ecology = MagicMock()

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'AI'},
                'source_id': 'semantic_consolidation:ai',
            })

        mock_ecology.find_matching_goals.assert_not_called()


class TestFaultTolerance:
    """Verify both functions handle missing/broken goal tables gracefully."""

    def test_extract_handles_goal_ecology_import_error(self):
        """If GoalEcologyService can't be imported, extract returns without crashing."""
        from services.goal_signal_service import extract_and_route_signals

        with patch('services.goal_ecology_service.GoalEcologyService',
                   side_effect=ImportError("goal_ecology_service not available")):
            result = extract_and_route_signals(
                'test', 'Some longer test message for processing', None
            )

        # Should return a list (may be partial or empty), never raise
        assert isinstance(result, list)

    def test_extract_handles_database_error(self):
        """Database errors in goal ecology don't propagate to caller."""
        from services.goal_signal_service import extract_and_route_signals

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.side_effect = Exception("no such table: goals")

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            result = extract_and_route_signals(
                'test', 'Some longer test message for processing', None
            )

        # find_matching_goals failure is caught internally; result is still a list
        assert isinstance(result, list)

    def test_route_handles_goal_ecology_import_error(self):
        """If GoalEcologyService can't be imported, route returns without crashing."""
        from services.goal_signal_service import route_cognitive_signal

        with patch('services.goal_ecology_service.GoalEcologyService',
                   side_effect=ImportError("goal_ecology_service not available")):
            # Should not raise
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'User is interested in cooking healthy meals regularly'},
                'source_id': 'semantic_consolidation:cooking',
            })

    def test_route_handles_database_error_on_find(self):
        """Database errors in find_matching_goals don't propagate."""
        from services.goal_signal_service import route_cognitive_signal

        mock_ecology = MagicMock()
        mock_ecology.find_matching_goals.side_effect = Exception("no such table: goals")

        with patch('services.goal_ecology_service.GoalEcologyService', return_value=mock_ecology):
            # Should not raise (entire route_cognitive_signal is wrapped in try/except)
            route_cognitive_signal({
                'signal_type': 'new_knowledge',
                'payload': {'content': 'User is interested in cooking healthy meals regularly'},
                'source_id': 'semantic_consolidation:cooking',
            })

    def test_route_with_none_input_no_crash(self):
        """None input to route_cognitive_signal should not raise."""
        from services.goal_signal_service import route_cognitive_signal

        route_cognitive_signal(None)

    def test_route_with_empty_dict_no_crash(self):
        """Empty dict to route_cognitive_signal should not raise."""
        from services.goal_signal_service import route_cognitive_signal

        route_cognitive_signal({})

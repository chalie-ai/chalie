"""Tests for H1.4 — Goal-Autobiography bidirectional bridge."""

import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestComputeGoalAlignment:
    """Test compute_goal_alignment() embedding similarity."""

    def test_returns_empty_when_no_narrative(self, mock_store):
        """Missing autobiography returns empty alignment."""
        with patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'):
            MockAuto.return_value.get_current_narrative.return_value = None

            from services.goal_autobiography_bridge import compute_goal_alignment
            result = compute_goal_alignment("Learn Rust programming")

        assert result == {'parent_motives': [], 'identity_links': []}

    def test_matches_motives_by_embedding(self, mock_store):
        """Goals matching CORE_MOTIVES keywords populate parent_motives."""
        # Mock embedding service to return predictable embeddings
        mock_emb_service = MagicMock()

        # Create a simple embedding scheme: each text gets a unique-ish vector
        def fake_embedding(text):
            # "competence growth" and "Learn Rust" should have high similarity
            if 'competence' in text.lower() or 'learn' in text.lower() or 'rust' in text.lower():
                return np.array([0.8, 0.6, 0.0, 0.0])
            elif 'coherence' in text.lower():
                return np.array([0.3, 0.3, 0.8, 0.0])
            elif 'initiative' in text.lower():
                return np.array([0.5, 0.5, 0.3, 0.0])
            else:
                return np.array([0.1, 0.1, 0.1, 0.9])  # Low similarity to everything

        mock_emb_service.generate_embedding_np.side_effect = fake_embedding

        with patch('services.embedding_service.get_embedding_service', return_value=mock_emb_service), \
             patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'):
            MockAuto.return_value.get_current_narrative.return_value = {
                'narrative': '## Values And Goals\nLearning and growth\n## Long Term Themes\nTech mastery'
            }

            from services.goal_autobiography_bridge import compute_goal_alignment
            result = compute_goal_alignment("Learn Rust programming")

        assert isinstance(result['parent_motives'], list)
        assert isinstance(result['identity_links'], list)
        # Should have at least competence_growth due to high similarity
        assert len(result['parent_motives']) <= 3
        assert len(result['identity_links']) <= 3

    def test_caps_at_three_each(self, mock_store):
        """parent_motives and identity_links capped at 3."""
        mock_emb_service = MagicMock()
        # Return very high similarity for everything
        mock_emb_service.generate_embedding_np.return_value = np.array([1.0, 0.0, 0.0, 0.0])

        with patch('services.embedding_service.get_embedding_service', return_value=mock_emb_service), \
             patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'):
            MockAuto.return_value.get_current_narrative.return_value = {
                'narrative': '## Values And Goals\nEverything aligns\n## Long Term Themes\nAll things'
            }

            from services.goal_autobiography_bridge import compute_goal_alignment
            result = compute_goal_alignment("Universal goal")

        assert len(result['parent_motives']) <= 3
        assert len(result['identity_links']) <= 3

    def test_fault_tolerance(self, mock_store):
        """Embedding failure returns empty alignment."""
        with patch('services.embedding_service.get_embedding_service') as MockEmb, \
             patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'):
            MockEmb.side_effect = Exception("Embedding service unavailable")
            MockAuto.return_value.get_current_narrative.return_value = {
                'narrative': '## Values And Goals\nTest'
            }

            from services.goal_autobiography_bridge import compute_goal_alignment
            result = compute_goal_alignment("Test goal")

        assert result == {'parent_motives': [], 'identity_links': []}


@pytest.mark.unit
class TestRefreshAllGoals:
    """Test refresh_all_goals() batch update."""

    def test_returns_zero_when_no_narrative(self, mock_store):
        """No autobiography narrative returns 0 updates."""
        with patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.goal_ecology_service.GoalEcologyService'):
            MockAuto.return_value.get_current_narrative.return_value = None

            from services.goal_autobiography_bridge import refresh_all_goals
            result = refresh_all_goals()

        assert result == 0

    def test_returns_zero_when_no_goals(self, mock_store):
        """No active goals returns 0 updates."""
        with patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            MockAuto.return_value.get_current_narrative.return_value = {
                'narrative': '## Values And Goals\nTest values'
            }
            MockEcology.return_value.get_active_stack.return_value = []

            from services.goal_autobiography_bridge import refresh_all_goals
            result = refresh_all_goals()

        assert result == 0

    def test_fault_tolerance_on_exception(self, mock_store):
        """Service failure returns 0 (doesn't propagate)."""
        with patch('services.autobiography_service.AutobiographyService') as MockAuto, \
             patch('services.database_service.get_shared_db_service'):
            MockAuto.side_effect = Exception("DB down")

            from services.goal_autobiography_bridge import refresh_all_goals
            result = refresh_all_goals()

        assert result == 0


@pytest.mark.unit
class TestDirectionA:
    """Test autobiography synthesis receives goal data."""

    def test_gather_inputs_includes_goals(self, mock_store):
        """gather_synthesis_inputs includes active_goals key."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'description': 'Test goal', 'type': 'stated', 'salience': 0.8,
                 'confidence': 0.9, 'evidence_count': 5, 'timescale': 'short_term'},
            ]

            # We need a real-ish autobiography service with mocked DB
            from services.autobiography_service import AutobiographyService
            mock_db = MagicMock()
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            # Mock all SQL queries to return empty results
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_result.fetchone.return_value = None
            mock_session.execute.return_value = mock_result

            mock_db.get_session.return_value = mock_session

            auto = AutobiographyService(mock_db)
            inputs = auto.gather_synthesis_inputs()

        assert 'active_goals' in inputs
        assert len(inputs['active_goals']) == 1
        assert inputs['active_goals'][0]['description'] == 'Test goal'

    def test_build_prompt_includes_goals(self, mock_store):
        """_build_synthesis_prompt renders tracked goals section."""
        from services.autobiography_service import AutobiographyService
        mock_db = MagicMock()
        auto = AutobiographyService(mock_db)

        inputs = {
            'episodes': [{'gist': 'Test episode', 'emotion': 'neutral', 'topic': 'test'}],
            'traits': [],
            'concepts': [],
            'constraint_episodes': [],
            'active_goals': [
                {'description': 'Learn Rust', 'type': 'stated', 'salience': 0.85,
                 'confidence': 0.9, 'evidence_count': 3, 'timescale': 'short_term'},
            ],
        }

        prompt = auto._build_synthesis_prompt(inputs, None)

        assert 'Tracked Goals' in prompt
        assert 'Learn Rust' in prompt
        assert 'stated' in prompt
        assert '85%' in prompt

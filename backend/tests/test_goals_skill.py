"""Tests for goals innate skill."""

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestGoalsSkillList:
    """Test goals skill list action."""

    def test_list_returns_active_goals(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'type': 'stated', 'description': 'Plan dinner', 'salience': 0.85, 'confidence': 0.9, 'evidence_count': 3, 'timescale': 'short_term', 'status': 'actionable'},
                {'type': 'emergent', 'description': 'Learn Rust', 'salience': 0.6, 'confidence': 0.5, 'evidence_count': 7, 'timescale': 'medium_term', 'status': 'strengthening'},
            ]

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'list'})

        assert '2 active goal' in result
        assert 'Plan dinner' in result
        assert 'Learn Rust' in result
        assert 'stated' in result
        assert 'emergent' in result

    def test_list_empty(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = []

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'list'})

        assert 'No active goals' in result


@pytest.mark.unit
class TestGoalsSkillView:
    """Test goals skill view action."""

    def test_view_shows_detail(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.database_service.get_lightweight_db_service') as MockDb:
            svc = MockEcology.return_value
            svc.get_goal.return_value = {
                'id': 'g1', 'description': 'Plan dinner for 20', 'type': 'stated',
                'status': 'actionable', 'salience': 0.85, 'confidence': 0.9,
                'urgency': 0.7, 'timescale': 'short_term', 'evidence_count': 3,
                'strategy': 'Research catering options', 'created_at': '2026-03-22T10:00:00',
                'outcome_feedback': '[]',
            }
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('explicit_statement', 'I need to plan dinner', 'topic:food', '2026-03-22T10:00:00'),
            ]
            mock_conn.cursor.return_value = mock_cursor
            MockDb.return_value.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            MockDb.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'view', 'goal_id': 'g1'})

        assert 'Plan dinner for 20' in result
        assert 'stated' in result
        assert 'Strategy:' in result
        assert 'I need to plan dinner' in result

    def test_view_missing_goal_id(self, mock_store):
        from services.innate_skills.goals_skill import handle_goals
        result = handle_goals('test-topic', {'action': 'view'})
        assert 'goal_id required' in result

    def test_view_not_found(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.return_value = None

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'view', 'goal_id': 'xxx'})

        assert 'not found' in result


@pytest.mark.unit
class TestGoalsSkillConfirm:
    """Test goals skill confirm action."""

    def test_confirm_upgrades_to_stated(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.confirm_goal.return_value = {
                'id': 'g1', 'description': 'Learn Rust', 'type': 'stated',
                'confidence': 0.85, 'salience': 0.9,
            }

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'confirm', 'goal_id': 'g1'})

        assert 'confirmed' in result.lower()
        assert 'Learn Rust' in result
        svc.confirm_goal.assert_called_once_with('g1')


@pytest.mark.unit
class TestGoalsSkillComplete:
    """Test goals skill complete action."""

    def test_complete_success(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.complete_goal.return_value = True

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'complete', 'goal_id': 'g1'})

        assert 'completed' in result.lower()

    def test_complete_failure(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.complete_goal.return_value = False

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'complete', 'goal_id': 'g1'})

        assert 'not found' in result.lower() or 'already' in result.lower()


@pytest.mark.unit
class TestGoalsSkillDismiss:
    """Test goals skill dismiss action."""

    def test_dismiss_success(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.dismiss_goal.return_value = True

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'dismiss', 'goal_id': 'g1'})

        assert 'dismissed' in result.lower()


@pytest.mark.unit
class TestGoalsSkillAdjust:
    """Test goals skill adjust action."""

    def test_adjust_urgency(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.update_goal.return_value = True

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'adjust', 'goal_id': 'g1', 'urgency': 0.9})

        assert 'adjusted' in result.lower()
        svc.update_goal.assert_called_once()

    def test_adjust_invalid_timescale(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'adjust', 'goal_id': 'g1', 'timescale': 'invalid'})

        assert 'Invalid timescale' in result

    def test_unknown_action(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'bogus'})

        assert 'Unknown action' in result


@pytest.mark.unit
class TestGoalsSkillMute:
    """Test goals skill mute and unmute actions."""

    def test_mute_success(self, mock_store):
        """mute action should confirm the goal was muted."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.mute_goal.return_value = True

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'mute', 'goal_id': 'g1'})

        assert 'muted' in result.lower()
        svc.mute_goal.assert_called_once_with('g1')

    def test_mute_not_found(self, mock_store):
        """mute action should report when the goal is not found."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.mute_goal.return_value = False

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'mute', 'goal_id': 'missing'})

        assert 'not found' in result.lower()

    def test_mute_missing_goal_id(self, mock_store):
        """mute action without goal_id should return an error."""
        from services.innate_skills.goals_skill import handle_goals
        result = handle_goals('test-topic', {'action': 'mute'})
        assert 'goal_id required' in result

    def test_unmute_success(self, mock_store):
        """unmute action should confirm the goal was unmuted."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.unmute_goal.return_value = True

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'unmute', 'goal_id': 'g1'})

        assert 'unmuted' in result.lower()
        svc.unmute_goal.assert_called_once_with('g1')

    def test_unmute_not_found(self, mock_store):
        """unmute action should report when the goal is not found."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.unmute_goal.return_value = False

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'unmute', 'goal_id': 'missing'})

        assert 'not found' in result.lower()

    def test_unmute_missing_goal_id(self, mock_store):
        """unmute action without goal_id should return an error."""
        from services.innate_skills.goals_skill import handle_goals
        result = handle_goals('test-topic', {'action': 'unmute'})
        assert 'goal_id required' in result

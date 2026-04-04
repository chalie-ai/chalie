"""Tests for goals innate skill."""

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

    def test_view_shows_detail(self, db, mock_store):
        """View action should display goal details and evidence from real DB."""
        # Seed a goal row so FK on goal_evidence is satisfied
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence, urgency, "
            "timescale, evidence_count, strategy, created_at, outcome_feedback) "
            "VALUES ('g1', 'Plan dinner for 20', 'stated', 'actionable', 0.85, 0.9, 0.7, "
            "'short_term', 3, 'Research catering options', '2026-03-22T10:00:00', '[]')"
        )
        db.execute(
            "INSERT INTO goal_evidence (goal_id, signal_type, content, source, created_at) "
            "VALUES ('g1', 'explicit_statement', 'I need to plan dinner', 'topic:food', '2026-03-22T10:00:00')"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.return_value = {
                'id': 'g1', 'description': 'Plan dinner for 20', 'type': 'stated',
                'status': 'actionable', 'salience': 0.85, 'confidence': 0.9,
                'urgency': 0.7, 'timescale': 'short_term', 'evidence_count': 3,
                'strategy': 'Research catering options', 'created_at': '2026-03-22T10:00:00',
                'outcome_feedback': '[]',
            }
            svc.get_children.return_value = []

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
        with patch('services.goal_ecology_service.GoalEcologyService'):
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'adjust', 'goal_id': 'g1', 'timescale': 'invalid'})

        assert 'Invalid timescale' in result

    def test_unknown_action(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService'):
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


@pytest.mark.unit
class TestGoalsSkillViewLineage:
    """Test goals skill view action shows lineage information."""

    def test_view_shows_parent_goal_when_lineage_exists(self, db, mock_store):
        """View should display parent goal info when lineage_parent_id is set."""
        # Seed goal rows for the evidence FK
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Plan dinner for 20', 'stated', 'actionable', 0.85, 0.9)"
        )
        db.commit()

        parent_goal = {
            'id': 'parent-1', 'type': 'developmental',
            'description': 'Build reputation as exceptional host',
            'salience': 0.8, 'confidence': 0.7,
        }
        child_goal = {
            'id': 'g1', 'description': 'Plan dinner for 20', 'type': 'stated',
            'status': 'actionable', 'salience': 0.85, 'confidence': 0.9,
            'urgency': 0.7, 'timescale': 'short_term', 'evidence_count': 3,
            'strategy': None, 'created_at': '2026-03-22T10:00:00',
            'outcome_feedback': '[]',
            'lineage_parent_id': 'parent-1',
        }

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.side_effect = lambda gid: (
                child_goal if gid == 'g1'
                else parent_goal if gid == 'parent-1'
                else None
            )
            svc.get_children.return_value = []

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'view', 'goal_id': 'g1'})

        assert 'Parent Goal' in result
        assert 'Build reputation as exceptional host' in result
        assert 'developmental' in result

    def test_view_shows_child_goals_when_children_exist(self, db, mock_store):
        """View should display child goals when get_children returns results."""
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Long-term vision', 'developmental', 'actionable', 0.8, 0.7)"
        )
        db.commit()

        goal_data = {
            'id': 'g1', 'description': 'Long-term vision', 'type': 'developmental',
            'status': 'actionable', 'salience': 0.8, 'confidence': 0.7,
            'urgency': 0.5, 'timescale': 'long_term', 'evidence_count': 5,
            'strategy': None, 'created_at': '2026-03-22T10:00:00',
            'outcome_feedback': '[]',
            'lineage_parent_id': None,
        }
        children = [
            {'id': 'c1', 'type': 'stated', 'description': 'Plan dinner for 20',
             'status': 'actionable', 'salience': 0.85, 'confidence': 0.9},
            {'id': 'c2', 'type': 'emergent', 'description': 'Host a workshop',
             'status': 'strengthening', 'salience': 0.5, 'confidence': 0.4},
        ]

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.return_value = goal_data
            svc.get_children.return_value = children

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'view', 'goal_id': 'g1'})

        assert 'Child Goals' in result
        assert 'Plan dinner for 20' in result
        assert 'Host a workshop' in result

    def test_view_no_lineage_shows_no_parent_section(self, db, mock_store):
        """View without lineage should not include Parent Goal section."""
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Standalone goal', 'stated', 'actionable', 0.85, 0.9)"
        )
        db.commit()

        goal_data = {
            'id': 'g1', 'description': 'Standalone goal', 'type': 'stated',
            'status': 'actionable', 'salience': 0.85, 'confidence': 0.9,
            'urgency': 0.5, 'timescale': 'short_term', 'evidence_count': 2,
            'strategy': None, 'created_at': '2026-03-22T10:00:00',
            'outcome_feedback': '[]',
            'lineage_parent_id': None,
        }

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.return_value = goal_data
            svc.get_children.return_value = []

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'view', 'goal_id': 'g1'})

        assert 'Parent Goal' not in result
        assert 'Child Goals' not in result


@pytest.mark.unit
class TestGoalsSkillNarrate:
    """Test goals skill narrate action."""

    def test_narrate_no_goals(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = []

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'narrate'})

        assert 'No active goals' in result

    def test_narrate_calls_llm_with_goal_context(self, db, mock_store):
        # Seed goal and evidence in real DB
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Plan dinner for 20', 'stated', 'actionable', 0.85, 0.9)"
        )
        db.execute(
            "INSERT INTO goal_evidence (goal_id, signal_type, content, source, created_at) "
            "VALUES ('g1', 'explicit_statement', 'User said they need to plan dinner', "
            "'conversation', '2026-03-20')"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'id': 'g1', 'type': 'stated', 'description': 'Plan dinner for 20',
                 'salience': 0.85, 'confidence': 0.9, 'evidence_count': 5,
                 'timescale': 'short_term', 'status': 'actionable',
                 'strategy': 'Research menus and venues', 'created_at': '2026-03-20',
                 'lineage_parent_id': None, 'outcome_feedback': []},
            ]
            svc.get_children.return_value = []

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.text = "You've been focused on planning a dinner for 20 people..."
            MockProxy.return_value.send_message.return_value = mock_response

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'narrate'})

        assert 'GOALS NARRATIVE' in result
        assert 'dinner' in result.lower()
        MockProxy.return_value.send_message.assert_called_once()

    def test_narrate_with_hierarchy(self, db, mock_store):
        # Seed goal in real DB (no evidence needed for this test)
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Plan dinner for 20', 'stated', 'actionable', 0.85, 0.9)"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'id': 'g1', 'type': 'stated', 'description': 'Plan dinner for 20',
                 'salience': 0.85, 'confidence': 0.9, 'evidence_count': 5,
                 'timescale': 'short_term', 'status': 'actionable',
                 'strategy': None, 'created_at': '2026-03-20',
                 'lineage_parent_id': 'g2', 'outcome_feedback': []},
            ]
            # Parent goal
            svc.get_goal.return_value = {
                'id': 'g2', 'description': 'Build reputation as exceptional host',
                'type': 'developmental', 'status': 'strengthening',
                'salience': 0.6, 'confidence': 0.65,
            }
            svc.get_children.return_value = []

            mock_response = MagicMock()
            mock_response.text = "Your dinner planning connects to a broader aspiration..."
            MockProxy.return_value.send_message.return_value = mock_response

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'narrate'})

        assert 'GOALS NARRATIVE' in result
        # Verify parent goal description was passed to LLM
        call_args = MockProxy.return_value.send_message.call_args
        assert 'Build reputation' in call_args[0][1]  # user_message

    def test_narrate_llm_failure_returns_fallback(self, db, mock_store):
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Test goal', 'stated', 'strengthening', 0.5, 0.5)"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'id': 'g1', 'type': 'stated', 'description': 'Test goal',
                 'salience': 0.5, 'confidence': 0.5, 'evidence_count': 1,
                 'timescale': 'short_term', 'status': 'strengthening',
                 'strategy': None, 'created_at': '2026-03-20',
                 'lineage_parent_id': None, 'outcome_feedback': []},
            ]
            svc.get_children.return_value = []

            MockProxy.return_value.send_message.side_effect = Exception("LLM unavailable")

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'narrate'})

        assert 'Unable to generate narrative' in result

    def test_narrate_truncates_long_response(self, db, mock_store):
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Test goal', 'stated', 'strengthening', 0.5, 0.5)"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'id': 'g1', 'type': 'stated', 'description': 'Test goal',
                 'salience': 0.5, 'confidence': 0.5, 'evidence_count': 1,
                 'timescale': 'short_term', 'status': 'strengthening',
                 'strategy': None, 'created_at': '2026-03-20',
                 'lineage_parent_id': None, 'outcome_feedback': []},
            ]
            svc.get_children.return_value = []

            mock_response = MagicMock()
            mock_response.text = "x" * 3000
            MockProxy.return_value.send_message.return_value = mock_response

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {'action': 'narrate'})

        assert 'GOALS NARRATIVE' in result
        # Should be truncated to ~2000 + prefix
        assert len(result) < 2100

    def test_narrate_with_feedback_history(self, db, mock_store):
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES ('g1', 'Learn Rust', 'stated', 'actionable', 0.7, 0.8)"
        )
        db.commit()

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            svc = MockEcology.return_value
            svc.get_active_stack.return_value = [
                {'id': 'g1', 'type': 'stated', 'description': 'Learn Rust',
                 'salience': 0.7, 'confidence': 0.8, 'evidence_count': 10,
                 'timescale': 'medium_term', 'status': 'actionable',
                 'strategy': 'Start with the Rust book', 'created_at': '2026-03-15',
                 'lineage_parent_id': None,
                 'outcome_feedback': [
                     {'response': 'engaged', 'timestamp': '2026-03-18'},
                     {'response': 'engaged', 'timestamp': '2026-03-19'},
                     {'response': 'rejected', 'timestamp': '2026-03-20'},
                 ]},
            ]
            svc.get_children.return_value = []

            mock_response = MagicMock()
            mock_response.text = "Your journey with Rust has been interesting..."
            MockProxy.return_value.send_message.return_value = mock_response

            from services.innate_skills.goals_skill import handle_goals
            handle_goals('test-topic', {'action': 'narrate'})

        # Verify feedback was passed to LLM
        call_args = MockProxy.return_value.send_message.call_args
        assert '2 engaged' in call_args[0][1]
        assert '1 rejected' in call_args[0][1]


@pytest.mark.unit
class TestGoalsSkillClusters:
    """Test goals skill cluster_confirm and cluster_dismiss actions."""

    def test_cluster_confirm_creates_parent_and_links(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.get_goal.return_value = {
                'id': 'g1', 'type': 'stated', 'timescale': 'short_term',
                'description': 'Learn pasta', 'status': 'actionable',
            }
            svc.TIMESCALE_ORDER = {
                'immediate': 0, 'short_term': 1,
                'medium_term': 2, 'long_term': 3,
            }
            svc.create_goal.return_value = {
                'id': 'parent1', 'description': 'Master Italian cooking',
                'type': 'stated', 'status': 'actionable',
            }
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 1
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            svc.db.connection.return_value = mock_conn

            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {
                'action': 'cluster_confirm',
                'goal_ids': ['g1', 'g2', 'g3'],
                'description': 'Master Italian cooking',
            })

        assert 'Cluster confirmed' in result
        assert 'Master Italian cooking' in result
        svc.create_goal.assert_called_once()

    def test_cluster_confirm_missing_ids(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService'):
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {
                'action': 'cluster_confirm',
                'goal_ids': [],
                'description': 'Test',
            })
        assert 'requires' in result.lower()

    def test_cluster_confirm_missing_description(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService'):
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {
                'action': 'cluster_confirm',
                'goal_ids': ['g1', 'g2', 'g3'],
                'description': '',
            })
        assert 'requires' in result.lower()

    def test_cluster_dismiss_sets_cooldown(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService'):
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {
                'action': 'cluster_dismiss',
                'goal_ids': ['g1', 'g2', 'g3'],
            })

        assert 'dismissed' in result.lower()
        # Check cooldown was set
        key = "goal_cluster:g1|g2|g3:cooldown"
        assert mock_store.get(key) is not None

    def test_cluster_dismiss_missing_ids(self, mock_store):
        with patch('services.goal_ecology_service.GoalEcologyService'):
            from services.innate_skills.goals_skill import handle_goals
            result = handle_goals('test-topic', {
                'action': 'cluster_dismiss',
                'goal_ids': [],
            })
        assert 'requires' in result.lower()

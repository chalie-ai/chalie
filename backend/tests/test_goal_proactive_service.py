"""Unit tests for GoalProactiveService -- trigger engine for actionable goals."""

import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

from services.memory_store import MemoryStore
from services.goal_proactive_service import (
    check_and_execute,
    _calculate_social_cost,
    _choose_output_style,
    _execute_proactive,
    _execute_via_persistent_task,
    _execute_via_proactive_push,
    _learn_quiet_hours,
    MIN_CONFIDENCE_FOR_ACTION,
    MAX_SOCIAL_COST,
    ASK_THRESHOLD,
    SUGGEST_THRESHOLD,
)


pytestmark = pytest.mark.unit


class TestOutputStyleSelection:

    def test_low_confidence_returns_ask(self):
        """Confidence below ASK_THRESHOLD should return 'ask' style."""
        assert _choose_output_style(0.60) == 'ask'
        assert _choose_output_style(0.65) == 'ask'
        assert _choose_output_style(0.69) == 'ask'

    def test_medium_confidence_returns_suggest(self):
        """Confidence between ASK and SUGGEST thresholds should return 'suggest'."""
        assert _choose_output_style(0.70) == 'suggest'
        assert _choose_output_style(0.75) == 'suggest'
        assert _choose_output_style(0.84) == 'suggest'

    def test_high_confidence_returns_act(self):
        """Confidence at or above SUGGEST_THRESHOLD should return 'act'."""
        assert _choose_output_style(0.85) == 'act'
        assert _choose_output_style(0.90) == 'act'
        assert _choose_output_style(1.0) == 'act'

    def test_thresholds_are_correct(self):
        """Verify the threshold constants."""
        assert ASK_THRESHOLD == 0.7
        assert SUGGEST_THRESHOLD == 0.85


class TestConfidenceGating:

    def test_below_min_confidence_returns_none(self):
        """Goals below MIN_CONFIDENCE_FOR_ACTION should not be executed."""
        goal = {
            'id': 'test-id',
            'description': 'Low confidence goal',
            'confidence': MIN_CONFIDENCE_FOR_ACTION - 0.1,
            'salience': 0.8,
            'strategy': 'do something',
        }
        result = check_and_execute(goal)
        assert result is None

    def test_at_min_confidence_proceeds(self):
        """Goals at exactly MIN_CONFIDENCE_FOR_ACTION should proceed (if social cost OK)."""
        goal = {
            'id': 'test-id',
            'description': 'Borderline confidence goal',
            'confidence': MIN_CONFIDENCE_FOR_ACTION,
            'salience': 0.8,
            'strategy': 'do something',
        }

        with patch('services.goal_proactive_service._calculate_social_cost', return_value=0.0), \
             patch('services.goal_proactive_service._execute_proactive') as mock_exec, \
             patch('services.goal_proactive_service._mark_acted'):
            mock_exec.return_value = {'action': 'proactive_push', 'style': 'ask'}
            result = check_and_execute(goal)

        assert result is not None

    def test_min_confidence_constant(self):
        """Verify MIN_CONFIDENCE_FOR_ACTION is 0.6."""
        assert MIN_CONFIDENCE_FOR_ACTION == 0.6


class TestSocialCostBlocking:

    def test_deep_focus_increases_cost(self):
        """Deep focus should add 0.3 to social cost."""
        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = True

        # Empty store — no ignored count, no message timestamps
        store = MemoryStore()

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost >= 0.3

    def test_no_deep_focus_low_cost(self):
        """Without deep focus and no ignored attempts, cost should be low."""
        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = False

        # Empty store — no ignored count, no message timestamps
        store = MemoryStore()

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost < MAX_SOCIAL_COST

    def test_recent_ignored_increases_cost(self):
        """Two or more recent ignored attempts should add 0.3 to cost."""
        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = False

        # Pre-populate the key _calculate_social_cost reads for ignored count
        store = MemoryStore()
        store.set('goal:recent_ignored_count', '3')

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost >= 0.3

    def test_deep_focus_plus_ignored_blocks(self):
        """Deep focus + recent ignored should exceed MAX_SOCIAL_COST."""
        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = True

        # Pre-populate the key _calculate_social_cost reads for ignored count
        store = MemoryStore()
        store.set('goal:recent_ignored_count', '2')

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost >= MAX_SOCIAL_COST

    def test_high_social_cost_blocks_execution(self):
        """Goals should not execute when social cost exceeds threshold."""
        goal = {
            'id': 'test-id',
            'description': 'Blocked by social cost',
            'confidence': 0.9,
            'salience': 0.8,
            'strategy': 'do something',
        }

        with patch('services.goal_proactive_service._calculate_social_cost',
                   return_value=MAX_SOCIAL_COST + 0.1):
            result = check_and_execute(goal)

        assert result is None

    def test_cost_capped_at_one(self):
        """Social cost should be capped at 1.0."""
        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = True

        # Pre-populate an extreme ignored count to drive cost above 1.0 before cap
        store = MemoryStore()
        store.set('goal:recent_ignored_count', '10')

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost <= 1.0

    def test_social_cost_active_conversation_penalty(self):
        """Active conversation (< 2 min ago) should increase social cost."""
        from services.time_utils import utc_now
        from services.memory_store import MemoryStore

        store = MemoryStore()
        store.set('last_user_message_ts', utc_now().isoformat())

        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = False

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        assert cost >= 0.2  # Active conversation penalty

    def test_social_cost_reentry_bonus(self):
        """Re-entry after 4 hour gap should reduce social cost."""
        from services.time_utils import utc_now
        from services.memory_store import MemoryStore
        from datetime import timedelta

        store = MemoryStore()
        four_hours_ago = (utc_now() - timedelta(hours=4)).isoformat()
        store.set('last_user_message_ts', four_hours_ago)

        mock_inference = MagicMock()
        mock_inference.is_user_deep_focus.return_value = False

        with patch('services.ambient_inference_service.AmbientInferenceService',
                   return_value=mock_inference), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            cost = _calculate_social_cost()

        # Should be negative (bonus) or at least lower than baseline
        assert cost <= 0.0  # Re-entry bonus should make cost negative or zero


class TestProactiveExecution:

    def test_ask_style_result_fields(self):
        """Ask-style result should have correct action, style, and goal_id."""
        goal = {
            'id': 'test-id',
            'description': 'Plan dinner for party',
            'strategy': 'research recipes',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            result = _execute_proactive(goal, 'ask')

        assert result['style'] == 'ask'
        assert result['goal_id'] == 'test-id'
        assert result['action'] == 'proactive_push'

    def test_suggest_style_result_fields(self):
        """Suggest-style result should have correct action, style, and goal_id."""
        goal = {
            'id': 'test-id',
            'description': 'Learn cooking techniques',
            'strategy': 'find tutorials',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            result = _execute_proactive(goal, 'suggest')

        assert result['style'] == 'suggest'
        assert result['action'] == 'proactive_push'

    def test_ask_prompt_contains_goal_metadata(self):
        """Ask-style prompt should include description, evidence count, goal type, and confidence."""
        goal = {
            'id': 'ask-meta-id',
            'description': 'Learn woodworking',
            'evidence_count': 5,
            'type': 'emergent',
            'confidence': 0.65,
            'strategy': 'watch tutorials',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'ask')

        # Verify state: exactly one item was pushed to the prompt-queue
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1
        pushed_payload = json.loads(items[0])
        prompt = pushed_payload['prompt']

        assert 'Learn woodworking' in prompt
        assert '5' in prompt           # evidence_count
        assert 'emergent' in prompt    # goal type
        assert '65%' in prompt         # confidence formatted as percentage

    def test_suggest_prompt_contains_strategy(self):
        """Suggest-style prompt should include strategy in addition to goal metadata."""
        goal = {
            'id': 'suggest-meta-id',
            'description': 'Get fit before summer',
            'evidence_count': 8,
            'type': 'stated',
            'confidence': 0.78,
            'strategy': 'run three times a week',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'suggest')

        # Verify state: inspect the item pushed to the prompt-queue
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1
        pushed_payload = json.loads(items[0])
        prompt = pushed_payload['prompt']

        assert 'Get fit before summer' in prompt
        assert 'run three times a week' in prompt  # strategy present
        assert '8' in prompt                        # evidence_count
        assert 'stated' in prompt                   # goal type
        assert '78%' in prompt                      # confidence

    def test_prompt_queue_payload_metadata_structure(self):
        """Prompt-queue payload must have type='proactive' and goal_id."""
        goal = {
            'id': 'meta-check-id',
            'description': 'Track fitness progress',
            'confidence': 0.72,
            'strategy': 'log workouts daily',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'suggest')

        # Verify state: inspect the metadata of the pushed payload
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1
        pushed_payload = json.loads(items[0])
        metadata = pushed_payload['metadata']

        assert metadata['type'] == 'proactive'
        assert metadata['goal_id'] == 'meta-check-id'
        assert metadata['source'] == 'goal_ecology'
        assert metadata['style'] == 'suggest'
        assert metadata['topic'] == 'proactive'

    def test_no_hardcoded_template_strings_in_ask(self):
        """Old corporate template strings must NOT appear in ask-style prompt."""
        goal = {
            'id': 'template-check-id',
            'description': 'Improve sleep schedule',
            'confidence': 0.63,
            'strategy': 'consistent bedtime',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'ask')

        # Verify state: inspect the prompt text for forbidden template strings
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1
        pushed_payload = json.loads(items[0])
        prompt = pushed_payload['prompt']

        assert "I've noticed a pattern in our conversations" not in prompt
        assert "Would it help if I looked into this?" not in prompt

    def test_no_hardcoded_template_strings_in_suggest(self):
        """Old corporate template strings must NOT appear in suggest-style prompt."""
        goal = {
            'id': 'template-check-suggest-id',
            'description': 'Read more books',
            'confidence': 0.75,
            'strategy': 'dedicate 30 min before bed',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'suggest')

        # Verify state: inspect the prompt text for forbidden template strings
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1
        pushed_payload = json.loads(items[0])
        prompt = pushed_payload['prompt']

        assert "Based on what you've been working on, I think this could be useful:" not in prompt

    def test_act_style_creates_persistent_task(self, db):
        """Act-style should create a persistent task, not just push a message."""
        # Seed master_account so the account_id resolution works
        db.execute(
            "INSERT INTO master_account (id, username, password_hash) VALUES (1, 'testuser', 'hash')"
        )
        db.commit()

        goal = {
            'id': 'test-goal-id',
            'description': 'Research ML papers',
            'strategy': 'search arxiv for recent publications',
        }

        mock_task_service = MagicMock()
        mock_task_service.find_duplicate.return_value = None
        mock_task_service.create_task.return_value = {'id': 42}
        mock_task_service.transition.return_value = (True, 'ok')

        with patch('services.persistent_task_service.PersistentTaskService', return_value=mock_task_service):
            result = _execute_proactive(goal, 'act')

        assert result['action'] == 'persistent_task_created'
        assert result['style'] == 'act'
        assert result['task_id'] == 42
        mock_task_service.create_task.assert_called_once()
        mock_task_service.transition.assert_called_once_with(42, 'accepted')

    def test_act_style_includes_strategy_in_task_goal(self, db):
        """When strategy exists, the task goal should include it."""
        db.execute(
            "INSERT INTO master_account (id, username, password_hash) VALUES (1, 'testuser', 'hash')"
        )
        db.commit()

        goal = {
            'id': 'test-id',
            'description': 'Monitor tech news',
            'strategy': 'search for Python and AI updates weekly',
        }

        mock_task_service = MagicMock()
        mock_task_service.find_duplicate.return_value = None
        mock_task_service.create_task.return_value = {'id': 99}
        mock_task_service.transition.return_value = (True, 'ok')

        with patch('services.persistent_task_service.PersistentTaskService', return_value=mock_task_service):
            _execute_via_persistent_task(goal)

        # Verify the task goal includes both description and strategy
        call_kwargs = mock_task_service.create_task.call_args
        task_goal = call_kwargs[1].get('goal') or call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1]['goal']
        assert 'Monitor tech news' in task_goal
        assert 'search for Python and AI updates weekly' in task_goal

    def test_act_style_deduplicates_tasks(self, db):
        """If a task already exists for this goal, don't create another."""
        db.execute(
            "INSERT INTO master_account (id, username, password_hash) VALUES (1, 'testuser', 'hash')"
        )
        db.commit()

        goal = {
            'id': 'test-id',
            'description': 'Already tracked goal',
            'strategy': 'existing approach',
        }

        mock_task_service = MagicMock()
        mock_task_service.find_duplicate.return_value = {'id': 77, 'goal': 'Already tracked goal'}

        with patch('services.persistent_task_service.PersistentTaskService', return_value=mock_task_service):
            result = _execute_via_persistent_task(goal)

        assert result['action'] == 'task_exists'
        assert result['task_id'] == 77
        mock_task_service.create_task.assert_not_called()

    def test_act_style_fallback_on_failure(self):
        """If persistent task creation fails, fall back to proactive push."""
        goal = {
            'id': 'test-id',
            'description': 'Fallback goal',
            'strategy': 'should fallback',
        }

        # Real store receives the fallback proactive-push payload
        store = MemoryStore()

        with patch('services.database_service.get_shared_db_service',
                   side_effect=Exception("DB unavailable")), \
             patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            result = _execute_via_persistent_task(goal)

        # Should fall back to proactive push with 'suggest' style
        assert result['action'] == 'proactive_push'
        assert result['style'] == 'suggest'

    def test_ask_pushes_to_prompt_queue(self):
        """Ask-style should push exactly one message to the prompt-queue."""
        goal = {
            'id': 'test-id',
            'description': 'Test goal',
            'strategy': 'test strategy',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'ask')

        # Verify state: one item enqueued in the prompt-queue
        items = store.lrange('prompt-queue', 0, -1)
        assert len(items) == 1

    def test_suggest_pushes_to_prompt_queue(self):
        """Suggest-style should push exactly one message to the prompt-queue."""
        goal = {
            'id': 'test-id',
            'description': 'Test goal',
            'strategy': 'test strategy',
        }

        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection',
                   return_value=store):
            _execute_via_proactive_push(goal, 'suggest')

        # Verify state: one item enqueued in the prompt-queue
        assert len(store.lrange('prompt-queue', 0, -1)) == 1


class TestCheckAndExecuteIntegration:

    def test_successful_execution_records_outcome(self):
        """Successful execution should call _mark_acted."""
        goal = {
            'id': 'test-id',
            'description': 'A good goal',
            'confidence': 0.8,
            'salience': 0.7,
            'strategy': 'do something',
        }

        with patch('services.goal_proactive_service._calculate_social_cost', return_value=0.0), \
             patch('services.goal_proactive_service._execute_proactive') as mock_exec, \
             patch('services.goal_proactive_service._mark_acted') as mock_mark:
            mock_exec.return_value = {'action': 'proactive_push', 'style': 'suggest'}
            result = check_and_execute(goal)

        mock_mark.assert_called_once_with('test-id')
        assert result is not None
        assert result['style'] == 'suggest'

    def test_execution_failure_returns_none(self):
        """If _execute_proactive raises, check_and_execute should return None."""
        goal = {
            'id': 'test-id',
            'description': 'Failing goal',
            'confidence': 0.9,
            'salience': 0.8,
            'strategy': 'do something',
        }

        with patch('services.goal_proactive_service._calculate_social_cost', return_value=0.0), \
             patch('services.goal_proactive_service._execute_proactive',
                   side_effect=Exception("Network error")):
            result = check_and_execute(goal)

        assert result is None

    def test_missing_goal_id_uses_unknown(self):
        """Goal dict without 'id' should use 'unknown' as fallback."""
        goal = {
            'description': 'No ID goal',
            'confidence': 0.3,  # Below threshold, will short-circuit
            'salience': 0.5,
        }
        result = check_and_execute(goal)
        assert result is None  # Should not crash


class TestMaxSocialCost:

    def test_max_social_cost_constant(self):
        """Verify MAX_SOCIAL_COST is 0.6."""
        assert MAX_SOCIAL_COST == 0.6

    def test_social_cost_at_exactly_max_allows_execution(self):
        """Social cost at exactly MAX should still block (> comparison)."""
        goal = {
            'id': 'test-id',
            'description': 'Borderline goal',
            'confidence': 0.9,
            'salience': 0.8,
            'strategy': 'do something',
        }

        # Cost at exactly MAX_SOCIAL_COST should NOT block (> not >=)
        with patch('services.goal_proactive_service._calculate_social_cost',
                   return_value=MAX_SOCIAL_COST), \
             patch('services.goal_proactive_service._execute_proactive') as mock_exec, \
             patch('services.goal_proactive_service._mark_acted'):
            mock_exec.return_value = {'action': 'proactive_push', 'style': 'act'}
            result = check_and_execute(goal)

        assert result is not None

    def test_social_cost_just_above_max_blocks(self):
        """Social cost just above MAX should block execution."""
        goal = {
            'id': 'test-id',
            'description': 'Blocked goal',
            'confidence': 0.9,
            'salience': 0.8,
            'strategy': 'do something',
        }

        with patch('services.goal_proactive_service._calculate_social_cost',
                   return_value=MAX_SOCIAL_COST + 0.01):
            result = check_and_execute(goal)

        assert result is None


class TestQuietHoursAndConversationPace:

    def test_social_cost_quiet_hours_penalty(self, mock_store):
        """Quiet hours should add 0.4 to social cost."""
        from services.time_utils import utc_now
        # Set current hour as quiet
        current_hour = utc_now().hour
        mock_store.set('goal:quiet_hours', json.dumps([current_hour]))

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient:
            MockAmbient.return_value.is_user_deep_focus.return_value = False

            cost = _calculate_social_cost()

        assert cost >= 0.4  # Quiet hours penalty

    def test_social_cost_non_quiet_hours_no_penalty(self, mock_store):
        """Non-quiet hours should not add penalty."""
        from services.time_utils import utc_now
        current_hour = utc_now().hour
        # Set a different hour as quiet
        quiet_hour = (current_hour + 12) % 24
        mock_store.set('goal:quiet_hours', json.dumps([quiet_hour]))

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient:
            MockAmbient.return_value.is_user_deep_focus.return_value = False

            cost = _calculate_social_cost()

        assert cost < 0.4  # No quiet hours penalty

    def test_social_cost_rapid_conversation_penalty(self, mock_store):
        """3+ messages in 5 minutes should add social cost."""
        mock_store.setex('recent_message_count_5min', 300, '4')

        with patch('services.ambient_inference_service.AmbientInferenceService') as MockAmbient:
            MockAmbient.return_value.is_user_deep_focus.return_value = False

            cost = _calculate_social_cost()

        assert cost >= 0.25  # Rapid conversation penalty

    def test_learn_quiet_hours_insufficient_data(self, db, mock_store):
        """Less than 50 interactions should return None."""
        # Seed interaction_log with only 8 rows
        for i in range(5):
            db.execute(
                "INSERT INTO interaction_log (id, event_type, payload, created_at) "
                "VALUES (?, 'message', '{}', '2026-03-20 10:00:00')",
                (str(uuid.uuid4()),),
            )
        for i in range(3):
            db.execute(
                "INSERT INTO interaction_log (id, event_type, payload, created_at) "
                "VALUES (?, 'message', '{}', '2026-03-20 14:00:00')",
                (str(uuid.uuid4()),),
            )
        db.commit()

        result = _learn_quiet_hours()
        assert result is None

    def test_learn_quiet_hours_with_sufficient_data(self, db, mock_store):
        """Hours with < 5% of interactions should be marked quiet."""
        # Seed interaction_log: hours 9-17 have lots of interactions, others sparse
        for h in range(9, 18):
            for i in range(100):
                db.execute(
                    "INSERT INTO interaction_log (id, event_type, payload, created_at) "
                    "VALUES (?, 'message', '{}', ?)",
                    (str(uuid.uuid4()), f'2026-03-20 {h:02d}:00:00'),
                )
        # Sparse hours
        for i in range(2):
            db.execute(
                "INSERT INTO interaction_log (id, event_type, payload, created_at) "
                "VALUES (?, 'message', '{}', '2026-03-20 03:00:00')",
                (str(uuid.uuid4()),),
            )
        db.execute(
            "INSERT INTO interaction_log (id, event_type, payload, created_at) "
            "VALUES (?, 'message', '{}', '2026-03-20 22:00:00')",
            (str(uuid.uuid4()),),
        )
        db.commit()

        result = _learn_quiet_hours()

        assert result is not None
        assert 3 in result   # Hour 3 has very few interactions
        assert 22 in result  # Hour 22 has very few interactions
        assert 10 not in result  # Hour 10 has lots of interactions
        # Hours with zero interactions should also be quiet
        assert 0 in result  # No interactions at midnight

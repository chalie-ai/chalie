"""Tests for H1.5 — Goal Strategy Service (strategy generation on actionable transition)."""

import pytest
import threading
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text: str):
    from services.llm_service import LLMResponse
    return LLMResponse(text=text, model='test-model', provider='mock')


def _make_db_mock(evidence_rows=None, description=None):
    """Return a (db_mock, cursor_mock) pair wired for the connection() pattern."""
    db = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = evidence_rows or []
    cursor.fetchone.return_value = (description,) if description is not None else None
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    db.connection.return_value = conn_ctx

    return db, cursor


# ---------------------------------------------------------------------------
# _gather_evidence_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGatherEvidenceContext:
    """Unit tests for _gather_evidence_context()."""

    def test_formats_evidence_rows(self, mock_store):
        evidence_rows = [
            ('explicit_statement', 'I want to learn Rust', 'topic:coding', 1.0),
            ('topic_recurrence', 'Asked about Rust again', None, 0.8),
        ]
        db, cursor = _make_db_mock(evidence_rows=evidence_rows)

        with patch('services.database_service.get_shared_db_service', return_value=db):
            from services.goal_strategy_service import _gather_evidence_context
            result = _gather_evidence_context('goal-id-123')

        assert '[explicit_statement]' in result
        assert 'I want to learn Rust' in result
        assert '(from topic:coding)' in result
        assert '[topic_recurrence]' in result
        assert 'Asked about Rust again' in result
        # Source is None for second row — should not appear
        assert result.count('(from') == 1

    def test_no_evidence_returns_placeholder(self, mock_store):
        db, cursor = _make_db_mock(evidence_rows=[])

        with patch('services.database_service.get_shared_db_service', return_value=db):
            from services.goal_strategy_service import _gather_evidence_context
            result = _gather_evidence_context('goal-id-456')

        assert 'No evidence' in result

    def test_db_failure_returns_unavailable(self, mock_store):
        with patch('services.database_service.get_shared_db_service', side_effect=RuntimeError("DB down")):
            from services.goal_strategy_service import _gather_evidence_context
            result = _gather_evidence_context('goal-id-789')

        assert 'unavailable' in result.lower()


# ---------------------------------------------------------------------------
# _get_user_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetUserContext:
    """Unit tests for _get_user_context()."""

    def test_returns_traits_from_data_graph(self, mock_store):
        from unittest.mock import MagicMock, patch
        mock_dgs = MagicMock()
        mock_dgs.fetch.return_value = [
            {'key': 'name', 'value': 'Dylan', 'retrieval_weight': 0.99},
            {'key': 'likes_coffee', 'value': 'true', 'retrieval_weight': 0.8},
        ]
        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            from services.goal_strategy_service import _get_user_context
            result = _get_user_context()

        assert 'name: Dylan' in result
        assert 'likes_coffee: true' in result

    def test_truncates_long_context(self, mock_store):
        from unittest.mock import MagicMock, patch
        mock_dgs = MagicMock()
        # Generate traits that exceed 500 chars
        mock_dgs.fetch.return_value = [
            {'key': f'trait_{i}', 'value': 'A' * 50, 'retrieval_weight': 0.9}
            for i in range(20)
        ]
        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            from services.goal_strategy_service import _get_user_context
            result = _get_user_context()

        assert len(result) <= 503  # 500 chars + "..."

    def test_no_traits_returns_empty(self, mock_store):
        from unittest.mock import MagicMock, patch
        mock_dgs = MagicMock()
        mock_dgs.fetch.return_value = []
        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            from services.goal_strategy_service import _get_user_context
            result = _get_user_context()

        assert result == ""

    def test_failure_returns_empty(self, mock_store):
        from unittest.mock import patch
        with patch('services.data_graph_service.get_data_graph_service', side_effect=RuntimeError("unavailable")):
            from services.goal_strategy_service import _get_user_context
            result = _get_user_context()

        assert result == ""


# ---------------------------------------------------------------------------
# _store_strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStoreStrategy:
    """Unit tests for _store_strategy()."""

    def test_updates_goals_row(self, mock_store):
        db, cursor = _make_db_mock()

        with patch('services.database_service.get_shared_db_service', return_value=db), \
             patch('services.time_utils.utc_now') as mock_now:
            mock_now.return_value.isoformat.return_value = '2026-03-22T12:00:00+00:00'

            from services.goal_strategy_service import _store_strategy
            _store_strategy('goal-abc', 'Research the latest Rust resources and summarize them.')

        # Verify the cursor was called with an UPDATE statement
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args[0]
        sql, params = call_args[0], call_args[1]

        assert 'UPDATE goals' in sql
        assert 'strategy' in sql
        assert params[0] == 'Research the latest Rust resources and summarize them.'
        assert params[2] == 'goal-abc'


# ---------------------------------------------------------------------------
# generate_strategy (integration of all helpers)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateStrategy:
    """Unit tests for generate_strategy() end-to-end."""

    def test_returns_strategy_and_stores_it(self, mock_store):
        goal = {
            'id': 'goal-111',
            'description': 'Learn Rust programming',
            'type': 'stated',
        }

        _make_llm_response(
            "Search for beginner Rust tutorials and bookmark the top three."
        )

        with patch('services.goal_strategy_service._gather_evidence_context', return_value="- [explicit] Learn Rust"), \
             patch('services.goal_strategy_service._get_user_context', return_value="Values: engineering growth"), \
             patch('services.goal_strategy_service._call_strategy_llm', return_value="Search for beginner Rust tutorials and bookmark the top three."), \
             patch('services.goal_strategy_service._store_strategy') as mock_store_fn:

            from services.goal_strategy_service import generate_strategy
            result = generate_strategy(goal)

        assert result == "Search for beginner Rust tutorials and bookmark the top three."
        mock_store_fn.assert_called_once_with(
            'goal-111',
            "Search for beginner Rust tutorials and bookmark the top three.",
        )

    def test_llm_failure_returns_none_gracefully(self, mock_store):
        goal = {'id': 'goal-222', 'description': 'Plan a trip', 'type': 'emergent'}

        with patch('services.goal_strategy_service._gather_evidence_context', return_value=""), \
             patch('services.goal_strategy_service._get_user_context', return_value=""), \
             patch('services.goal_strategy_service._call_strategy_llm', return_value=None), \
             patch('services.goal_strategy_service._store_strategy') as mock_store_fn:

            from services.goal_strategy_service import generate_strategy
            result = generate_strategy(goal)

        assert result is None
        mock_store_fn.assert_not_called()

    def test_exception_in_gather_returns_none(self, mock_store):
        goal = {'id': 'goal-333', 'description': 'Read more books', 'type': 'inferred'}

        with patch('services.goal_strategy_service._gather_evidence_context', side_effect=RuntimeError("boom")), \
             patch('services.goal_strategy_service._store_strategy') as mock_store_fn:

            from services.goal_strategy_service import generate_strategy
            result = generate_strategy(goal)

        assert result is None
        mock_store_fn.assert_not_called()

    def test_empty_llm_response_returns_none(self, mock_store):
        goal = {'id': 'goal-444', 'description': 'Exercise daily', 'type': 'stated'}

        with patch('services.goal_strategy_service._gather_evidence_context', return_value="evidence"), \
             patch('services.goal_strategy_service._get_user_context', return_value="context"), \
             patch('services.goal_strategy_service._call_strategy_llm', return_value=""), \
             patch('services.goal_strategy_service._store_strategy') as mock_store_fn:

            from services.goal_strategy_service import generate_strategy
            result = generate_strategy(goal)

        assert result is None
        mock_store_fn.assert_not_called()


# ---------------------------------------------------------------------------
# _call_strategy_llm
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallStrategyLlm:
    """Unit tests for _call_strategy_llm()."""

    def test_returns_strategy_text_from_llm(self, mock_store):
        proxy_mock = MagicMock()
        proxy_mock.send_message.return_value = _make_llm_response(
            "Look into coding bootcamps and compile a shortlist."
        )

        with patch('services.background_llm_queue.create_background_llm_proxy', return_value=proxy_mock):
            from services.goal_strategy_service import _call_strategy_llm
            result = _call_strategy_llm(
                description='Learn to code',
                goal_type='inferred',
                evidence_context='- [topic_recurrence] Mentioned coding 3 times',
                user_context='Values: tech skills',
            )

        assert result == "Look into coding bootcamps and compile a shortlist."

    def test_llm_none_response_returns_none(self, mock_store):
        proxy_mock = MagicMock()
        proxy_mock.send_message.return_value = None

        with patch('services.background_llm_queue.create_background_llm_proxy', return_value=proxy_mock):
            from services.goal_strategy_service import _call_strategy_llm
            result = _call_strategy_llm('Goal', 'emergent', 'evidence', 'context')

        assert result is None

    def test_strategy_truncated_to_500_chars(self, mock_store):
        long_text = "X" * 600
        proxy_mock = MagicMock()
        proxy_mock.send_message.return_value = _make_llm_response(long_text)

        with patch('services.background_llm_queue.create_background_llm_proxy', return_value=proxy_mock):
            from services.goal_strategy_service import _call_strategy_llm
            result = _call_strategy_llm('Goal', 'stated', '', '')

        assert result is not None
        assert len(result) == 500

    def test_proxy_exception_returns_none(self, mock_store):
        with patch('services.background_llm_queue.create_background_llm_proxy', side_effect=RuntimeError("proxy down")):
            from services.goal_strategy_service import _call_strategy_llm
            result = _call_strategy_llm('Goal', 'stated', '', '')

        assert result is None


# ---------------------------------------------------------------------------
# Actionable transition in recalculate_salience
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestActionableTransitionInRecalculate:
    """Verify that strategy generation is triggered on actionable transition."""

    def _make_ecology_db(self, old_status='strengthening', gtype='stated'):
        """Return a db mock configured for recalculate_salience queries."""
        db = MagicMock()
        cursor = MagicMock()

        # First fetchone: goal data row
        goal_row = (
            'goal-xyz',   # id
            gtype,        # type
            old_status,   # status
            5,            # evidence_count
            0.5,          # urgency
            '[]',         # parent_motives
            '[]',         # identity_links
            '2026-03-20T10:00:00+00:00',  # created_at
            '2026-03-22T09:00:00+00:00',  # last_reinforced_at
            None,         # lineage_parent_id
        )
        # Second fetchone: distinct signal_type count
        distinct_row = (3,)
        cursor.fetchone.side_effect = [goal_row, distinct_row]

        conn = MagicMock()
        conn.cursor.return_value = cursor
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=conn)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        db.connection.return_value = conn_ctx

        return db, cursor

    def test_actionable_transition_spawns_thread(self):
        """When status goes from strengthening → actionable, strategy generation is submitted."""
        db, cursor = self._make_ecology_db(old_status='strengthening', gtype='stated')

        with patch('services.database_service.get_shared_db_service', return_value=db), \
             patch('services.goal_ecology_service._strategy_pool') as mock_pool:

            from services.goal_ecology_service import GoalEcologyService
            svc = GoalEcologyService(db_service=db)
            svc.recalculate_salience('goal-xyz')

        assert mock_pool.submit.called, (
            "Expected _strategy_pool.submit() to be called on actionable transition"
        )

    def test_non_actionable_transition_no_thread(self):
        """When status stays at strengthening, no strategy generation is submitted."""
        db, cursor = self._make_ecology_db(old_status='candidate', gtype='developmental')
        # developmental type needs high salience + confidence to become actionable
        # With evidence_count=5 and developmental type, confidence = 0.27, not >= 0.6
        # so status will stay candidate/strengthening

        with patch('services.database_service.get_shared_db_service', return_value=db), \
             patch('services.goal_ecology_service._strategy_pool') as mock_pool:

            from services.goal_ecology_service import GoalEcologyService
            svc = GoalEcologyService(db_service=db)
            svc.recalculate_salience('goal-xyz')

        assert not mock_pool.submit.called, (
            "Should NOT submit strategy generation when goal does not become actionable"
        )

    def test_already_actionable_no_duplicate_thread(self):
        """When status is already actionable, no new strategy thread should spawn."""
        db, cursor = self._make_ecology_db(old_status='actionable', gtype='stated')

        spawned_threads = []

        original_thread_init = threading.Thread.__init__

        def mock_thread_init(self, *args, **kwargs):
            if 'goal-strategy' in kwargs.get('name', ''):
                spawned_threads.append(kwargs.get('name'))
            original_thread_init(self, target=lambda: None, daemon=True)

        with patch('services.database_service.get_shared_db_service', return_value=db), \
             patch.object(threading.Thread, '__init__', mock_thread_init), \
             patch.object(threading.Thread, 'start'):

            from services.goal_ecology_service import GoalEcologyService
            svc = GoalEcologyService(db_service=db)
            svc.recalculate_salience('goal-xyz')

        assert len(spawned_threads) == 0, (
            "Should NOT re-spawn strategy thread when goal is already actionable"
        )


# ---------------------------------------------------------------------------
# Rejected strategy history
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRejectedStrategyHistory:
    """Test rejected strategy history integration."""

    def test_gather_rejected_strategies_empty(self, mock_store):
        """No rejected strategies returns empty string."""
        from services.goal_strategy_service import _gather_rejected_strategies
        with patch('services.database_service.get_shared_db_service') as MockDb:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ('[]',)
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            MockDb.return_value.connection.return_value = mock_conn

            result = _gather_rejected_strategies('test-id')
        assert result == ""

    def test_gather_rejected_strategies_with_history(self, mock_store):
        """Failed strategies returned as formatted list."""
        import json
        feedback = json.dumps([
            {'event': 'strategy_failed', 'strategy': 'Research directly', 'rejection_count': 2},
            {'response': 'engaged', 'timestamp': '2026-03-20'},
        ])
        from services.goal_strategy_service import _gather_rejected_strategies
        with patch('services.database_service.get_shared_db_service') as MockDb:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (feedback,)
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            MockDb.return_value.connection.return_value = mock_conn

            result = _gather_rejected_strategies('test-id')
        assert 'Research directly' in result
        assert 'rejected' in result.lower()
        assert 'different approach' in result.lower()

    def test_strategy_llm_includes_rejected_history(self, mock_store):
        """Strategy LLM prompt includes rejected history when available."""
        from services.goal_strategy_service import _call_strategy_llm
        with patch('services.background_llm_queue.create_background_llm_proxy') as MockProxy:
            mock_response = MagicMock()
            mock_response.text = "New approach"
            MockProxy.return_value.send_message.return_value = mock_response

            rejected = "Previous approaches rejected:\n1. \"Research directly\" (rejected 2 times)"
            _call_strategy_llm("Learn Rust", "stated", "evidence", "context", rejected)

        call_args = MockProxy.return_value.send_message.call_args
        assert 'Research directly' in call_args[0][1]  # user_message includes rejected history

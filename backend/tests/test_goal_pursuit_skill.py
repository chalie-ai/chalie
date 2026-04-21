"""Tests for the goal_pursuit innate skill.

Verifies input validation, thread spawning, and OutputService
surface behaviour on both success and failure.
"""

import json
import sqlite3
from contextlib import contextmanager

import pytest
from unittest.mock import MagicMock, patch

from services.innate_skills.goal_pursuit_skill import handle_goal_pursuit

pytestmark = pytest.mark.unit


# ── Input validation ───────────────────────────────────────────────────────────

class TestHandleGoalPursuitValidation:
    def test_empty_goal_returns_error_json(self):
        """Empty string goal returns a JSON error without spawning a thread."""
        result = json.loads(handle_goal_pursuit('user', {'goal': ''}))
        assert result['success'] is False
        assert 'error' in result

    def test_whitespace_only_goal_returns_error_json(self):
        """Whitespace-only goal is treated as empty and returns an error."""
        result = json.loads(handle_goal_pursuit('user', {'goal': '   \t\n  '}))
        assert result['success'] is False
        assert 'error' in result

    def test_missing_goal_key_returns_error_json(self):
        """Missing 'goal' key in params returns a JSON error."""
        result = json.loads(handle_goal_pursuit('user', {}))
        assert result['success'] is False
        assert 'error' in result

    def test_empty_goal_does_not_start_thread(self):
        """No thread must be spawned when the goal is invalid."""
        with patch('threading.Thread') as mock_thread_cls:
            handle_goal_pursuit('user', {'goal': ''})
            mock_thread_cls.assert_not_called()


# ── Happy-path return contract ─────────────────────────────────────────────────

class TestHandleGoalPursuitSuccess:
    def _call_with_patched_thread(self, goal='Summarise the latest news'):
        """Call handle_goal_pursuit with threading.Thread and GoalEcologyService patched."""
        with patch('threading.Thread') as mock_thread_cls, \
             patch('services.goal_ecology_service.GoalEcologyService') as mock_ecology_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            mock_ecology_cls.return_value = MagicMock()
            raw = handle_goal_pursuit('user', {'goal': goal})
            parsed = json.loads(raw)
            return parsed, mock_thread_cls, mock_thread

    def test_valid_goal_returns_success_true(self):
        """A well-formed goal must return success=True."""
        result, _, _ = self._call_with_patched_thread()
        assert result['success'] is True

    def test_valid_goal_returns_pursuit_id(self):
        """Return JSON must include a non-empty 'pursuit_id'."""
        result, _, _ = self._call_with_patched_thread()
        assert 'pursuit_id' in result
        assert result['pursuit_id']  # non-empty string

    def test_valid_goal_response_contains_working_on_it(self):
        """The 'response' field must communicate that work is in progress."""
        result, _, _ = self._call_with_patched_thread()
        assert 'response' in result
        assert 'Working on it' in result['response']

    def test_valid_goal_returns_json_string(self):
        """handle_goal_pursuit must return a JSON-parseable string."""
        with patch('threading.Thread') as mock_thread_cls, \
             patch('services.goal_ecology_service.GoalEcologyService') as mock_ecology_cls:
            mock_thread_cls.return_value = MagicMock()
            mock_ecology_cls.return_value = MagicMock()
            raw = handle_goal_pursuit('user', {'goal': 'do something'})
        assert isinstance(raw, str)
        # Must not raise
        json.loads(raw)

    def test_pursuit_id_is_hex_string(self):
        """pursuit_id must be the hex representation of a UUID (32 hex chars)."""
        result, _, _ = self._call_with_patched_thread()
        pid = result['pursuit_id']
        assert len(pid) == 32
        int(pid, 16)  # raises ValueError if not hex


# ── Background thread behaviour: success path ─────────────────────────────────

class TestGoalPursuitThreadSuccess:
    def _run_background_target(self, goal='Finish the task', response_text='Task complete.'):
        """Capture and immediately invoke the thread target function.

        The production code constructs GoalPursuitProcessor(raw_input=..., metadata=...)
        and calls .send() on the result (v2 API). The mock wires .send.return_value so
        the response text propagates correctly into OutputService.enqueue_proactive.
        """
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            m = MagicMock()
            return m

        mock_processor_instance = MagicMock()
        mock_processor_instance.send.return_value = response_text

        mock_output_instance = MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_ecology_service.GoalEcologyService') as mock_ecology_cls, \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            mock_ecology_cls.return_value = MagicMock()
            handle_goal_pursuit('user', {'goal': goal})
            # Invoke the background function synchronously
            captured_target['fn']()

        return mock_processor_instance, mock_output_instance

    def test_success_calls_enqueue_proactive(self):
        """On success, OutputService.enqueue_proactive must be called once."""
        _, mock_output = self._run_background_target()
        mock_output.enqueue_proactive.assert_called_once()

    def test_success_source_is_goal_pursuit(self):
        """Proactive message source must be 'goal_pursuit'."""
        _, mock_output = self._run_background_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('source') == 'goal_pursuit'

    def test_success_topic_is_user(self):
        """Proactive message topic must be 'user'."""
        _, mock_output = self._run_background_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('topic') == 'user'

    def test_success_response_text_propagated(self):
        """The response text from the processor must appear in the proactive message."""
        _, mock_output = self._run_background_target(response_text='I found what you needed.')
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('response') == 'I found what you needed.'

    def test_empty_processor_response_uses_fallback_message(self):
        """When processor returns empty text, a fallback message is used instead."""
        mock_processor_instance = MagicMock()
        # .send() returns an empty string (v2 API returns str, not dict)
        mock_processor_instance.send.return_value = ''

        mock_output_instance = MagicMock()
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            return MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_ecology_service.GoalEcologyService') as mock_ecology_cls, \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            mock_ecology_cls.return_value = MagicMock()
            handle_goal_pursuit('user', {'goal': 'a task'})
            captured_target['fn']()

        _, kwargs = mock_output_instance.enqueue_proactive.call_args
        # Must not be empty
        assert kwargs.get('response')
        assert kwargs['response'] != ''


# ── Background thread behaviour: failure path ─────────────────────────────────

class TestGoalPursuitThreadFailure:
    def _run_failing_target(self, error_message='Something went wrong'):
        """Run the thread target where GoalPursuitProcessor.send() raises (v2 API)."""
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            return MagicMock()

        mock_processor_instance = MagicMock()
        mock_processor_instance.send.side_effect = RuntimeError(error_message)

        mock_output_instance = MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_ecology_service.GoalEcologyService') as mock_ecology_cls, \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            mock_ecology_cls.return_value = MagicMock()
            handle_goal_pursuit('user', {'goal': 'a task'})
            captured_target['fn']()  # must not raise

        return mock_output_instance

    def test_processor_failure_calls_enqueue_proactive(self):
        """On processor failure, an error surface via enqueue_proactive must still happen."""
        mock_output = self._run_failing_target()
        mock_output.enqueue_proactive.assert_called_once()

    def test_processor_failure_source_is_goal_pursuit(self):
        """Error proactive message source must still be 'goal_pursuit'."""
        mock_output = self._run_failing_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('source') == 'goal_pursuit'

    def test_processor_failure_response_mentions_failure(self):
        """Error message surfaced to user must reference the failure."""
        mock_output = self._run_failing_target(error_message='network timeout')
        _, kwargs = mock_output.enqueue_proactive.call_args
        response = kwargs.get('response', '')
        assert 'failed' in response.lower() or 'error' in response.lower() or 'network timeout' in response

    def test_processor_failure_does_not_raise(self):
        """The background thread target must never propagate exceptions to the caller."""
        # If _run_failing_target didn't raise, the test passes
        self._run_failing_target()


# ── Goal persistence ───────────────────────────────────────────────────────────

_GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'emergent'
        CHECK (type IN ('stated', 'inferred', 'emergent', 'developmental')),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'strengthening', 'actionable', 'active', 'completed', 'decayed', 'evolved')),
    salience REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.1,
    commitment REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    outcome_feedback TEXT DEFAULT '[]',
    is_muted INTEGER DEFAULT 0,
    urgency REAL DEFAULT 0.0,
    timescale TEXT DEFAULT 'medium_term',
    strategy TEXT,
    strategy_hash INTEGER,
    parent_motives TEXT DEFAULT '[]',
    identity_links TEXT DEFAULT '[]',
    lineage_parent_id TEXT,
    channel TEXT,
    last_reinforced_at TEXT,
    last_acted_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    derived_from TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS goal_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    signal_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    strength REAL NOT NULL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def _make_goals_db():
    """Return an (db_svc_mock, conn) pair backed by an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_GOAL_SCHEMA)
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


class TestHandleGoalPursuitPersistence:
    """Verify that handle_goal_pursuit persists a goal row before spawning the thread.

    These tests do NOT patch threading.Thread so the real WriteQueueService drain
    thread can run.  They DO provide a real in-memory SQLite database so
    GoalEcologyService.create_goal() can complete its submit_sync() write.
    """

    def _run(self, goal='research cold water swimming benefits'):
        db_svc, conn = _make_goals_db()
        # Patch the name as bound inside goal_ecology_service (from-import), not
        # the canonical location, so GoalEcologyService.__init__ receives the mock.
        with patch('services.goal_ecology_service.get_shared_db_service', return_value=db_svc):
            raw = handle_goal_pursuit('user', {'goal': goal})
        result = json.loads(raw)
        return result, conn

    def test_goal_row_written_to_goals_table(self):
        """A row matching the goal description must exist in the goals table."""
        _, conn = self._run()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM goals WHERE description LIKE '%cold water%' OR description LIKE '%swimming%'"
        )
        count = cursor.fetchone()[0]
        assert count >= 1, f"Expected at least 1 goal row, got {count}"

    def test_goal_row_type_is_stated(self):
        """The persisted goal must have type='stated'."""
        _, conn = self._run()
        cursor = conn.execute(
            "SELECT type FROM goals WHERE description LIKE '%cold water%' OR description LIKE '%swimming%'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'stated'

    def test_returns_success_true(self):
        """The skill must still return success=True when persistence succeeds."""
        result, _ = self._run()
        assert result['success'] is True

    def test_persistence_failure_does_not_abort_success_response(self):
        """Even when GoalEcologyService raises, the skill returns success=True."""
        with patch('services.goal_ecology_service.GoalEcologyService',
                   side_effect=RuntimeError('db unavailable')):
            raw = handle_goal_pursuit('user', {'goal': 'research cold water swimming'})
        result = json.loads(raw)
        assert result['success'] is True

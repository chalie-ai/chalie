"""
Tests for backend/services/innate_skills/persistent_task_skill.py

Covers: _create() JSON contract — success/failure shapes, duplicate
detection, and missing-goal guard. All tests are unit-level with the
real SQLite sandbox so no external services are required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.innate_skills.persistent_task_skill import _create


# ── Helpers ──────────────────────────────────────────────────────────

def _make_service(task=None, duplicate=None, raise_on_create=None):
    """Build a mock PersistentTaskService with canned responses."""
    svc = MagicMock()
    svc.find_duplicate.return_value = duplicate
    if raise_on_create:
        svc.create_task.side_effect = raise_on_create
    else:
        svc.create_task.return_value = task or {'id': 42}
    return svc


# ── _create: success path ────────────────────────────────────────────

@pytest.mark.unit
class TestCreateSuccess:
    def test_returns_valid_json_on_success(self):
        """_create() always returns a JSON-parseable string on success."""
        svc = _make_service()
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Research quantum computing'})

        parsed = json.loads(result)  # Must not raise
        assert isinstance(parsed, dict)

    def test_success_flag_is_true(self):
        """_create() sets success=True when the task is created."""
        svc = _make_service()
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Summarise all notes'})

        parsed = json.loads(result)
        assert parsed['success'] is True

    def test_response_key_present_on_success(self):
        """_create() includes a 'response' key in the JSON on success."""
        svc = _make_service()
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Track all meetings'})

        parsed = json.loads(result)
        assert 'response' in parsed
        assert isinstance(parsed['response'], str)
        assert len(parsed['response']) > 0

    def test_no_error_key_on_success(self):
        """_create() does not include an 'error' key when the task is created successfully."""
        svc = _make_service()
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Organise inbox'})

        parsed = json.loads(result)
        assert 'error' not in parsed


# ── _create: failure paths ───────────────────────────────────────────

@pytest.mark.unit
class TestCreateFailure:
    def test_returns_valid_json_on_service_exception(self):
        """_create() returns JSON even when the underlying service raises."""
        svc = _make_service(raise_on_create=RuntimeError('DB locked'))
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Analyse emails'})

        parsed = json.loads(result)  # Must not raise
        assert isinstance(parsed, dict)

    def test_success_flag_is_false_on_service_exception(self):
        """_create() sets success=False when service raises an exception."""
        svc = _make_service(raise_on_create=ValueError('invalid state'))
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Deep dive on topic X'})

        parsed = json.loads(result)
        assert parsed['success'] is False

    def test_error_key_present_on_service_exception(self):
        """_create() includes an 'error' key when the service raises."""
        svc = _make_service(raise_on_create=RuntimeError('connection failed'))
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Map all contacts'})

        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'connection failed' in parsed['error']

    def test_missing_goal_returns_json_error(self):
        """_create() returns JSON with success=False when no goal is supplied."""
        svc = MagicMock()
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create'})

        parsed = json.loads(result)
        assert parsed['success'] is False
        assert 'error' in parsed
        # Service should never be called when goal is missing
        svc.create_task.assert_not_called()


# ── _create: duplicate detection ─────────────────────────────────────

@pytest.mark.unit
class TestCreateDuplicate:
    def test_duplicate_returns_valid_json(self):
        """_create() returns JSON when a duplicate task is detected."""
        duplicate = {
            'id': 7,
            'goal': 'Research quantum computing',
            'status': 'in_progress',
            'progress': {'coverage_estimate': 0.4},
        }
        svc = _make_service(duplicate=duplicate)
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Research quantum computing'})

        parsed = json.loads(result)  # Must not raise
        assert isinstance(parsed, dict)

    def test_duplicate_sets_success_false(self):
        """_create() sets success=False when a duplicate is detected."""
        duplicate = {
            'id': 7,
            'goal': 'Research quantum computing',
            'status': 'in_progress',
            'progress': {'coverage_estimate': 0.4},
        }
        svc = _make_service(duplicate=duplicate)
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Research quantum computing'})

        parsed = json.loads(result)
        assert parsed['success'] is False

    def test_duplicate_includes_error_key(self):
        """_create() includes an 'error' key with context when a duplicate exists."""
        duplicate = {
            'id': 12,
            'goal': 'Monitor all emails from Alice',
            'status': 'accepted',
            'progress': {},
        }
        svc = _make_service(duplicate=duplicate)
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            result = _create('test-topic', {'action': 'create', 'goal': 'Monitor emails from Alice'})

        parsed = json.loads(result)
        assert 'error' in parsed
        assert isinstance(parsed['error'], str)
        assert len(parsed['error']) > 0

    def test_duplicate_does_not_call_create_task(self):
        """_create() does not call create_task when a duplicate is found."""
        duplicate = {
            'id': 3,
            'goal': 'Summarise recent news',
            'status': 'in_progress',
            'progress': {'coverage_estimate': 0.2},
        }
        svc = _make_service(duplicate=duplicate)
        with (
            patch('services.innate_skills.persistent_task_skill._get_service', return_value=svc),
            patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1),
        ):
            _create('test-topic', {'action': 'create', 'goal': 'Summarise recent news'})

        svc.create_task.assert_not_called()

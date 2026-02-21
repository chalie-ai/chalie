"""
Unit tests for GoalService.

Tests: lifecycle, status transitions, dormancy, prompt formatting.
No external dependencies (mocked DB).
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    """Mock DatabaseService with context-manager connection."""
    db = MagicMock()

    # connection() context manager
    conn_ctx = MagicMock()
    cursor = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn_ctx)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    conn_ctx.cursor = MagicMock(return_value=cursor)
    db.connection = MagicMock(return_value=conn_ctx)

    return db, conn_ctx, cursor


@pytest.fixture
def goal_service(mock_db):
    from services.goal_service import GoalService
    db, conn_ctx, cursor = mock_db
    return GoalService(db), db, conn_ctx, cursor


# ---------------------------------------------------------------------------
# Tests: create_goal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_goal_returns_id(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 1

    goal_id = svc.create_goal(title="Build Chalie ecosystem")
    assert isinstance(goal_id, str)
    assert len(goal_id) == 8  # 4 bytes hex = 8 chars


@pytest.mark.unit
def test_create_goal_clamps_priority(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 1

    # Verify priority is clamped — we can check via the execute args
    svc.create_goal(title="Test", priority=99)
    call_args = cursor.execute.call_args
    # priority should be 10 (clamped from 99)
    assert 10 in call_args[0][1]


@pytest.mark.unit
def test_create_goal_truncates_title(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 1

    long_title = "A" * 300
    svc.create_goal(title=long_title)
    call_args = cursor.execute.call_args
    # title (2nd positional arg after goal_id) should be 200 chars
    assert len(call_args[0][1][2]) == 200


# ---------------------------------------------------------------------------
# Tests: update_status
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_update_status_valid_transition(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.fetchone.return_value = ('active',)
    cursor.rowcount = 1

    result = svc.update_status('abc123', 'progressing')
    assert result is True


@pytest.mark.unit
def test_update_status_invalid_transition_rejects(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    # active cannot go to achieved directly
    cursor.fetchone.return_value = ('active',)

    result = svc.update_status('abc123', 'achieved')
    assert result is False


@pytest.mark.unit
def test_update_status_terminal_state_blocks_transition(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    # achieved is terminal
    cursor.fetchone.return_value = ('achieved',)

    result = svc.update_status('abc123', 'active')
    assert result is False


@pytest.mark.unit
def test_update_status_goal_not_found(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.fetchone.return_value = None

    result = svc.update_status('nonexistent', 'progressing')
    assert result is False


# ---------------------------------------------------------------------------
# Tests: add_progress_note
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_add_progress_note_success(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 1

    result = svc.add_progress_note('abc123', "Completed initial research")
    assert result is True


@pytest.mark.unit
def test_add_progress_note_goal_not_found(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 0

    result = svc.add_progress_note('nonexistent', "Some note")
    assert result is False


# ---------------------------------------------------------------------------
# Tests: get_goals_for_prompt
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_goals_for_prompt_empty(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.fetchall.return_value = []

    result = svc.get_goals_for_prompt()
    assert result == ""


@pytest.mark.unit
def test_get_goals_for_prompt_formats_goals(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    now = datetime.now(timezone.utc)
    cursor.fetchall.return_value = [
        ('goal1', 'Build Chalie', 'desc', 'active', 8, 'explicit',
         [], now, now, []),
        ('goal2', 'Learn ML', '', 'progressing', 5, 'inferred',
         [], now, now, []),
    ]

    result = svc.get_goals_for_prompt()
    assert "## Active Goals" in result
    assert "Build Chalie" in result
    assert "high" in result  # priority 8 = high


@pytest.mark.unit
def test_get_goals_for_prompt_topic_prioritization(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    now = datetime.now(timezone.utc)
    cursor.fetchall.return_value = [
        ('g1', 'Unrelated goal', '', 'active', 5, 'explicit',
         [], now, now, []),
        ('g2', 'Build chalie ecosystem', '', 'active', 4, 'explicit',
         ['chalie'], now, now, []),
    ]

    result = svc.get_goals_for_prompt(topic='chalie', limit=1)
    # Topic-related goal should appear first / be included
    assert "Build chalie ecosystem" in result


# ---------------------------------------------------------------------------
# Tests: apply_dormancy
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_apply_dormancy_updates_stale_goals(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 3

    count = svc.apply_dormancy()
    assert count == 3
    cursor.execute.assert_called()
    # Verify the SQL mentions 'dormant'
    sql = cursor.execute.call_args[0][0]
    assert 'dormant' in sql.lower()


@pytest.mark.unit
def test_apply_dormancy_returns_zero_on_none_updated(goal_service):
    svc, db, conn_ctx, cursor = goal_service
    cursor.rowcount = 0

    count = svc.apply_dormancy()
    assert count == 0


# ---------------------------------------------------------------------------
# Tests: status transition table completeness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_all_terminal_states_have_empty_transitions():
    from services.goal_service import _TRANSITIONS
    assert _TRANSITIONS['achieved'] == set()
    assert _TRANSITIONS['abandoned'] == set()


@pytest.mark.unit
def test_dormant_can_only_go_to_active():
    from services.goal_service import _TRANSITIONS
    assert _TRANSITIONS['dormant'] == {'active'}

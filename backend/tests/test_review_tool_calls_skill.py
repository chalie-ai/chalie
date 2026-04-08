"""Tests for the review_tool_calls innate skill.

Verifies time-range retrieval, output formatting, and graceful error handling.
"""

import contextlib
import json
import sqlite3
from datetime import timedelta

import pytest
from unittest.mock import patch

from services.time_utils import utc_now
from services.innate_skills.review_tool_calls_skill import handle_review_tool_calls

pytestmark = pytest.mark.unit


# ── Minimal schema ─────────────────────────────────────────────────────────────

TOOL_CALLS_DDL = [
    """
    CREATE TABLE IF NOT EXISTS transcript (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        channel TEXT NOT NULL DEFAULT 'user',
        role    TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id INTEGER NOT NULL,
        tool_name     TEXT NOT NULL,
        params        TEXT DEFAULT '{}',
        result        TEXT DEFAULT '',
        invoked_by    TEXT NOT NULL CHECK(invoked_by IN ('system', 'llm')),
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        ephemeral     INTEGER NOT NULL DEFAULT 0
    )
    """,
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """Real SQLite DB with tool_calls schema."""
    from services.database_service import DatabaseService

    db_path = str(tmp_path / "test_review_skill.db")
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    for ddl in TOOL_CALLS_DDL:
        shared_conn.execute(ddl)
    shared_conn.commit()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True
    db.db_path = db_path

    _depth = [0]

    @contextlib.contextmanager
    def _connection():
        _depth[0] += 1
        try:
            yield shared_conn
            if _depth[0] == 1:
                shared_conn.commit()
        except Exception:
            if _depth[0] == 1:
                shared_conn.rollback()
            raise
        finally:
            _depth[0] -= 1

    def _fetch_all(sql, params=()):
        cursor = shared_conn.cursor()
        cursor.execute(sql, params)
        rows = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        return rows

    db.connection = _connection
    db.fetch_all = _fetch_all

    yield db
    shared_conn.close()


@pytest.fixture
def patched_db(db_service):
    """Patch get_shared_db_service used by ToolCallService."""
    with patch('services.tool_call_service.get_shared_db_service', return_value=db_service):
        yield db_service


def _insert_tool_call(db_service, tool_name, params, result, created_at, invoked_by='llm', ephemeral=0):
    """Helper: insert a raw tool_call row with a fixed created_at."""
    with db_service.connection() as conn:
        # Insert a dummy transcript entry first
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'x')"
        )
        transcript_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO tool_calls "
            "(transcript_id, tool_name, params, result, invoked_by, created_at, ephemeral) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                transcript_id,
                tool_name,
                json.dumps(params) if isinstance(params, dict) else params,
                result,
                invoked_by,
                created_at,
                ephemeral,
            ),
        )
        cursor.close()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestReviewToolCallsSkill:
    def test_returns_records_in_range(self, patched_db):
        """Records within ±5 min of center are returned."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(
            patched_db, 'memory', {'query': 'coffee'}, 'Coffee result', center_iso
        )

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        assert '1 record(s)' in result['text']
        assert 'memory' in result['text']

    def test_returns_empty_when_no_records(self, patched_db):
        """When no records exist within the window, returns a graceful message."""
        center = utc_now()
        result = handle_review_tool_calls('user', {'date_time': center.isoformat()})

        assert 'No tool calls found' in result['text']
        assert center.isoformat()[:16] in result['text']

    def test_formats_output_includes_tool_name_params_result(self, patched_db):
        """Output includes tool_name, params, and result for each record."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(
            patched_db, 'goals', {'action': 'list'}, 'goal list here', center_iso
        )

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        text = result['text']
        assert 'goals' in text
        assert 'goal list here' in text

    def test_formats_multiple_records(self, patched_db):
        """Multiple records each appear as a separate line."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(patched_db, 'memory', {}, 'result A', center_iso)
        _insert_tool_call(patched_db, 'goals', {}, 'result B', center_iso)

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        text = result['text']
        assert '2 record(s)' in text
        assert 'memory' in text
        assert 'goals' in text

    def test_excludes_records_outside_window(self, patched_db):
        """Records 10 min before center are NOT returned (outside ±5 min)."""
        center = utc_now()
        outside = (center - timedelta(minutes=10)).isoformat()
        _insert_tool_call(patched_db, 'memory', {}, 'old result', outside)

        result = handle_review_tool_calls('user', {'date_time': center.isoformat()})

        assert 'No tool calls found' in result['text']

    def test_includes_ephemeral_records(self, patched_db):
        """Ephemeral records (synthesis, steers) are returned — no filtering."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(
            patched_db, 'tool_synthesis', {}, 'Thinking about it...', center_iso, ephemeral=1
        )

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        assert 'tool_synthesis' in result['text']

    def test_missing_date_time_param_returns_error(self, patched_db):
        """When date_time is missing, returns error message without raising."""
        result = handle_review_tool_calls('user', {})

        assert 'error' in result['text'].lower() or 'required' in result['text'].lower()

    def test_empty_date_time_string_returns_error(self, patched_db):
        """When date_time is an empty string, returns error message."""
        result = handle_review_tool_calls('user', {'date_time': ''})

        assert 'error' in result['text'].lower() or 'required' in result['text'].lower()

    def test_channel_param_unused_does_not_raise(self, patched_db):
        """Channel parameter is accepted but not used — any value is valid."""
        center = utc_now()
        result = handle_review_tool_calls('arbitrary_channel', {'date_time': center.isoformat()})

        # Returns a valid dict (not an exception)
        assert isinstance(result, dict)
        assert 'text' in result

    def test_long_result_truncated_in_output(self, patched_db):
        """Results longer than 300 chars are truncated with ellipsis."""
        center = utc_now()
        center_iso = center.isoformat()
        long_result = 'x' * 400
        _insert_tool_call(patched_db, 'memory', {}, long_result, center_iso)

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        text = result['text']
        # The truncated result should appear (300 chars + '...')
        assert '...' in text
        # The full 400-char result must NOT appear verbatim
        assert long_result not in text

    def test_error_status_hint_for_error_result(self, patched_db):
        """Results starting with 'error' are annotated with (error) status hint."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(patched_db, 'memory', {}, 'Error: not found', center_iso)

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        assert '(error)' in result['text']

    def test_ok_status_hint_for_normal_result(self, patched_db):
        """Results not starting with 'error' are annotated with (ok) status hint."""
        center = utc_now()
        center_iso = center.isoformat()
        _insert_tool_call(patched_db, 'memory', {}, 'Some memory result', center_iso)

        result = handle_review_tool_calls('user', {'date_time': center_iso})

        assert '(ok)' in result['text']

    def test_query_failure_returns_error_message(self, patched_db):
        """When ToolCallService.get_by_timerange raises, returns graceful error dict."""
        center = utc_now()
        # The skill lazy-imports ToolCallService inside the function body, so we
        # patch the class method on the services module where it lives.
        with patch(
            'services.tool_call_service.ToolCallService.get_by_timerange',
            side_effect=Exception("DB connection lost")
        ):
            result = handle_review_tool_calls('user', {'date_time': center.isoformat()})

        assert 'error' in result['text'].lower()
        assert isinstance(result, dict)

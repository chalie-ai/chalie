"""Tests for ToolCallService — unified API for tool_calls audit entries.

Covers store, store_batch, get_by_transcript, get_by_timerange,
and get_find_tools_results.
"""

import contextlib
import json
import sqlite3
from datetime import timedelta

import pytest
from unittest.mock import patch

from services.database_service import DatabaseService
from services.tool_call_service import ToolCallService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── Minimal schema for tool_calls table ─────────────────────────────────────

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
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        ephemeral     INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (transcript_id) REFERENCES transcript(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_transcript ON tool_calls(transcript_id)",
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC)",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """Real SQLite DB with tool_calls schema, single shared connection."""
    db_path = str(tmp_path / "test_tool_calls.db")
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
def svc(db_service):
    """ToolCallService wired to the isolated database."""
    with patch('services.tool_call_service.get_shared_db_service', return_value=db_service):
        yield ToolCallService()


@pytest.fixture
def transcript_id(db_service):
    """Insert a dummy transcript row and return its rowid."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'hello')"
        )
        rowid = cursor.lastrowid
        cursor.close()
    return rowid


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_rows(db_service):
    return db_service.fetch_all("SELECT * FROM tool_calls ORDER BY id")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestStore:
    def test_store_single_record(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'memory', {'query': 'coffee'}, 'result text')
        rows = _all_rows(db_service)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'
        assert rows[0]['transcript_id'] == transcript_id
        assert rows[0]['result'] == 'result text'

    def test_store_ephemeral_true(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        rows = _all_rows(db_service)
        assert rows[0]['ephemeral'] == 1

    def test_store_ephemeral_false(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'memory', {}, 'result', ephemeral=False)
        rows = _all_rows(db_service)
        assert rows[0]['ephemeral'] == 0

    def test_store_params_serialized_from_dict(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'memory', {'key': 'value'}, 'ok')
        rows = _all_rows(db_service)
        assert json.loads(rows[0]['params']) == {'key': 'value'}

    def test_store_params_passthrough_when_string(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'memory', '{"raw": true}', 'ok')
        rows = _all_rows(db_service)
        assert rows[0]['params'] == '{"raw": true}'


class TestStoreBatch:
    def test_store_batch(self, svc, db_service, transcript_id):
        tool_calls = [
            {'id': 'tc1', 'name': 'memory', 'input': {'query': 'x'}},
            {'id': 'tc2', 'name': 'schedule', 'input': {'action': 'list'}},
        ]
        results = [
            {'result': 'memory result', 'status': 'ok'},
            {'result': 'schedule result', 'status': 'ok'},
        ]
        svc.store_batch(transcript_id, tool_calls, results)
        rows = _all_rows(db_service)
        assert len(rows) == 2
        assert rows[0]['tool_name'] == 'memory'
        assert rows[1]['tool_name'] == 'schedule'

    def test_store_batch_empty_does_nothing(self, svc, db_service, transcript_id):
        svc.store_batch(transcript_id, [], [])
        assert _all_rows(db_service) == []

    def test_store_batch_default_ephemeral_true(self, svc, db_service, transcript_id):
        tool_calls = [{'id': 'tc1', 'name': 'memory', 'input': {}}]
        results = [{'result': 'ok', 'status': 'ok'}]
        svc.store_batch(transcript_id, tool_calls, results)
        rows = _all_rows(db_service)
        assert rows[0]['ephemeral'] == 1

    def test_store_batch_find_tools_special_case(self, svc, db_service, transcript_id):
        """find_tools result with _discovered_tools stores tool names in params, not raw input."""
        tool_calls = [
            {'id': 'tc1', 'name': 'find_tools', 'input': {'query': 'email'}},
        ]
        results = [
            {'result': 'Found 2 tools', '_discovered_tools': ['send_email', 'read_email'], 'status': 'ok'},
        ]
        svc.store_batch(transcript_id, tool_calls, results)
        rows = _all_rows(db_service)
        assert len(rows) == 1
        params = json.loads(rows[0]['params'])
        assert params == {'discovered': ['send_email', 'read_email']}

    def test_store_batch_find_tools_without_discovered_uses_input(self, svc, db_service, transcript_id):
        """find_tools result without _discovered_tools falls back to raw input."""
        tool_calls = [
            {'id': 'tc1', 'name': 'find_tools', 'input': {'query': 'email'}},
        ]
        results = [{'result': 'no tools found', 'status': 'ok'}]
        svc.store_batch(transcript_id, tool_calls, results)
        rows = _all_rows(db_service)
        params = json.loads(rows[0]['params'])
        assert params == {'query': 'email'}


class TestGetByTranscript:
    def test_get_by_transcript_include_ephemeral(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        svc.store(transcript_id, 'memory', {}, 'result', ephemeral=False)
        rows = svc.get_by_transcript(transcript_id, include_ephemeral=True)
        assert len(rows) == 2

    def test_get_by_transcript_exclude_ephemeral(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        svc.store(transcript_id, 'memory', {}, 'result', ephemeral=False)
        rows = svc.get_by_transcript(transcript_id, include_ephemeral=False)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'

    def test_get_by_transcript_returns_empty_when_none(self, svc, db_service, transcript_id):
        rows = svc.get_by_transcript(transcript_id)
        assert rows == []

    def test_get_by_transcript_only_returns_matching_transcript(self, svc, db_service, transcript_id):
        # Insert a second transcript row
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'other')"
            )
            other_id = cursor.lastrowid
            cursor.close()

        svc.store(transcript_id, 'memory', {}, 'mine')
        svc.store(other_id, 'memory', {}, 'not mine')

        rows = svc.get_by_transcript(transcript_id)
        assert len(rows) == 1
        assert rows[0]['result'] == 'mine'


class TestGetByTimerange:
    def test_get_by_timerange_returns_records_within_window(self, svc, db_service, transcript_id):
        center = utc_now()
        center_iso = center.isoformat()

        # Insert at center time by using the service and then manually updating created_at
        svc.store(transcript_id, 'memory', {}, 'in range')
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE tool_calls SET created_at = ? WHERE tool_name = 'memory'",
                (center_iso,),
            )

        rows = svc.get_by_timerange(center_iso)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'

    def test_get_by_timerange_empty_when_no_records(self, svc, transcript_id):
        center = utc_now()
        rows = svc.get_by_timerange(center.isoformat())
        assert rows == []

    def test_get_by_timerange_excludes_records_outside_window(self, svc, db_service, transcript_id):
        center = utc_now()
        center_iso = center.isoformat()

        # Record 10 minutes before (outside ±5 min window)
        outside_ts = (center - timedelta(minutes=10)).isoformat()
        svc.store(transcript_id, 'memory', {}, 'outside range')
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE tool_calls SET created_at = ? WHERE tool_name = 'memory'",
                (outside_ts,),
            )

        rows = svc.get_by_timerange(center_iso)
        assert rows == []

    def test_get_by_timerange_includes_boundary_records(self, svc, db_service, transcript_id):
        center = utc_now()
        # Exactly 5 minutes before — should be included (>= boundary)
        boundary_ts = (center - timedelta(minutes=5)).isoformat()
        svc.store(transcript_id, 'schedule', {}, 'on boundary')
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE tool_calls SET created_at = ? WHERE tool_name = 'schedule'",
                (boundary_ts,),
            )
        rows = svc.get_by_timerange(center.isoformat())
        assert len(rows) == 1


class TestGetFindToolsResults:
    def test_get_find_tools_results_returns_deduped_flat_list(self, svc, db_service, transcript_id):
        # Two find_tools calls with overlapping discovered tools
        with db_service.connection() as conn:
            conn.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral) "
                "VALUES (?, 'find_tools', ?, '', 1)",
                (transcript_id, json.dumps({'discovered': ['send_email', 'read_email']})),
            )
            conn.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral) "
                "VALUES (?, 'find_tools', ?, '', 1)",
                (transcript_id, json.dumps({'discovered': ['read_email', 'delete_email']})),
            )

        results = svc.get_find_tools_results(transcript_id)
        assert results == ['send_email', 'read_email', 'delete_email']

    def test_get_find_tools_results_empty_when_no_find_tools_records(self, svc, db_service, transcript_id):
        svc.store(transcript_id, 'memory', {}, 'result')
        results = svc.get_find_tools_results(transcript_id)
        assert results == []

    def test_get_find_tools_results_empty_when_no_records(self, svc, transcript_id):
        results = svc.get_find_tools_results(transcript_id)
        assert results == []

    def test_get_find_tools_results_handles_malformed_params(self, svc, db_service, transcript_id):
        with db_service.connection() as conn:
            conn.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral) "
                "VALUES (?, 'find_tools', 'not valid json', '', 1)",
                (transcript_id,),
            )
        # Malformed JSON must not raise — returns empty gracefully
        results = svc.get_find_tools_results(transcript_id)
        assert results == []

    def test_get_find_tools_results_ignores_other_tools(self, svc, db_service, transcript_id):
        with db_service.connection() as conn:
            conn.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral) "
                "VALUES (?, 'memory', ?, '', 0)",
                (transcript_id, json.dumps({'discovered': ['should_not_appear']})),
            )
        results = svc.get_find_tools_results(transcript_id)
        assert results == []

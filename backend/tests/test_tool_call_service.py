"""Tests for ToolCallService — unified API for tool_calls audit entries.

Covers store, store_batch, get_by_transcript, and get_by_timerange.
"""

import pytest

from services.tool_call_service import ToolCallService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


@pytest.fixture
def svc(db):
    """ToolCallService wired to the real schema via the db fixture."""
    return ToolCallService()


@pytest.fixture
def transcript_id(db):
    """Insert a dummy transcript row and return its rowid."""
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'hello')"
    )
    rowid = cursor.lastrowid
    cursor.close()
    return rowid


def _all_rows(db):
    return db.execute("SELECT * FROM tool_calls ORDER BY id").fetchall()


class TestStore:
    def test_store_single_record(self, svc, db, transcript_id):
        svc.store(transcript_id, 'memory', {'query': 'coffee'}, 'result text')
        rows = _all_rows(db)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'
        assert rows[0]['transcript_id'] == transcript_id
        assert rows[0]['result'] == 'result text'

    def test_store_ephemeral_true(self, svc, db, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        rows = _all_rows(db)
        assert rows[0]['ephemeral'] == 1



class TestStoreBatch:
    def test_store_batch(self, svc, db, transcript_id):
        tool_calls = [
            {'id': 'tc1', 'name': 'memory', 'input': {'query': 'x'}},
            {'id': 'tc2', 'name': 'schedule', 'input': {'action': 'list'}},
        ]
        results = [
            {'result': 'memory result', 'status': 'ok'},
            {'result': 'schedule result', 'status': 'ok'},
        ]
        svc.store_batch(transcript_id, tool_calls, results)
        rows = _all_rows(db)
        assert len(rows) == 2
        assert rows[0]['tool_name'] == 'memory'
        assert rows[1]['tool_name'] == 'schedule'


class TestGetByTranscript:
    def test_get_by_transcript_include_ephemeral(self, svc, db, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        svc.store(transcript_id, 'memory', {}, 'result', ephemeral=False)
        rows = svc.get_by_transcript(transcript_id, include_ephemeral=True)
        assert len(rows) == 2

    def test_get_by_transcript_exclude_ephemeral(self, svc, db, transcript_id):
        svc.store(transcript_id, 'tool_synthesis', {}, 'narration', ephemeral=True)
        svc.store(transcript_id, 'memory', {}, 'result', ephemeral=False)
        rows = svc.get_by_transcript(transcript_id, include_ephemeral=False)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'

    def test_get_by_transcript_only_returns_matching_transcript(self, svc, db, transcript_id):
        cursor = db.cursor()
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
    def test_get_by_timerange_returns_records_within_window(self, svc, db, transcript_id):
        center = utc_now()
        center_iso = center.isoformat()

        svc.store(transcript_id, 'memory', {}, 'in range')
        db.execute(
            "UPDATE tool_calls SET created_at = ? WHERE tool_name = 'memory'",
            (center_iso,),
        )

        rows = svc.get_by_timerange(center_iso)
        assert len(rows) == 1
        assert rows[0]['tool_name'] == 'memory'




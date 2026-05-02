"""Unit tests for compaction_persistence.get_compaction() canonical SQL lookup.

Invariants tested:
- Returns None when no compaction rows exist for the channel
- Returns None when only failure-status rows exist (status filter)
- Returns the latest success row by tool_calls.id (append-only ordering)
- Ignores rows from other channels (channel isolation)
- Returns correct compacted_text, compacted_up_to_id, tool_call_id
"""

import json
import pytest

pytestmark = pytest.mark.unit

_CHANNEL = 'test_cp_canonical'
_OTHER_CHANNEL = 'test_cp_other'


def _seed_transcript(db, channel, content='anchor'):
    cursor = db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, 'user', ?, '2026-01-01 00:00:00')",
        (channel, content),
    )
    db.commit()
    return cursor.lastrowid


def _seed_tool_call(db, transcript_id, status, compacted_text, compacted_up_to_id):
    params = json.dumps({'status': status, 'compacted_up_to_id': compacted_up_to_id})
    cursor = db.execute(
        "INSERT INTO tool_calls "
        "(transcript_id, tool_name, params, result, ephemeral, created_at) "
        "VALUES (?, 'compaction', ?, ?, 0, '2026-01-01 00:00:01')",
        (transcript_id, params, compacted_text),
    )
    db.commit()
    return cursor.lastrowid


class TestGetCompactionCanonicalLookup:
    def test_returns_none_when_no_rows(self, db):
        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_returns_none_when_only_failure_rows(self, db):
        t_id = _seed_transcript(db, _CHANNEL)
        _seed_tool_call(db, t_id, 'failure', 'some failed output', t_id)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_returns_success_row(self, db):
        t_id = _seed_transcript(db, _CHANNEL)
        _seed_tool_call(db, t_id, 'success', 'the summary text', t_id)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is not None
        assert result['compacted_text'] == 'the summary text'
        assert result['compacted_up_to_id'] == t_id

    def test_returns_latest_success_by_id(self, db):
        t1 = _seed_transcript(db, _CHANNEL, 'first')
        t2 = _seed_transcript(db, _CHANNEL, 'second')
        _seed_tool_call(db, t1, 'success', 'old summary', t1)
        _seed_tool_call(db, t2, 'success', 'new summary', t2)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_text'] == 'new summary'
        assert result['compacted_up_to_id'] == t2

    def test_success_after_failure_returns_success(self, db):
        t1 = _seed_transcript(db, _CHANNEL, 'first')
        t2 = _seed_transcript(db, _CHANNEL, 'second')
        _seed_tool_call(db, t1, 'success', 'good summary', t1)
        _seed_tool_call(db, t2, 'failure', 'failed attempt', t2)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_text'] == 'good summary'

    def test_channel_isolation(self, db):
        t_other = _seed_transcript(db, _OTHER_CHANNEL, 'other')
        _seed_tool_call(db, t_other, 'success', 'other channel summary', t_other)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_channel_isolation_own_channel_visible(self, db):
        t_own = _seed_transcript(db, _CHANNEL, 'own')
        t_other = _seed_transcript(db, _OTHER_CHANNEL, 'other')
        _seed_tool_call(db, t_own, 'success', 'own summary', t_own)
        _seed_tool_call(db, t_other, 'success', 'other summary', t_other)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_text'] == 'own summary'

    def test_result_contains_tool_call_id(self, db):
        t_id = _seed_transcript(db, _CHANNEL)
        tc_id = _seed_tool_call(db, t_id, 'success', 'summary', t_id)

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['tool_call_id'] == tc_id

    def test_compacted_up_to_id_zero_when_param_missing(self, db):
        """If params lack compacted_up_to_id the field defaults to 0."""
        t_id = _seed_transcript(db, _CHANNEL)
        db.execute(
            "INSERT INTO tool_calls "
            "(transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'compaction', ?, 'text', 0, '2026-01-01 00:00:01')",
            (t_id, json.dumps({'status': 'success'})),
        )
        db.commit()

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_up_to_id'] == 0

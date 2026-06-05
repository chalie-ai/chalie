"""Unit tests for compaction_persistence.get_compaction() canonical SQL lookup.

Storage model (design §3.6): the compaction summary lives in the transcript
table as a row with role='compaction' whose OWN id is the watermark
(compacted_up_to_id). There is no status/success/failure concept — _compact()
only ever writes a row when summary extraction succeeds, so a persisted row is
always a real summary.

Rows are seeded through the production factory transcript_service.write_input_row
— the exact path _compact() uses — so this test exercises the real write+read
seam, not a hand-rolled INSERT.

Invariants tested:
- Returns None when no compaction row exists for the channel
- Returns the newest compaction row by id (append-only ordering)
- Ignores rows from other channels (channel isolation)
- Result shape: compacted_text, compacted_up_to_id (= the row's own id),
  tool_call_id (always None — legacy field), created_at
"""

import pytest

pytestmark = pytest.mark.unit

_CHANNEL = 'test_cp_canonical'
_OTHER_CHANNEL = 'test_cp_other'


def _seed_compaction(channel, summary):
    """Persist a compaction summary the way _compact() does (design §3.6):
    a transcript row with role='compaction'. Returns its id (the watermark)."""
    from services import transcript_service
    return transcript_service.write_input_row(channel, 'compaction', summary)


class TestGetCompactionCanonicalLookup:
    def test_returns_none_when_no_rows(self, db):
        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_returns_compaction_row(self, db):
        watermark = _seed_compaction(_CHANNEL, 'the summary text')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is not None
        assert result['compacted_text'] == 'the summary text'
        assert result['compacted_up_to_id'] == watermark

    def test_returns_latest_by_id(self, db):
        _seed_compaction(_CHANNEL, 'old summary')
        newer = _seed_compaction(_CHANNEL, 'new summary')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_text'] == 'new summary'
        assert result['compacted_up_to_id'] == newer

    def test_channel_isolation(self, db):
        _seed_compaction(_OTHER_CHANNEL, 'other channel summary')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_own_channel_visible_alongside_other_channel(self, db):
        _seed_compaction(_CHANNEL, 'own summary')
        _seed_compaction(_OTHER_CHANNEL, 'other summary')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result['compacted_text'] == 'own summary'

    def test_result_shape_tool_call_id_none_and_created_at_present(self, db):
        watermark = _seed_compaction(_CHANNEL, 'summary')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        # tool_call_id is a retired legacy field: the watermark is no longer a
        # tool_call, so the canonical reader always returns None for it.
        assert result['tool_call_id'] is None
        # created_at is populated from the transcript row's DB default.
        assert result['created_at'] is not None
        assert result['compacted_up_to_id'] == watermark

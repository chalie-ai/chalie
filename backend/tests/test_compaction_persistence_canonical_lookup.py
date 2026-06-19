"""Unit tests for compaction_persistence.get_compaction() canonical SQL lookup.

Storage model (design §3.6): the compaction summary lives in the transcript
table as a row with role='compaction' whose OWN id is the watermark
(compacted_up_to_id). There is no status/success/failure concept — _compact()
only ever writes a row when summary extraction succeeds, so a persisted row is
always a real summary.

Rows are seeded through the production factory transcript_service.write_input_row
— the exact path _compact() uses — so this test exercises the real write+read
seam, not a hand-rolled INSERT.
"""

import sqlite3

import pytest

pytestmark = pytest.mark.unit

_CHANNEL = 'test_cp_canonical'
_OTHER_CHANNEL = 'test_cp_other'


def _seed_compaction(channel: str, summary: str) -> int:
    from services import transcript_service
    return transcript_service.write_input_row(channel, 'compaction', summary)


class TestGetCompactionCanonicalLookup:
    def test_returns_none_when_no_rows(self, db: sqlite3.Connection) -> None:
        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_returns_compaction_row(self, db: sqlite3.Connection) -> None:
        watermark = _seed_compaction(_CHANNEL, 'the summary text')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is not None
        assert result['compacted_text'] == 'the summary text'
        assert result['compacted_up_to_id'] == watermark

    def test_returns_latest_by_id(self, db: sqlite3.Connection) -> None:
        _seed_compaction(_CHANNEL, 'old summary')
        newer = _seed_compaction(_CHANNEL, 'new summary')

        from services.compaction_persistence import get_compaction
        from typing import cast
        result = get_compaction(_CHANNEL)
        assert cast("dict[str, object]", result)['compacted_text'] == 'new summary'
        assert cast("dict[str, object]", result)['compacted_up_to_id'] == newer

    def test_channel_isolation(self, db: sqlite3.Connection) -> None:
        _seed_compaction(_OTHER_CHANNEL, 'other channel summary')

        from services.compaction_persistence import get_compaction
        result = get_compaction(_CHANNEL)
        assert result is None

    def test_own_channel_visible_alongside_other_channel(self, db: sqlite3.Connection) -> None:
        _seed_compaction(_CHANNEL, 'own summary')
        _seed_compaction(_OTHER_CHANNEL, 'other summary')

        from services.compaction_persistence import get_compaction
        from typing import cast
        result = get_compaction(_CHANNEL)
        assert cast("dict[str, object]", result)['compacted_text'] == 'own summary'

    def test_result_shape_tool_call_id_none_and_created_at_present(self, db: sqlite3.Connection) -> None:
        watermark = _seed_compaction(_CHANNEL, 'summary')

        from services.compaction_persistence import get_compaction
        from typing import cast
        result = get_compaction(_CHANNEL)
        # tool_call_id is a retired legacy field: the watermark is no longer a
        # tool_call, so the canonical reader always returns None for it.
        assert cast("dict[str, object]", result)['tool_call_id'] is None
        # created_at is populated from the transcript row's DB default.
        assert cast("dict[str, object]", result)['created_at'] is not None
        assert cast("dict[str, object]", result)['compacted_up_to_id'] == watermark

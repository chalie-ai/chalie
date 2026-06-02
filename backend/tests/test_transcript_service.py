"""Tests for Transcript Service.

Tests cover:
- append()
- get_recent() with and without since_id
- TestDbStateExtractionTrigger: DB-state-driven extraction trigger

All tests use the real production stack against the shared `db` fixture
(in-memory SQLite built from schema.sql).  No mocks except the single
acceptable boundary: patch('_trigger_episode_extraction') in
TestDbStateExtractionTrigger, which stubs the daemon-thread spawn only —
the query logic itself is real.
"""

from unittest.mock import patch


class TestAppend:
    def test_basic_append(self, db):
        from services.transcript_service import append

        rowid = append('test-topic', 'user', 'Hi')
        assert rowid is not None
        assert rowid > 0

        cursor = db.cursor()
        cursor.execute("SELECT channel, role, content FROM transcript WHERE id = ?", (rowid,))
        row = cursor.fetchone()
        assert row['channel'] == 'test-topic'
        assert row['role'] == 'user'
        assert row['content'] == 'Hi'

    def test_append_with_tool_info(self, db):
        from services.transcript_service import append

        rowid = append(
            'test-topic', 'tool', 'ok',
            tool_call_id='tc_123', tool_name='search',
        )
        cursor = db.cursor()
        cursor.execute(
            "SELECT tool_call_id, tool_name FROM transcript WHERE id = ?",
            (rowid,),
        )
        row = cursor.fetchone()
        assert row['tool_call_id'] == 'tc_123'
        assert row['tool_name'] == 'search'

    def test_append_internal_flag(self, db):
        from services.transcript_service import append

        rowid = append('test-topic', 'internal', 'notes', internal=True)
        cursor = db.cursor()
        cursor.execute("SELECT internal FROM transcript WHERE id = ?", (rowid,))
        assert cursor.fetchone()[0] == 1

    def test_append_empty_content_returns_none(self, db):
        from services.transcript_service import append
        assert append('test-topic', 'user', '') is None

    def test_append_empty_topic_returns_none(self, db):
        from services.transcript_service import append
        assert append('', 'user', 'content') is None


class TestGetRecent:
    def test_get_recent_returns_ordered(self, db):
        from services.transcript_service import append, get_recent

        append('test', 'user', 'First')
        append('test', 'assistant', 'Second')
        append('test', 'user', 'Third')

        results = get_recent('test', limit=10)
        assert len(results) == 3
        # Should be in chronological order (oldest first)
        assert results[0]['content'] == 'First'
        assert results[2]['content'] == 'Third'

    def test_get_recent_respects_limit(self, db):
        from services.transcript_service import append, get_recent

        for i in range(10):
            append('test', 'user', f'Msg{i}')

        results = get_recent('test', limit=3)
        assert len(results) == 3

    def test_get_recent_since_id(self, db):
        from services.transcript_service import append, get_recent

        id1 = append('test', 'user', 'First')
        append('test', 'assistant', 'Second')
        append('test', 'user', 'Third')

        results = get_recent('test', since_id=id1)
        assert len(results) == 2
        assert results[0]['content'] == 'Second'
        assert results[1]['content'] == 'Third'

    def test_get_recent_filters_by_topic(self, db):
        from services.transcript_service import append, get_recent

        append('topic-a', 'user', 'A message')
        append('topic-b', 'user', 'B message')

        results = get_recent('topic-a')
        assert len(results) == 1
        assert results[0]['content'] == 'A message'


class TestDbStateExtractionTrigger:
    """Extraction trigger is DB-state-driven: counts transcripts with
    id > MAX(episodes.transcript_id_end) for the channel. When the tail
    reaches _EXTRACTION_THRESHOLD, extraction fires. No process-local
    state, so restarts cannot desync from accumulated history.

    Only acceptable mock: patch.object(ts, '_trigger_episode_extraction') —
    a fire-and-forget daemon-thread boundary spy. The query logic is real.
    """

    def test_fires_when_untriggered_tail_crosses_threshold(self, db):
        """Seed THRESHOLD untriggered transcripts in a channel; the next
        _maybe_trigger_extraction call must fire."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        # Episodes are only produced for the 'user' channel — the production gate
        # in _maybe_trigger_extraction returns early for any other channel
        # (transcript_service.py, commit cdc3c832). Seed the gated channel.
        channel = 'user'

        for _ in range(threshold):
            db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (channel,),
            )
        db.commit()

        fired = []
        with patch.object(ts, '_trigger_episode_extraction', side_effect=lambda c, r: fired.append((c, r))):
            ts._maybe_trigger_extraction(channel, 999)

        assert fired == [(channel, 999)]

    def test_does_not_fire_below_threshold(self, db):
        """Below threshold, no trigger."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        channel = 'ch-quiet'

        for _ in range(threshold - 1):
            db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (channel,),
            )
        db.commit()

        fired = []
        with patch.object(ts, '_trigger_episode_extraction', side_effect=lambda c, r: fired.append((c, r))):
            ts._maybe_trigger_extraction(channel, 999)

        assert fired == []

    def test_episodes_in_other_channels_do_not_mask(self, db):
        """Episodes in other channels must not suppress a stale channel."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        # 'user' is the only channel the production gate fires for; an episode in
        # a different channel must still not mask the 'user' tail.
        stale = 'user'
        other = 'ch-other'

        for _ in range(threshold):
            db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (stale,),
            )
        # Episode in a DIFFERENT channel — must not suppress stale channel's trigger
        db.execute(
            "INSERT INTO episodes (id, channel, gist, salience, transcript_id_start, "
            "transcript_id_end, created_at) VALUES ('other-ep', ?, 'g', 5, 1, 9999, datetime('now'))",
            (other,),
        )
        db.commit()

        fired = []
        with patch.object(ts, '_trigger_episode_extraction', side_effect=lambda c, r: fired.append((c, r))):
            ts._maybe_trigger_extraction(stale, 999)

        assert fired == [(stale, 999)]

    def test_latest_episode_end_suppresses_until_tail_grows(self, db):
        """An episode already covering the tail means count_since = 0 →
        no fire. Only once more transcripts accumulate past the episode's
        transcript_id_end does the trigger re-fire."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        channel = 'user'  # production gate only fires for the 'user' channel

        # Seed transcripts 1..threshold
        for _ in range(threshold):
            db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (channel,),
            )
        # Episode covers the full tail (transcript_id_end = last row)
        last_id = db.execute(
            "SELECT MAX(id) FROM transcript WHERE channel = ?", (channel,)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO episodes (id, channel, gist, salience, transcript_id_start, "
            "transcript_id_end, created_at) VALUES ('ep-cov', ?, 'g', 5, 1, ?, datetime('now'))",
            (channel, last_id),
        )
        db.commit()

        fired = []
        with patch.object(ts, '_trigger_episode_extraction', side_effect=lambda c, r: fired.append((c, r))):
            # First call — episode covers everything, no untriggered tail
            ts._maybe_trigger_extraction(channel, last_id)
            assert fired == []

            # Add THRESHOLD more transcripts past the episode
            for _ in range(threshold):
                db.execute(
                    "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                    (channel,),
                )
            db.commit()

            new_last = db.execute(
                "SELECT MAX(id) FROM transcript WHERE channel = ?", (channel,)
            ).fetchone()[0]
            ts._maybe_trigger_extraction(channel, new_last)
            assert fired == [(channel, new_last)]

    def test_soft_deleted_episodes_do_not_mask(self, db):
        """Soft-deleted episodes must be ignored when computing the tail."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        channel = 'user'  # production gate only fires for the 'user' channel

        for _ in range(threshold):
            db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (channel,),
            )
        last_id = db.execute(
            "SELECT MAX(id) FROM transcript WHERE channel = ?", (channel,)
        ).fetchone()[0]
        # Episode exists but is soft-deleted — must not suppress
        db.execute(
            "INSERT INTO episodes (id, channel, gist, salience, transcript_id_start, "
            "transcript_id_end, created_at, deleted_at) "
            "VALUES ('ep-del', ?, 'g', 5, 1, ?, datetime('now'), datetime('now'))",
            (channel, last_id),
        )
        db.commit()

        fired = []
        with patch.object(ts, '_trigger_episode_extraction', side_effect=lambda c, r: fired.append((c, r))):
            ts._maybe_trigger_extraction(channel, last_id)
        assert fired == [(channel, last_id)]

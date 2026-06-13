"""Tests for Transcript Service.

Tests cover:
- append()
- get_recent() with and without since_id
- TestDbStateExtractionTrigger: DB-state-driven extraction trigger, gated by
  the per-source profile (only extract_episodes channels reach the encoder).

All tests use the real production stack against the shared `db` fixture
(in-memory SQLite built from schema.sql).

The per-source gate test (``test_muted_channel_never_fires_even_above_threshold``)
drives the REAL ``_maybe_trigger_extraction`` for a muted channel and asserts the
real DB outcome (no episode row) — no production code is patched, because the
profile gate returns BEFORE any thread is spawned, so the outcome is fully
deterministic and observable in the episodes table.

The six pre-existing ``TestDbStateExtractionTrigger`` count-path tests patch the
daemon-thread spawn (``_trigger_episode_extraction``) to observe the
fire/no-fire decision without running the encoder thread. They are pre-existing
and out of scope for this ticket; a separate sweep will rework them onto a
real-DB outcome assertion.
"""

import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


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

    ``test_muted_channel_never_fires_even_above_threshold`` (the TKT-926
    per-source gate invariant) drives the REAL trigger and asserts the real DB
    outcome (no episode row) — no production code is patched.

    The six count-path tests below patch the daemon-thread spawn
    (``patch.object(ts, '_trigger_episode_extraction')``) to observe the
    fire/no-fire decision without running the encoder thread. They are
    pre-existing and out of scope for this ticket; the query logic and the
    profile gate both run for real in every one.
    """

    def test_fires_when_untriggered_tail_crosses_threshold(self, db):
        """Seed THRESHOLD untriggered transcripts in a channel; the next
        _maybe_trigger_extraction call must fire."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        # Episodes are produced only for channels whose source profile has
        # extract_episodes (user, dmn, external-agent:*). Any other channel
        # resolves to the muted default and returns before the COUNT. Seed an
        # extracting channel so this exercises the THRESHOLD path, not the gate.
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

    def test_does_not_fire_below_threshold_on_extracting_channel(self, db):
        """On an EXTRACTING channel (user), one short of the threshold must not
        fire. This pins the real count path: the gate is open, so only the
        accumulated-tail count decides. (Previously this used a muted channel
        and passed via the gate short-circuit, never exercising the threshold.)"""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        channel = 'user'

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

    def test_muted_channel_never_fires_even_above_threshold(self, db):
        """Per-source gate (the headline invariant of TKT-926): a channel whose
        profile is muted (no extract_episodes) produces NO episodes no matter how
        much transcript it accumulates. Seed WELL ABOVE the threshold on a muted
        delegate channel, drive the REAL ``_maybe_trigger_extraction``, and assert
        the real DB outcome — zero episode rows for that channel.

        No production code is patched. The profile gate returns BEFORE the encoder
        thread is ever spawned, so there is no thread to join and no LLM call: the
        outcome is fully deterministic. A revert of the gate (back to a bare
        ``channel != 'user'`` short-circuit, or dropping the profile check) would
        let the count path fire the encoder on a delegate channel and write an
        episode row, reddening this assertion.
        """
        import services.transcript_service as ts
        from services.source_profiles import profile_for

        channel = 'delegate:research'
        assert not profile_for(channel).extract_episodes, (
            "test premise: delegate:* must be a muted source profile"
        )

        last_id = None
        for _ in range(ts._EXTRACTION_THRESHOLD * 2):
            cur = db.execute(
                "INSERT INTO transcript (channel, role, content) VALUES (?, 'user', 'x')",
                (channel,),
            )
            last_id = cur.lastrowid
        db.commit()

        # Real production entry point — no mock. For a muted channel the gate
        # returns before any DB read or thread spawn.
        ts._maybe_trigger_extraction(channel, last_id)

        episode_count = db.execute(
            "SELECT COUNT(*) FROM episodes WHERE channel = ?", (channel,)
        ).fetchone()[0]
        assert episode_count == 0, (
            "a muted-source channel must never produce an episode regardless of "
            f"how much transcript it accumulates; found {episode_count} episode(s) "
            f"for channel {channel!r}"
        )

    def test_episodes_in_other_channels_do_not_mask(self, db):
        """Episodes in other channels must not suppress a stale channel."""
        import services.transcript_service as ts

        threshold = ts._EXTRACTION_THRESHOLD
        # 'user' is an episode-producing (extracting) channel; an episode in a
        # DIFFERENT channel's watermark must not mask the 'user' tail (the
        # COUNT watermark is per-channel: WHERE episodes.channel = ?).
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
        channel = 'user'  # an episode-producing channel (gate open; count decides)

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
        channel = 'user'  # an episode-producing channel (gate open; count decides)

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

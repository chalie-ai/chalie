"""Unit tests for DMNService and DMNMessageProcessor.

Strategy: input → output, not wiring.
- Real in-memory SQLite via the `db` fixture (full production schema).
- Real MemoryStore via the `store` fixture.
- Constructor injection patches are used *only* to wire db+store into the
  service at instantiation — all subsequent calls hit real objects.
"""

import time
from datetime import datetime, timezone, timedelta

import pytest

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utc_offset(seconds: float) -> datetime:
    """Return a timezone-aware UTC datetime offset from now by *seconds*."""
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _make_service(db, store):
    """Instantiate DMNService with the test db + store, bypassing process singletons.

    The patches are infrastructure injection only — the returned service object
    uses real db and store instances for all method calls.
    """
    from unittest.mock import patch
    from services.dmn_service import DMNService
    from services.database_service import get_shared_db_service

    with patch('services.database_service.get_shared_db_service',
               return_value=get_shared_db_service()), \
         patch('services.memory_client.MemoryClientService.create_connection',
               return_value=store):
        svc = DMNService()
    return svc


def _insert_episode(db, ep_id, gist, salience=5,
                    emotional_valence=None,
                    retrieval_weight=1.0, deleted_at=None):
    """Insert a minimal episode row into the real database."""
    from services.time_utils import utc_now
    ts = utc_now().isoformat()
    db.execute(
        "INSERT INTO episodes "
        "(id, gist, salience, channel, "
        "created_at, emotional_valence, retrieval_weight, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ep_id, gist, salience, 'general', ts, emotional_valence,
         retrieval_weight, deleted_at),
    )
    db.commit()


# ── TestFormatEpisodes ─────────────────────────────────────────────────────────

class TestFormatEpisodes:
    """_format_episodes() — direct input/output tests on the formatting logic.

    Each row is (gist, salience, emotional_valence, created_at).
    """

    def test_basic_fields_appear(self, db, store):
        """gist, salience, and timestamp appear in output for a minimal row."""
        svc = _make_service(db, store)
        rows = [('User asked about weather', 7, None, '2026-04-10T14:30:00')]
        result = svc._format_episodes(rows)

        assert 'User asked about weather' in result
        assert 'salience:7' in result
        assert '2026-04-10 14:30' in result

    def test_emotional_valence_included_when_present(self, db, store):
        """A numeric emotional_valence is formatted to 1 decimal place."""
        svc = _make_service(db, store)
        rows = [('Good session', 6, 0.8, '2026-04-10T08:00:00')]
        result = svc._format_episodes(rows)

        assert 'valence:0.8' in result

    def test_null_valence_omitted(self, db, store):
        """A NULL emotional_valence does not produce a valence token in output."""
        svc = _make_service(db, store)
        rows = [('Neutral event', 4, None, '2026-04-10T07:00:00')]
        result = svc._format_episodes(rows)

        assert 'valence:' not in result

    def test_multiple_episodes_numbered(self, db, store):
        """Multiple rows produce incrementing 1., 2., 3. prefixes."""
        svc = _make_service(db, store)
        rows = [
            ('First', 3, None, '2026-04-10T01:00:00'),
            ('Second', 4, None, '2026-04-10T02:00:00'),
            ('Third', 5, None, '2026-04-10T03:00:00'),
        ]
        result = svc._format_episodes(rows)

        assert '1.' in result
        assert '2.' in result
        assert '3.' in result

    def test_gist_always_rendered(self, db, store):
        """The gist string always appears in the output."""
        svc = _make_service(db, store)
        rows = [('Just a gist', 3, None, '2026-04-10T09:00:00')]
        result = svc._format_episodes(rows)

        assert 'Just a gist' in result


# ── TestGatherRecentContext ────────────────────────────────────────────────────

class TestGatherRecentContext:
    """_gather_recent_context() — reads from the real episodes table."""

    def test_returns_empty_string_when_no_episodes(self, db, store):
        """No episodes → empty string returned, no exception."""
        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert result == ''

    def test_single_episode_gist_appears(self, db, store):
        """A single episode's gist appears in the returned string."""
        _insert_episode(db, 'ep-1', 'User asked about the weather forecast')

        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert 'User asked about the weather forecast' in result
        assert result != ''

    def test_multiple_episodes_all_appear(self, db, store):
        """All inserted episodes appear in the output."""
        _insert_episode(db, 'ep-a', 'Episode Alpha')
        _insert_episode(db, 'ep-b', 'Episode Beta')
        _insert_episode(db, 'ep-c', 'Episode Gamma')

        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert 'Episode Alpha' in result
        assert 'Episode Beta' in result
        assert 'Episode Gamma' in result

    def test_deleted_episodes_excluded(self, db, store):
        """Episodes with a deleted_at timestamp are not included in context."""
        _insert_episode(db, 'ep-live', 'Active episode')
        _insert_episode(db, 'ep-dead', 'Deleted episode',
                        deleted_at='2026-04-09T00:00:00+00:00')

        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert 'Active episode' in result
        assert 'Deleted episode' not in result

    def test_salience_value_appears(self, db, store):
        """The salience integer is visible in the formatted output."""
        _insert_episode(db, 'ep-sal', 'High salience moment', salience=9)

        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert 'salience:9' in result

    def test_episodes_numbered_starting_from_one(self, db, store):
        """Output starts with '1.' — episodes are numbered."""
        _insert_episode(db, 'ep-num', 'Some episode')

        svc = _make_service(db, store)
        result = svc._gather_recent_context()

        assert '1.' in result


# ── TestGatherSalienceContext ──────────────────────────────────────────────────

class TestGatherSalienceContext:
    """_gather_salience_context() — reads concepts and salient episodes."""

    def test_returns_empty_string_when_all_tables_empty(self, db, store):
        """Empty knowledge + episodes + goals → empty string, no exception."""
        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert result == ''

    def test_concept_key_and_value_appear(self, db, store):
        """A data_graph concept's key and value appear in the output."""
        from services.time_utils import utc_now
        ts = utc_now().isoformat()
        db.execute(
            "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
            "first_seen_at, last_confirmed_at) VALUES ('user_specific', ?, ?, 0.9, ?, ?)",
            ('preferred_language', 'Python', ts, ts),
        )
        db.commit()

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert 'preferred_language' in result
        assert 'Python' in result

    def test_concept_section_header_present(self, db, store):
        """'High-priority memories:' section header appears when data_graph rows exist."""
        from services.time_utils import utc_now
        ts = utc_now().isoformat()
        db.execute(
            "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
            "first_seen_at, last_confirmed_at) VALUES ('user_specific', 'hobby', 'hiking', 0.8, ?, ?)",
            (ts, ts),
        )
        db.commit()

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert 'High-priority memories:' in result

    def test_high_retrieval_weight_episode_in_notable_section(self, db, store):
        """Episodes with retrieval_weight >= 0.7 appear under 'Notable episodes:'."""
        _insert_episode(db, 'ep-high', 'Big breakthrough moment',
                        retrieval_weight=0.9)

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert 'Big breakthrough moment' in result
        assert 'Notable episodes:' in result

    def test_low_retrieval_weight_episode_excluded(self, db, store):
        """Episodes with retrieval_weight < 0.7 do not appear in salience context."""
        _insert_episode(db, 'ep-low', 'Routine note', retrieval_weight=0.5)

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert 'Routine note' not in result

    def test_non_concept_knowledge_excluded(self, db, store):
        """Knowledge rows with kind != 'concept' do not appear in salience context."""
        from services.time_utils import utc_now
        ts = utc_now().isoformat()
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence, "
            "created_at, updated_at) VALUES ('fact', 'user', 'birthday', '1990-01-01', 0.9, ?, ?)",
            (ts, ts),
        )
        db.commit()

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        # 'fact' kind rows should not be in the High-priority memories section
        assert 'birthday' not in result

    def test_both_sections_when_data_present(self, db, store):
        """When concepts and salient episodes both exist, both sections appear."""
        from services.time_utils import utc_now
        ts = utc_now().isoformat()

        # Seed data_graph
        db.execute(
            "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
            "first_seen_at, last_confirmed_at) VALUES ('user_specific', 'skill', 'Python', 0.9, ?, ?)",
            (ts, ts),
        )
        db.commit()
        _insert_episode(db, 'ep-notable', 'Key milestone achieved',
                        retrieval_weight=0.85)

        svc = _make_service(db, store)
        result = svc._gather_salience_context()

        assert 'High-priority memories:' in result
        assert 'Notable episodes:' in result


# ── TestActiveTopicWhitelist ───────────────────────────────────────────────────
#
# After Commit 9 (channel collapse), _active_topic() is an Option A whitelist:
# it only accepts channel='user' rows. Any other channel — regardless of name —
# is invisible, and the method always returns either 'user' (when a user row
# exists) or 'general' (when the transcript is empty or has only background rows).
#
# The old TestActiveTopic class asserted pre-collapse behaviour (arbitrary
# topic names like 'work-chat', 'cooking', etc.) which is no longer correct.

class TestActiveTopicWhitelist:
    """_active_topic() — Commit 9 whitelist: only channel='user' rows are visible."""

    def test_returns_user_when_user_rows_present(self, db, store):
        """When the transcript contains a 'user' channel row, _active_topic() returns 'user'."""
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'hello', '2026-04-10T10:00:00')"
        )
        db.commit()

        svc = _make_service(db, store)
        result = svc._active_topic()

        assert result == 'user'

    def test_returns_user_even_when_only_background_channels(self, db, store):
        """Whitelist semantics: even with no 'user' rows, the fallback is still 'general'.

        Inserting only background-channel rows (goal_pursuit, dmn, scheduled)
        must NOT cause _active_topic() to return a background channel name.
        The whitelist forces the result to either 'user' or 'general'.
        """
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'goal_pursuit', 'running...', '2026-04-10T12:00:00')"
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('dmn', 'proactive_thought', 'idle review', '2026-04-10T13:00:00')"
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('scheduled', 'scheduled', 'daily task', '2026-04-10T14:00:00')"
        )
        db.commit()

        svc = _make_service(db, store)
        result = svc._active_topic()

        # Whitelist: background rows are invisible; no 'user' row → fall through to 'general'
        assert result == 'general'


# ── TestOnTurn ─────────────────────────────────────────────────────────────────

class TestOnTurn:
    """on_turn() — resets state so the idle timer restarts and recent_fired clears."""

    def test_resets_idle_timer_to_approximately_now(self, db, store):
        """After on_turn(), the measured idle time is less than 2 seconds."""
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        # Wind back the timer by 30 minutes to simulate past idle
        svc._last_turn_ts = utc_now() - timedelta(minutes=30)

        svc.on_turn()

        idle_s = (utc_now() - svc._last_turn_ts).total_seconds()
        assert idle_s < 2, f"Expected idle ~0s after on_turn(), got {idle_s:.2f}s"

    def test_clears_recent_fired_when_true(self, db, store):
        """on_turn() sets _recent_fired to False even when it was True."""
        svc = _make_service(db, store)
        svc._recent_fired = True

        svc.on_turn()

        assert svc._recent_fired is False

    def test_recent_fired_remains_false_when_already_false(self, db, store):
        """on_turn() is idempotent — _recent_fired stays False if already False."""
        svc = _make_service(db, store)
        svc._recent_fired = False

        svc.on_turn()

        assert svc._recent_fired is False


# ── TestCheck ─────────────────────────────────────────────────────────────────

class TestCheck:
    """check() — fires 'recent' after FIRST_IDLE_S, fires 'salience' after REPEAT_S.

    These tests verify the time-gating logic by setting internal timestamps
    directly and confirming which mode _fire() is called with (or not at all).
    Because _fire() makes LLM calls, we patch only _fire() itself — the
    time-gating code under test remains real.
    """

    def test_recent_fires_when_idle_exceeds_threshold(self, db, store):
        """_fire('recent') is called when idle time >= FIRST_IDLE_S."""
        from unittest.mock import patch
        from services.dmn_service import FIRST_IDLE_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._last_turn_ts = utc_now() - timedelta(seconds=FIRST_IDLE_S + 120)
        svc._recent_fired = False

        fired_modes = []
        with patch.object(svc, '_fire', side_effect=lambda m: fired_modes.append(m)):
            svc.check()

        assert 'recent' in fired_modes

    def test_recent_does_not_fire_before_threshold(self, db, store):
        """_fire() is not called when idle time < FIRST_IDLE_S."""
        from unittest.mock import patch
        from services.dmn_service import FIRST_IDLE_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._last_turn_ts = utc_now() - timedelta(seconds=FIRST_IDLE_S - 120)
        svc._recent_fired = False

        fired_modes = []
        with patch.object(svc, '_fire', side_effect=lambda m: fired_modes.append(m)):
            svc.check()

        assert fired_modes == []

    def test_recent_does_not_fire_twice(self, db, store):
        """_fire('recent') is NOT called again once _recent_fired is already True."""
        from unittest.mock import patch
        from services.dmn_service import FIRST_IDLE_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._last_turn_ts = utc_now() - timedelta(seconds=FIRST_IDLE_S + 300)
        svc._recent_fired = True
        # Set last_salience_ts to recent so salience won't fire either
        svc._last_salience_ts = utc_now()

        fired_modes = []
        with patch.object(svc, '_fire', side_effect=lambda m: fired_modes.append(m)):
            svc.check()

        assert 'recent' not in fired_modes

    def test_salience_fires_when_repeat_interval_elapsed(self, db, store):
        """_fire('salience') is called when time since last salience >= REPEAT_S."""
        from unittest.mock import patch
        from services.dmn_service import REPEAT_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._recent_fired = True
        svc._last_salience_ts = utc_now() - timedelta(seconds=REPEAT_S + 120)

        fired_modes = []
        with patch.object(svc, '_fire', side_effect=lambda m: fired_modes.append(m)):
            svc.check()

        assert 'salience' in fired_modes

    def test_salience_does_not_fire_before_repeat_interval(self, db, store):
        """_fire('salience') is not called when less than REPEAT_S has elapsed."""
        from unittest.mock import patch
        from services.dmn_service import REPEAT_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._recent_fired = True
        svc._last_salience_ts = utc_now() - timedelta(seconds=REPEAT_S - 120)

        fired_modes = []
        with patch.object(svc, '_fire', side_effect=lambda m: fired_modes.append(m)):
            svc.check()

        assert fired_modes == []

    def test_check_updates_recent_fired_after_fire(self, db, store):
        """After check() fires 'recent', _recent_fired is set to True."""
        from unittest.mock import patch
        from services.dmn_service import FIRST_IDLE_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._last_turn_ts = utc_now() - timedelta(seconds=FIRST_IDLE_S + 60)
        svc._recent_fired = False

        with patch.object(svc, '_fire'):
            svc.check()

        assert svc._recent_fired is True

    def test_check_updates_last_salience_ts_after_recent_fire(self, db, store):
        """After check() fires 'recent', _last_salience_ts is set to a recent time."""
        from unittest.mock import patch
        from services.dmn_service import FIRST_IDLE_S
        from services.time_utils import utc_now

        svc = _make_service(db, store)
        svc._last_turn_ts = utc_now() - timedelta(seconds=FIRST_IDLE_S + 60)
        svc._recent_fired = False
        svc._last_salience_ts = None

        with patch.object(svc, '_fire'):
            svc.check()

        assert svc._last_salience_ts is not None
        age_s = (utc_now() - svc._last_salience_ts).total_seconds()
        assert age_s < 2


# ── TestRateLimit ──────────────────────────────────────────────────────────────

class TestRateLimit:
    """_rate_limit_exceeded() — uses the real MemoryStore ZSET."""

    def test_returns_false_when_no_deliveries(self, db, store):
        """No entries in the ZSET → rate limit not exceeded."""
        svc = _make_service(db, store)

        assert svc._rate_limit_exceeded() is False

    def test_returns_false_below_max(self, db, store):
        """Fewer than MAX_PROACTIVE_PER_DAY recent entries → not exceeded."""
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET

        svc = _make_service(db, store)
        now_ts = time.time()
        for i in range(MAX_PROACTIVE_PER_DAY - 1):
            store.zadd(_DELIVERY_ZSET, {f'entry-{i}': now_ts - i})

        assert svc._rate_limit_exceeded() is False

    def test_returns_true_at_max(self, db, store):
        """Exactly MAX_PROACTIVE_PER_DAY recent entries → rate limit exceeded."""
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET

        svc = _make_service(db, store)
        now_ts = time.time()
        for i in range(MAX_PROACTIVE_PER_DAY):
            store.zadd(_DELIVERY_ZSET, {f'entry-{i}': now_ts - i})

        assert svc._rate_limit_exceeded() is True

    def test_returns_true_above_max(self, db, store):
        """More than MAX_PROACTIVE_PER_DAY recent entries → rate limit exceeded."""
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET

        svc = _make_service(db, store)
        now_ts = time.time()
        for i in range(MAX_PROACTIVE_PER_DAY + 2):
            store.zadd(_DELIVERY_ZSET, {f'entry-{i}': now_ts - i})

        assert svc._rate_limit_exceeded() is True

    def test_stale_entries_outside_24h_do_not_count(self, db, store):
        """Deliveries older than 24 hours are pruned and do not count against the limit."""
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET, _24H_S

        svc = _make_service(db, store)
        # Insert exactly MAX entries, all more than 24 h ago
        old_ts = time.time() - _24H_S - 3600
        for i in range(MAX_PROACTIVE_PER_DAY):
            store.zadd(_DELIVERY_ZSET, {f'old-{i}': old_ts - i})

        # _rate_limit_exceeded() prunes stale entries; count should drop to 0
        assert svc._rate_limit_exceeded() is False


# ── TestInQuietHours ───────────────────────────────────────────────────────────

class TestInQuietHours:
    """_in_quiet_hours() — returns True during 23:00–08:00 local."""

    def test_returns_true_at_midnight(self, db, store):
        """Hour 0 (midnight) is in the quiet window."""
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        svc = _make_service(db, store)
        midnight_utc = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=midnight_utc):
            result = svc._in_quiet_hours()

        assert result is True

    def test_returns_true_at_2am(self, db, store):
        """Hour 2 is inside the quiet window."""
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        svc = _make_service(db, store)
        two_am = datetime(2026, 4, 10, 2, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=two_am):
            result = svc._in_quiet_hours()

        assert result is True

    def test_returns_true_at_23h(self, db, store):
        """Hour 23 (11 PM) is the start of the quiet window."""
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        svc = _make_service(db, store)
        eleven_pm = datetime(2026, 4, 10, 23, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=eleven_pm):
            result = svc._in_quiet_hours()

        assert result is True

    def test_returns_false_at_noon(self, db, store):
        """Hour 12 (noon) is outside the quiet window."""
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        svc = _make_service(db, store)
        noon = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=noon):
            result = svc._in_quiet_hours()

        assert result is False

    def test_returns_false_at_8am(self, db, store):
        """Hour 8 (8 AM) is the first active hour — not in the quiet window."""
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        svc = _make_service(db, store)
        eight_am = datetime(2026, 4, 10, 8, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=eight_am):
            result = svc._in_quiet_hours()

        assert result is False

    def test_returns_true_on_exception(self, db, store):
        """When get_user_tz raises, the method defaults to True (safe/conservative)."""
        from unittest.mock import patch

        svc = _make_service(db, store)

        with patch('services.dmn_service.get_user_tz', side_effect=RuntimeError('tz error')):
            result = svc._in_quiet_hours()

        assert result is True

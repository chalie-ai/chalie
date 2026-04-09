"""Unit tests for DMNService."""

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(dt: datetime) -> datetime:
    """Ensure dt is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _make_service(db_service, store):
    """Instantiate DMNService with injected db + store, bypassing singletons."""
    from services.dmn_service import DMNService
    with patch('services.database_service.get_shared_db_service', return_value=db_service), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
        svc = DMNService()
    return svc


# ---------------------------------------------------------------------------
# TestDMNOnTurn
# ---------------------------------------------------------------------------

class TestDMNOnTurn:

    def test_on_turn_resets_idle_timer(self, db, store):
        """After calling on_turn, the idle time should be approximately zero."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        # Wind back the last-turn timestamp by 30 minutes to simulate prior idle
        svc._last_turn_ts = _utc(datetime.now(timezone.utc) - timedelta(minutes=30))

        svc.on_turn()

        from services.time_utils import utc_now
        idle_s = (utc_now() - svc._last_turn_ts).total_seconds()
        assert idle_s < 2, f"Expected idle ~0s after on_turn, got {idle_s:.2f}s"

    def test_on_turn_resets_recent_fired(self, db, store):
        """on_turn must clear _recent_fired regardless of its prior value."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        svc._recent_fired = True

        svc.on_turn()

        assert svc._recent_fired is False


# ---------------------------------------------------------------------------
# TestDMNCheck
# ---------------------------------------------------------------------------

class TestDMNCheck:

    def test_recent_fires_after_idle_threshold(self, db, store):
        """check() must call _fire('recent') when idle exceeds FIRST_IDLE_S."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import FIRST_IDLE_S

        svc = _make_service(get_shared_db_service(), store)
        svc._last_turn_ts = _utc(datetime.now(timezone.utc) - timedelta(seconds=FIRST_IDLE_S + 60))
        svc._recent_fired = False

        with patch.object(svc, '_fire') as mock_fire:
            svc.check()

        mock_fire.assert_called_once_with('recent')

    def test_no_fire_before_threshold(self, db, store):
        """check() must not call _fire when idle is below FIRST_IDLE_S."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import FIRST_IDLE_S

        svc = _make_service(get_shared_db_service(), store)
        svc._last_turn_ts = _utc(datetime.now(timezone.utc) - timedelta(seconds=FIRST_IDLE_S - 120))
        svc._recent_fired = False

        with patch.object(svc, '_fire') as mock_fire:
            svc.check()

        mock_fire.assert_not_called()

    def test_salience_fires_after_repeat(self, db, store):
        """check() must call _fire('salience') when REPEAT_S has elapsed since last salience."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import REPEAT_S

        svc = _make_service(get_shared_db_service(), store)
        svc._recent_fired = True
        svc._last_salience_ts = _utc(datetime.now(timezone.utc) - timedelta(seconds=REPEAT_S + 60))

        with patch.object(svc, '_fire') as mock_fire:
            svc.check()

        mock_fire.assert_called_once_with('salience')

    def test_salience_does_not_fire_before_repeat(self, db, store):
        """check() must not fire salience when less than REPEAT_S has elapsed."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import REPEAT_S

        svc = _make_service(get_shared_db_service(), store)
        svc._recent_fired = True
        svc._last_salience_ts = _utc(datetime.now(timezone.utc) - timedelta(seconds=REPEAT_S - 120))

        with patch.object(svc, '_fire') as mock_fire:
            svc.check()

        mock_fire.assert_not_called()


# ---------------------------------------------------------------------------
# TestDMNGates
# ---------------------------------------------------------------------------

class TestDMNGates:

    def test_quiet_hours_prevents_firing(self, db, store):
        """_fire must skip generation when the local hour falls in the quiet window."""
        from services.database_service import get_shared_db_service
        from zoneinfo import ZoneInfo

        svc = _make_service(get_shared_db_service(), store)

        # Build a UTC time whose local equivalent in UTC+0 is 2 AM (inside quiet hours)
        quiet_utc = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)

        with patch('services.dmn_service.get_user_tz', return_value=ZoneInfo('UTC')), \
             patch('services.dmn_service.utc_now', return_value=quiet_utc), \
             patch.object(svc, '_gather_context') as mock_ctx, \
             patch.object(svc, '_proactive_generate') as mock_gen:
            svc._fire('recent')

        # Neither context gathering nor generation should be attempted
        mock_ctx.assert_not_called()
        mock_gen.assert_not_called()

    def test_rate_limit_prevents_firing(self, db, store):
        """_fire must skip generation once MAX_PROACTIVE_PER_DAY deliveries have been recorded."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET

        svc = _make_service(get_shared_db_service(), store)

        # Pre-fill the delivery ZSET with MAX_PROACTIVE_PER_DAY entries
        now_ts = time.time()
        for i in range(MAX_PROACTIVE_PER_DAY):
            store.zadd(_DELIVERY_ZSET, {f'entry-{i}': now_ts - i})

        with patch.object(svc, '_in_quiet_hours', return_value=False), \
             patch.object(svc, '_gather_context') as mock_ctx, \
             patch.object(svc, '_proactive_generate') as mock_gen:
            svc._fire('recent')

        mock_ctx.assert_not_called()
        mock_gen.assert_not_called()

    def test_rate_limit_does_not_block_under_limit(self, db, store):
        """_fire must proceed when fewer than MAX_PROACTIVE_PER_DAY deliveries exist."""
        from services.database_service import get_shared_db_service
        from services.dmn_service import MAX_PROACTIVE_PER_DAY, _DELIVERY_ZSET

        svc = _make_service(get_shared_db_service(), store)

        # Pre-fill with fewer than the limit
        now_ts = time.time()
        entries_to_add = MAX_PROACTIVE_PER_DAY - 2
        for i in range(entries_to_add):
            store.zadd(_DELIVERY_ZSET, {f'entry-{i}': now_ts - i})

        with patch.object(svc, '_in_quiet_hours', return_value=False), \
             patch.object(svc, '_gather_context', return_value='some context') as mock_ctx, \
             patch.object(svc, '_proactive_generate', return_value=False), \
             patch.object(svc, '_log_cycle'):
            svc._fire('recent')
            # _gather_context was reached, meaning the rate-limit gate passed
            mock_ctx.assert_called_once_with('recent')


# ---------------------------------------------------------------------------
# TestDMNContextGathering
# ---------------------------------------------------------------------------

class TestDMNContextGathering:

    def test_recent_context_with_episodes(self, db, store):
        """_gather_recent_context returns numbered lines for each episode row."""
        from services.database_service import get_shared_db_service
        from services.time_utils import utc_now

        ts = utc_now().isoformat()
        # intent, context, emotion are JSONB columns — must be valid JSON strings.
        # action, outcome, gist, topic are plain TEXT; salience is a 1-10 integer.
        db.execute(
            "INSERT INTO episodes "
            "(id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ('ep-1', '"query"', '{}', 'searched_weather', '"neutral"', 'success',
             'User asked about weather', 1, 'general', ts),
        )
        db.execute(
            "INSERT INTO episodes "
            "(id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ('ep-2', '"discuss"', '{}', '', '"neutral"', 'success',
             'Discussed meal plan', 1, 'general', ts),
        )
        db.commit()

        svc = _make_service(get_shared_db_service(), store)
        result = svc._gather_recent_context()

        assert result != ''
        # Both gists must appear
        assert 'User asked about weather' in result
        assert 'Discussed meal plan' in result
        # Action should appear in brackets
        assert '[searched_weather]' in result
        # Lines are numbered
        assert '1.' in result
        assert '2.' in result

    def test_recent_context_empty_table(self, db, store):
        """_gather_recent_context returns an empty string when the episodes table is empty."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        result = svc._gather_recent_context()

        assert result == ''

    def test_salience_context_with_data(self, db, store):
        """_gather_salience_context includes knowledge, goals, and tasks sections."""
        from services.database_service import get_shared_db_service
        from services.time_utils import utc_now

        ts = utc_now().isoformat()

        # Insert a concept into knowledge — kind, entity, key, created_at, updated_at are NOT NULL
        db.execute(
            "INSERT INTO knowledge (kind, entity, key, value, confidence, created_at, updated_at) "
            "VALUES ('concept', 'user', ?, ?, 0.9, ?, ?)",
            ('favourite_language', 'Python', ts, ts),
        )
        # Insert an active goal — description, type, status, salience, confidence are NOT NULL
        db.execute(
            "INSERT INTO goals (id, description, type, status, salience, confidence) "
            "VALUES (?, ?, 'emergent', 'active', 0.8, 0.9)",
            ('g-1', 'Learn more about astronomy'),
        )
        db.commit()

        svc = _make_service(get_shared_db_service(), store)
        result = svc._gather_salience_context()

        assert 'favourite_language' in result
        assert 'Python' in result
        assert 'Learn more about astronomy' in result
        assert 'High-priority memories:' in result
        assert 'Active goals:' in result

    def test_salience_context_empty_tables(self, db, store):
        """_gather_salience_context returns an empty string when all tables are empty."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        result = svc._gather_salience_context()

        assert result == ''


# ---------------------------------------------------------------------------
# TestDMNBuildPrompt
# ---------------------------------------------------------------------------

class TestDMNBuildPrompt:

    def test_recent_prompt_contains_context(self, db, store):
        """The 'recent' prompt embeds the context text and references recent episodes."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        context = 'Episode alpha\nEpisode beta'
        prompt = svc._build_prompt('recent', context)

        assert context in prompt
        assert 'Recent episodes:' in prompt
        assert 'DMN_NO_ACTION' in prompt

    def test_salience_prompt_contains_context(self, db, store):
        """The 'salience' prompt embeds the context text and references high-priority memories."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        context = 'High-priority memories:\n- key: value'
        prompt = svc._build_prompt('salience', context)

        assert context in prompt
        assert 'high-priority memories' in prompt.lower()
        assert 'DMN_NO_ACTION' in prompt


# ---------------------------------------------------------------------------
# TestDMNProactiveGenerate
# ---------------------------------------------------------------------------

class TestDMNProactiveGenerate:

    def _base_configs(self):
        return {
            'cortex': {
                'config': {'model': 'test'},
                'prompt_map': {},
            }
        }

    def test_proactive_generate_returns_false_on_no_action(self, db, store):
        """_proactive_generate returns False when the LLM responds with DMN_NO_ACTION."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)

        with patch('workers.digest_singletons.load_configs', return_value=self._base_configs()), \
             patch('workers.digest_worker.unified_generate',
                   return_value=({'response': 'DMN_NO_ACTION'}, {})), \
             patch('services.output_service.OutputService') as mock_output_cls:
            result = svc._proactive_generate('recent', 'some context')

        assert result is False
        mock_output_cls.return_value.enqueue_text.assert_not_called()

    def test_proactive_generate_returns_true_on_output(self, db, store):
        """_proactive_generate returns True and enqueues output when the LLM produces a response."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)

        response_data = {
            'response': 'Hello, here is your reminder about the meeting tomorrow.',
            'confidence': 0.85,
            'generation_time': 1.2,
        }

        with patch('workers.digest_singletons.load_configs', return_value=self._base_configs()), \
             patch('workers.digest_worker.unified_generate',
                   return_value=(response_data, {})), \
             patch('services.output_service.OutputService') as mock_output_cls:
            mock_output_instance = MagicMock()
            mock_output_cls.return_value = mock_output_instance
            result = svc._proactive_generate('recent', 'some context')

        assert result is True
        mock_output_instance.enqueue_text.assert_called_once()
        call_kwargs = mock_output_instance.enqueue_text.call_args
        assert call_kwargs.kwargs.get('mode') == 'DMN' or call_kwargs.args[2] == 'DMN' \
            or 'DMN' in str(call_kwargs)

    def test_proactive_generate_catches_exception(self, db, store):
        """_proactive_generate returns False and does not re-raise when unified_generate raises."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)

        with patch('workers.digest_singletons.load_configs', side_effect=RuntimeError('LLM down')):
            result = svc._proactive_generate('recent', 'some context')

        assert result is False


# ---------------------------------------------------------------------------
# TestDMNLogCycle
# ---------------------------------------------------------------------------

class TestDMNLogCycle:

    def test_log_cycle_writes_to_interaction_log(self, db, store):
        """_log_cycle persists a dmn_fired event row in the interaction_log table."""
        from services.database_service import get_shared_db_service

        svc = _make_service(get_shared_db_service(), store)
        svc._log_cycle('recent', produced_output=True)

        row = db.execute(
            "SELECT event_type, payload, source FROM interaction_log "
            "WHERE event_type = 'dmn_fired' LIMIT 1"
        ).fetchone()

        assert row is not None, "Expected a dmn_fired row in interaction_log"
        event_type, payload_json, source = row
        assert event_type == 'dmn_fired'
        assert source == 'dmn'
        payload = json.loads(payload_json)
        assert payload['mode'] == 'recent'
        assert payload['produced_output'] is True

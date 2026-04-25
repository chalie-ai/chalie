"""
Tests for services/innate_skills/introspect_skill.py

Tests use the real `db` fixture (fully-migrated SQLite) for all database
interactions.  Non-DB services (SelfModelService, MemoryClientService) remain
mocked where needed because they pull data from memory stores or perform complex
internal logic that is not under test here.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ── Patch target constants ────────────────────────────────────────────────────

_SELF_MODEL  = 'services.self_model_service.SelfModelService'
_DB          = 'services.database_service.get_shared_db_service'
_MEMORY_CLI  = 'services.memory_client.MemoryClientService'

# A fixed "now" for deterministic relative-time assertions.
_FIXED_NOW = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _invoke(topic='test', params=None):
    from services.innate_skills.introspect_skill import handle_introspect
    return handle_introspect(topic, params or {})


def _seed_episode(db, episode_id='ep-1', updated_at='2026-01-01 10:00:00'):
    """Insert a minimal valid episode row for consolidation-timestamp tests."""
    db.execute(
        "INSERT INTO episodes (id, gist, salience, channel, updated_at) "
        "VALUES (?, 'gist', 5, 'test', ?)",
        (episode_id, updated_at),
    )
    db.commit()



def _seed_scheduled_item(db, message, due_at, status='pending', item_id=None):
    """Insert a scheduled_items row."""
    import uuid
    item_id = item_id or str(uuid.uuid4())
    db.execute(
        "INSERT INTO scheduled_items (id, message, due_at, status) VALUES (?, ?, ?, ?)",
        (item_id, message, due_at, status),
    )
    db.commit()


# ── Tests: top-level output structure ────────────────────────────────────────

@pytest.mark.unit
class TestOutputStructure:

    def test_starts_with_introspect_prefix(self, db, store):
        result = _invoke()
        assert result.startswith('[INTROSPECT]')

    def test_returns_all_three_scope_headers(self, db, store):
        result = _invoke()
        assert '## Memory Health'   in result
        assert '## Reasoning State' in result
        assert '## Identity'        in result

    def test_sections_appear_in_order(self, db, store):
        result = _invoke()
        positions = [
            result.index('## Memory Health'),
            result.index('## Reasoning State'),
            result.index('## Identity'),
        ]
        assert positions == sorted(positions)

    def test_result_is_string(self, db, store):
        result = _invoke()
        assert isinstance(result, str)


# ── Tests: Memory Health scope ────────────────────────────────────────────────

@pytest.mark.unit
class TestMemoryHealthScope:

    def _make_snapshot(self, episode_count=120, concept_count=45, wm_depth=7):
        return {
            'operational': {
                'memory_pressure': {
                    'episode_count': episode_count,
                    'concept_count': concept_count,
                }
            },
            'epistemic': {
                'working_memory_depth': wm_depth,
            },
        }

    @patch(_SELF_MODEL)
    def test_memory_health_shows_episode_count(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(
            episode_count=300, concept_count=80, wm_depth=5
        )
        _seed_episode(db, updated_at='2026-01-01 10:00:00')

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert '300' in result

    @patch(_SELF_MODEL)
    def test_memory_health_shows_concept_count(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(
            episode_count=300, concept_count=80, wm_depth=5
        )
        _seed_episode(db, updated_at='2026-01-01 10:00:00')

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert '80' in result

    @patch(_SELF_MODEL)
    def test_memory_health_shows_working_memory_depth(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(
            episode_count=300, concept_count=80, wm_depth=5
        )
        _seed_episode(db, updated_at='2026-01-01 10:00:00')

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert '5/12' in result

    @patch(_SELF_MODEL)
    def test_memory_health_label_healthy(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(episode_count=200)

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert 'healthy' in result

    @patch(_SELF_MODEL)
    def test_memory_health_label_developing(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(episode_count=75)

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert 'developing' in result

    @patch(_SELF_MODEL)
    def test_memory_health_label_sparse(self, mock_sms_cls, db):
        mock_sms_cls.return_value.get_snapshot.return_value = self._make_snapshot(episode_count=10)

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert 'sparse' in result

    @patch(_SELF_MODEL)
    def test_memory_health_graceful_degradation(self, mock_sms_cls, db):
        """When SelfModelService raises, the scope must not propagate the exception."""
        mock_sms_cls.return_value.get_snapshot.side_effect = RuntimeError('service down')

        from services.innate_skills.introspect_skill import _scope_memory_health
        result = _scope_memory_health()

        assert result == 'Memory: unavailable.'

    def test_memory_health_graceful_degradation_via_full_output(self, store):
        """Degraded memory scope must not crash the overall introspect output."""
        with patch(_SELF_MODEL, side_effect=Exception('import fail')), \
             patch(_DB, side_effect=Exception('db fail')), \
             patch('services.data_graph_service.get_data_graph_service', side_effect=Exception('dgs fail')):
            result = _invoke()

        assert '[INTROSPECT]'      in result
        assert '## Memory Health'  in result
        assert 'unavailable'       in result


# ── Tests: Reasoning State scope ─────────────────────────────────────────────

@pytest.mark.unit
class TestReasoningStateScope:

    def test_reasoning_state_shows_reminders(self, db):
        # due_at must be in the future relative to SQL datetime('now')
        future1 = (datetime.utcnow() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        future2 = (datetime.utcnow() + timedelta(hours=20)).strftime('%Y-%m-%d %H:%M:%S')
        _seed_scheduled_item(db, 'Team standup call', future1)
        _seed_scheduled_item(db, 'Pay rent', future2)

        from services.innate_skills.introspect_skill import _reasoning_upcoming_reminders
        result = _reasoning_upcoming_reminders()

        assert 'Team standup call' in result

    def test_reasoning_state_reminder_count_singular(self, db):
        future = (datetime.utcnow() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
        _seed_scheduled_item(db, 'Dentist appointment', future)

        from services.innate_skills.introspect_skill import _reasoning_upcoming_reminders
        result = _reasoning_upcoming_reminders()

        assert '1 upcoming reminder' in result

    def test_reasoning_state_reminder_count_plural(self, db):
        future1 = (datetime.utcnow() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
        future2 = (datetime.utcnow() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')
        _seed_scheduled_item(db, 'First reminder', future1)
        _seed_scheduled_item(db, 'Second reminder', future2)

        from services.innate_skills.introspect_skill import _reasoning_upcoming_reminders
        result = _reasoning_upcoming_reminders()

        assert '2 upcoming reminders' in result

    def test_reasoning_state_no_reminders_returns_empty(self, db):
        from services.innate_skills.introspect_skill import _reasoning_upcoming_reminders
        result = _reasoning_upcoming_reminders()

        assert result == ''

    def test_reasoning_state_no_active_state_fallback_text(self, db):
        """When all sub-parts are empty, the scope returns the fallback sentence."""
        from services.innate_skills.introspect_skill import _scope_reasoning_state
        result = _scope_reasoning_state()

        assert 'No active reasoning state.' in result


# ── Tests: Identity scope ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestIdentityScope:

    @patch('services.data_graph_service.get_data_graph_service')
    def test_identity_communication_style_verbosity(self, mock_dgs_fn, db):
        mock_dgs = MagicMock()
        mock_dgs.fetch.return_value = [
            {'key': 'communication_style_verbosity', 'value': '9'},
            {'key': 'communication_style_directness', 'value': '5'},
            {'key': 'communication_style_formality', 'value': '5'},
        ]
        mock_dgs_fn.return_value = mock_dgs

        from services.innate_skills.introspect_skill import _identity_communication_style
        result = _identity_communication_style()

        assert 'very verbose' in result

    @patch('services.data_graph_service.get_data_graph_service')
    def test_identity_communication_style_directness(self, mock_dgs_fn, db):
        mock_dgs = MagicMock()
        mock_dgs.fetch.return_value = [
            {'key': 'communication_style_verbosity', 'value': '5'},
            {'key': 'communication_style_directness', 'value': '9'},
            {'key': 'communication_style_formality', 'value': '5'},
        ]
        mock_dgs_fn.return_value = mock_dgs

        from services.innate_skills.introspect_skill import _identity_communication_style
        result = _identity_communication_style()

        assert 'very direct' in result

    @patch('services.data_graph_service.get_data_graph_service')
    def test_identity_communication_style_formality(self, mock_dgs_fn, db):
        mock_dgs = MagicMock()
        mock_dgs.fetch.return_value = [
            {'key': 'communication_style_verbosity', 'value': '5'},
            {'key': 'communication_style_directness', 'value': '5'},
            {'key': 'communication_style_formality', 'value': '9'},
        ]
        mock_dgs_fn.return_value = mock_dgs

        from services.innate_skills.introspect_skill import _identity_communication_style
        result = _identity_communication_style()

        assert 'formal' in result

    @patch('services.data_graph_service.get_data_graph_service', side_effect=RuntimeError('dgs fail'))
    @patch(_DB)
    def test_identity_graceful_degradation(
        self, mock_db_fn, mock_dgs_fn
    ):
        """When every identity sub-service fails the scope returns the fallback string."""
        mock_db_fn.return_value = MagicMock(
            connection=MagicMock(
                side_effect=RuntimeError('connection refused')
            )
        )

        from services.innate_skills.introspect_skill import _scope_identity_snapshot
        result = _scope_identity_snapshot()

        assert result == 'Identity data unavailable.'


# ── Tests: _relative_time helper ─────────────────────────────────────────────

@pytest.mark.unit
class TestRelativeTimeFormatting:
    """TimeFormatterService.ago() is now the shared chokepoint; test it via the service."""

    def _call(self, dt_value, now=_FIXED_NOW):
        with patch('services.time_formatter_service.utc_now', return_value=now):
            from services.time_formatter_service import TimeFormatterService
            return TimeFormatterService.ago(dt_value)

    def test_seconds_ago(self):
        ts = (_FIXED_NOW - timedelta(seconds=30)).isoformat()
        result = self._call(ts)
        assert result == '30s ago'

    def test_minutes_ago(self):
        ts = (_FIXED_NOW - timedelta(minutes=15)).isoformat()
        result = self._call(ts)
        assert result == '15m ago'

    def test_one_hour_ago(self):
        ts = (_FIXED_NOW - timedelta(hours=1)).isoformat()
        result = self._call(ts)
        assert result == '1h 0m ago'

    def test_multiple_hours_ago(self):
        ts = (_FIXED_NOW - timedelta(hours=5)).isoformat()
        result = self._call(ts)
        assert result == '5h 0m ago'

    def test_one_day_ago(self):
        ts = (_FIXED_NOW - timedelta(days=1)).isoformat()
        result = self._call(ts)
        assert result == '1d 0h ago'

    def test_multiple_days_ago(self):
        ts = (_FIXED_NOW - timedelta(days=4)).isoformat()
        result = self._call(ts)
        assert result == '4d 0h ago'

    def test_future_datetime_clamps_to_zero(self):
        ts = (_FIXED_NOW + timedelta(hours=2)).isoformat()
        result = self._call(ts)
        assert result == '0s ago'

    def test_naive_sqlite_string(self):
        """SQLite-style naive timestamp strings must be handled without error."""
        ts = '2026-03-19 11:00:00'   # 1 hour before _FIXED_NOW, no tz suffix
        result = self._call(ts)
        assert result == '1h 0m ago'

    def test_unparseable_value_does_not_crash(self):
        result = self._call('not-a-date')
        assert isinstance(result, str) and len(result) > 0

    def test_none_value_does_not_crash(self):
        result = self._call(None)
        assert isinstance(result, str) and len(result) > 0

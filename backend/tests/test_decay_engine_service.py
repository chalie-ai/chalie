"""Tests for DecayEngineService — periodic decay across all memory types."""

import pytest
from unittest.mock import patch

from services.decay_engine_service import DecayEngineService


pytestmark = pytest.mark.unit


class TestDecayEngineService:
    """Tests for DecayEngineService construction and decay cycle."""

    # ── Constructor / Configuration ───────────────────────────────────

    def test_constructor_loads_config_rates(self):
        """Constructor should load decay rates from ConfigService."""
        mock_config = {
            'retrieval_decay_exponent': 0.08,
        }
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value=mock_config,
        ):
            svc = DecayEngineService(decay_interval=600)

        assert svc.retrieval_decay_exponent == 0.08
        assert svc.decay_interval == 600

    def test_constructor_uses_defaults_on_config_failure(self):
        """When ConfigService raises, default decay rates should be used."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            side_effect=Exception('config unavailable'),
        ):
            svc = DecayEngineService()

        assert svc.retrieval_decay_exponent == 0.5

    def test_default_decay_interval(self):
        """Default decay interval should be 1800 seconds (30 minutes)."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        assert svc.decay_interval == 1800

    # ── run_decay_cycle — richness gating ────────────────────────────

    def test_run_decay_cycle_low_richness_skips_identity_and_external(self):
        """With richness < 0.3, identity inertia and external knowledge decay must not run.

        We verify actual gating behaviour — identity/external sub-cycles return 0
        without being called — not just that internal helper methods were invoked.
        """
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        call_log = []

        def _record(name, retval):
            def _fn(*args, **kwargs):
                call_log.append(name)
                return retval
            return _fn

        with patch.object(svc, '_decay_episodic',                   side_effect=_record('episodic', 2)), \
             patch.object(svc, '_process_pending_reconsolidation',   side_effect=_record('reconsolidation', 0)), \
             patch.object(svc, '_run_episode_consolidation',          side_effect=_record('consolidation', 0)), \
             patch.object(svc, '_decay_knowledge',                   side_effect=_record('knowledge', 1)), \
             patch.object(svc, '_decay_goals',                       side_effect=_record('goals', 0)), \
             patch.object(svc, '_cleanup_transcript',                side_effect=_record('transcript', 0)), \
             patch.object(svc, '_purge_tool_calls',                  side_effect=_record('tool_calls', 0)), \
             patch.object(svc, '_apply_identity_inertia',            side_effect=_record('identity', 3)), \
             patch.object(svc, '_decay_external_knowledge',          side_effect=_record('external', 1)):

            svc.run_decay_cycle(richness=0.1)  # below 0.3 gate

        # Essential cycles ran
        assert 'episodic'  in call_log
        assert 'knowledge' in call_log
        assert 'goals'     in call_log
        # Non-essential cycles were skipped
        assert 'identity' not in call_log
        assert 'external' not in call_log

    def test_run_decay_cycle_normal_richness_runs_all_cycles(self):
        """With richness >= 0.3, all decay sub-cycles including identity must run."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        call_log = []

        def _record(name, retval):
            def _fn(*args, **kwargs):
                call_log.append(name)
                return retval
            return _fn

        with patch.object(svc, '_decay_episodic',                   side_effect=_record('episodic', 5)), \
             patch.object(svc, '_process_pending_reconsolidation',   side_effect=_record('reconsolidation', 0)), \
             patch.object(svc, '_run_episode_consolidation',          side_effect=_record('consolidation', 0)), \
             patch.object(svc, '_decay_knowledge',                   side_effect=_record('knowledge', 3)), \
             patch.object(svc, '_decay_goals',                       side_effect=_record('goals', 0)), \
             patch.object(svc, '_cleanup_transcript',                side_effect=_record('transcript', 0)), \
             patch.object(svc, '_purge_tool_calls',                  side_effect=_record('tool_calls', 0)), \
             patch.object(svc, '_apply_identity_inertia',            side_effect=_record('identity', 2)), \
             patch.object(svc, '_decay_external_knowledge',          side_effect=_record('external', 1)):

            svc.run_decay_cycle(richness=1.0)  # above 0.3 gate

        # All cycles must have run
        for name in ('episodic', 'knowledge', 'goals', 'identity', 'external'):
            assert name in call_log, f"Expected '{name}' to run at full richness"

    # ── Individual decay targets ──────────────────────────────────────

    def test_decay_episodic_returns_zero_on_db_failure(self):
        """Episodic decay should return 0 when DB is unavailable."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        with patch(
            'services.database_service.get_shared_db_service',
            side_effect=Exception('DB unavailable'),
        ):
            result = svc._decay_episodic()

        assert result == 0

    def test_decay_knowledge_returns_zero_on_db_failure(self):
        """Knowledge decay should return 0 when DB is unavailable."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        with patch(
            'services.database_service.get_shared_db_service',
            side_effect=Exception('DB unavailable'),
        ):
            result = svc._decay_knowledge()

        assert result == 0


    def test_apply_identity_inertia_returns_zero_on_failure(self):
        """Identity inertia should return 0 on any exception."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        with patch(
            'services.database_service.get_shared_db_service',
            side_effect=Exception('DB down'),
        ):
            result = svc._apply_identity_inertia()

        assert result == 0

    # ── Real DB: _decay_episodic ──────────────────────────────────────

    def test_decay_episodic_reduces_retrieval_weight_for_old_episodes(self, db):
        """Episodic decay applies power-law decay to episodes last accessed > 1 hour ago.

        Seeds an episode with last_accessed_at 2 hours in the past and retrieval_weight
        of 0.9. After _decay_episodic(), the weight must decrease and the return
        value must be 1.
        """
        # Insert a stale episode — last accessed 2 hours ago so it qualifies for decay
        db.execute(
            "INSERT INTO episodes "
            "(id, intent, context, action, emotion, outcome, gist, salience, channel, "
            " retrieval_weight, last_accessed_at) "
            "VALUES ('ep-decay-1', '{}', '{}', 'test', '{}', 'ok', 'test gist', 5, 'ch:1', "
            "  0.9, datetime('now', '-2 hours'))",
        )
        db.commit()
        # Read the initial weight so we can compare after decay
        initial_weight = db.execute(
            "SELECT retrieval_weight FROM episodes WHERE id = 'ep-decay-1'"
        ).fetchone()[0]
        assert initial_weight == pytest.approx(0.9)

        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={'retrieval_decay_exponent': 0.5},
        ):
            svc = DecayEngineService()

        updated_count = svc._decay_episodic()

        # _decay_episodic closes the shared pool — get a fresh connection to verify
        from services.database_service import get_shared_db_service
        new_conn = get_shared_db_service()._get_connection()
        row = new_conn.execute(
            "SELECT retrieval_weight FROM episodes WHERE id = 'ep-decay-1'"
        ).fetchone()

        assert updated_count == 1
        assert row is not None
        # Weight must have dropped from 0.9 but stay above the 0.01 floor
        assert 0.01 <= row[0] < initial_weight

    def test_decay_episodic_skips_recently_accessed_episodes(self, db):
        """Episodes accessed within the last hour are not decayed.

        The WHERE clause filters to last_accessed_at < now - 1h, so a fresh
        episode should produce updated_count == 0.
        """
        db.execute(
            "INSERT INTO episodes "
            "(id, intent, context, action, emotion, outcome, gist, salience, channel, "
            " retrieval_weight, last_accessed_at) "
            "VALUES ('ep-fresh-1', '{}', '{}', 'fresh', '{}', 'ok', 'fresh gist', 5, 'ch:1', "
            "  0.9, datetime('now', '-10 minutes'))",
        )
        db.commit()

        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        updated_count = svc._decay_episodic()

        assert updated_count == 0

    # ── Real DB: _purge_tool_calls ────────────────────────────────────

    def test_purge_tool_calls_removes_excess_rows(self, db):
        """_purge_tool_calls deletes oldest rows when count exceeds max_rows.

        Seeds 5 transcript + tool_calls rows, then calls _purge_tool_calls with
        max_rows=3. Expects 2 rows deleted and 3 remaining.
        """
        # Seed transcript rows first (required by FK on tool_calls)
        for i in range(5):
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('ch:1', 'user', 'msg', datetime('now', ? || ' seconds'))",
                (str(i * 10),),
            )
        db.commit()

        # Fetch the auto-generated transcript IDs
        transcript_ids = [
            r[0] for r in db.execute("SELECT id FROM transcript ORDER BY id").fetchall()
        ]

        # Seed tool_calls — one per transcript row
        for tid in transcript_ids:
            db.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, invoked_by, created_at) "
                "VALUES (?, 'memory', 'llm', datetime('now'))",
                (tid,),
            )
        db.commit()

        # Verify 5 rows before purge
        count_before = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        assert count_before == 5

        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        purged = svc._purge_tool_calls(max_rows=3)

        assert purged == 2  # 5 - 3 = 2 excess rows deleted

        # _purge_tool_calls uses the shared DB service — get a fresh connection to verify
        from services.database_service import get_shared_db_service
        new_conn = get_shared_db_service()._get_connection()
        count_after = new_conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        assert count_after == 3

    def test_purge_tool_calls_no_op_when_within_limit(self, db):
        """_purge_tool_calls returns 0 when row count is at or below max_rows."""
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('ch:1', 'user', 'msg')"
        )
        db.commit()
        tid = db.execute("SELECT id FROM transcript ORDER BY id DESC LIMIT 1").fetchone()[0]
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, invoked_by) "
            "VALUES (?, 'memory', 'llm')",
            (tid,),
        )
        db.commit()

        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        purged = svc._purge_tool_calls(max_rows=1000)
        assert purged == 0

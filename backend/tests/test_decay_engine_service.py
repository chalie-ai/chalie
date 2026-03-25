"""Tests for DecayEngineService — periodic decay across all memory types."""

import pytest
from unittest.mock import patch, MagicMock

from services.decay_engine_service import DecayEngineService


pytestmark = pytest.mark.unit


class TestDecayEngineService:
    """Tests for DecayEngineService construction and decay cycle."""

    # ── Constructor / Configuration ───────────────────────────────────

    def test_constructor_loads_config_rates(self):
        """Constructor should load decay rates from ConfigService."""
        mock_config = {
            'episodic_decay_rate': 0.08,
        }
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value=mock_config,
        ):
            svc = DecayEngineService(decay_interval=600)

        assert svc.episodic_decay_rate == 0.08
        assert svc.decay_interval == 600

    def test_constructor_uses_defaults_on_config_failure(self):
        """When ConfigService raises, default decay rates should be used."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            side_effect=Exception('config unavailable'),
        ):
            svc = DecayEngineService()

        assert svc.episodic_decay_rate == 0.05

    def test_default_decay_interval(self):
        """Default decay interval should be 1800 seconds (30 minutes)."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        assert svc.decay_interval == 1800

    # ── run_decay_cycle ───────────────────────────────────────────────

    def test_run_decay_cycle_calls_all_decay_targets(self):
        """run_decay_cycle should call episodic, knowledge, goals, identity, and external decay."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        with patch.object(svc, '_decay_episodic', return_value=5) as mock_ep, \
             patch.object(svc, '_decay_knowledge', return_value=3) as mock_know, \
             patch.object(svc, '_decay_goals', return_value=0) as mock_goals, \
             patch.object(svc, '_apply_identity_inertia', return_value=2) as mock_id, \
             patch.object(svc, '_decay_external_knowledge', return_value=1) as mock_ext:

            svc.run_decay_cycle()

        mock_ep.assert_called_once()
        mock_know.assert_called_once()
        mock_goals.assert_called_once()
        mock_id.assert_called_once()
        mock_ext.assert_called_once()

    # ── Individual decay targets ──────────────────────────────────────

    def test_decay_episodic_returns_zero_on_db_failure(self):
        """Episodic decay should return 0 when DB is unavailable."""
        with patch(
            'services.decay_engine_service.ConfigService.get_agent_config',
            return_value={},
        ):
            svc = DecayEngineService()

        with patch(
            'services.database_service.get_lightweight_db_service',
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
            'services.database_service.get_lightweight_db_service',
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
            'services.database_service.get_lightweight_db_service',
            side_effect=Exception('DB down'),
        ):
            result = svc._apply_identity_inertia()

        assert result == 0

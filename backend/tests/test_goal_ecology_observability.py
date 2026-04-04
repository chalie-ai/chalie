"""Tests for H1.0 — Goal ecology observability and metrics."""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestDriftEngineObservability:
    """Drift engine writes MemoryStore counters after each cycle."""

    def test_cycle_writes_last_run_timestamp(self, mock_store):
        """After a cycle, goal_ecology:last_run should be set."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.decay_unreinforced.return_value = 0
            svc.get_actionable_goals.return_value = []

            from services.cognitive_drift_engine import CognitiveDriftEngine
            engine = CognitiveDriftEngine(check_interval=300)
            engine._run_goal_ecology_cycle()

        last_run = mock_store.get("goal_ecology:last_run")
        assert last_run is not None
        assert "T" in last_run  # ISO format

    def test_cycle_increments_cycles_total(self, mock_store):
        """Each cycle increments goal_ecology:cycles_total."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.decay_unreinforced.return_value = 0
            svc.get_actionable_goals.return_value = []

            from services.cognitive_drift_engine import CognitiveDriftEngine
            engine = CognitiveDriftEngine(check_interval=300)
            engine._run_goal_ecology_cycle()
            engine._run_goal_ecology_cycle()

        assert int(mock_store.get("goal_ecology:cycles_total") or 0) == 2

    def test_decayed_goals_counted(self, mock_store):
        """Decayed goals are tracked in goals_decayed_total."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.decay_unreinforced.return_value = 3
            svc.get_actionable_goals.return_value = []

            from services.cognitive_drift_engine import CognitiveDriftEngine
            engine = CognitiveDriftEngine(check_interval=300)
            engine._run_goal_ecology_cycle()

        assert int(mock_store.get("goal_ecology:goals_decayed_total") or 0) == 3

    def test_proactive_attempts_counted(self, mock_store):
        """Actionable goals trigger proactive_attempts_total counter."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology, \
             patch('services.goal_proactive_service.check_and_execute'):
            svc = MockEcology.return_value
            svc.decay_unreinforced.return_value = 0
            svc.get_actionable_goals.return_value = [{'id': 'g1', 'salience': 0.9}]

            from services.cognitive_drift_engine import CognitiveDriftEngine
            engine = CognitiveDriftEngine(check_interval=300)
            engine._run_goal_ecology_cycle()

        assert int(mock_store.get("goal_ecology:proactive_attempts_total") or 0) == 1

    def test_no_counters_on_empty_cycle(self, mock_store):
        """Empty cycle (nothing decayed, nothing actionable) only writes last_run + cycles_total."""
        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value
            svc.decay_unreinforced.return_value = 0
            svc.get_actionable_goals.return_value = []

            from services.cognitive_drift_engine import CognitiveDriftEngine
            engine = CognitiveDriftEngine(check_interval=300)
            engine._run_goal_ecology_cycle()

        assert mock_store.get("goal_ecology:last_run") is not None
        assert int(mock_store.get("goal_ecology:cycles_total") or 0) == 1
        assert mock_store.get("goal_ecology:goals_decayed_total") is None
        assert mock_store.get("goal_ecology:goals_created_total") is None
        assert mock_store.get("goal_ecology:proactive_attempts_total") is None

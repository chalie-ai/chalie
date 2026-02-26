"""Unit tests for FrontalCortexService._get_onboarding_nudge."""

import json
import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service_instance():
    """Return a bare FrontalCortexService instance with no __init__ side-effects."""
    from services.frontal_cortex_service import FrontalCortexService
    return object.__new__(FrontalCortexService)


def _mock_identity_and_redis(identity_blob: dict, exchange_count: int, thread_id: str = "t1"):
    """
    Context manager that patches IdentityStateService and RedisClientService
    for onboarding nudge tests.

    Both are imported locally inside _get_onboarding_nudge, so we patch them
    at their source modules.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch('services.identity_state_service.IdentityStateService') as mock_id_cls, \
             patch('services.redis_client.RedisClientService') as mock_redis_cls:

            mock_id = MagicMock()
            mock_id.get_all.return_value = identity_blob
            mock_id.set_onboarding_state.return_value = True
            mock_id_cls.return_value = mock_id

            mock_r = MagicMock()
            mock_r.hget.return_value = str(exchange_count)
            mock_redis_cls.create_connection.return_value = mock_r

            yield mock_id, mock_r

    return _ctx()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOnboardingNudge:

    def test_no_nudge_below_min_turn(self):
        """No nudge when exchange_count < min_turn (3)."""
        svc = _make_service_instance()

        with _mock_identity_and_redis({}, exchange_count=2):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_nudge_when_name_already_set(self):
        """No nudge when IdentityStateService already has a name value."""
        svc = _make_service_instance()
        identity = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []}
        }

        with _mock_identity_and_redis(identity, exchange_count=10):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_first_nudge_at_min_turn(self):
        """Nudge emitted at exchange_count == min_turn (5) with no name set."""
        svc = _make_service_instance()

        with _mock_identity_and_redis({}, exchange_count=5) as (mock_id, _):
            result = svc._get_onboarding_nudge("t1")

        assert result != ""
        assert "Onboarding note" in result
        # Onboarding state should be updated
        mock_id.set_onboarding_state.assert_called_once()
        call_arg = mock_id.set_onboarding_state.call_args[0][0]
        assert call_arg['name']['attempts'] == 1

    def test_no_nudge_during_cooldown(self):
        """No nudge at exchange_count=6 when last nudge was at turn 5 (cooldown=8)."""
        svc = _make_service_instance()
        identity = {
            '_onboarding': {'name': {'nudged_at_turn': 5, 'attempts': 1}},
        }

        with _mock_identity_and_redis(identity, exchange_count=6):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_second_nudge_after_cooldown(self):
        """Second nudge fires when cooldown (8 turns) has elapsed."""
        svc = _make_service_instance()
        # Last nudge at turn 5, cooldown=8, so turn 13 should work
        identity = {
            '_onboarding': {'name': {'nudged_at_turn': 5, 'attempts': 1}},
        }

        with _mock_identity_and_redis(identity, exchange_count=13) as (mock_id, _):
            result = svc._get_onboarding_nudge("t1")

        assert result != ""
        call_arg = mock_id.set_onboarding_state.call_args[0][0]
        assert call_arg['name']['attempts'] == 2

    def test_no_nudge_after_max_attempts(self):
        """No nudge after max_attempts reached for all scheduled traits."""
        svc = _make_service_instance()
        identity = {
            '_onboarding': {
                'name': {'nudged_at_turn': 13, 'attempts': 2},
                'timezone': {'nudged_at_turn': 20, 'attempts': 1},
                'interests': {'nudged_at_turn': 30, 'attempts': 1},
            },
        }

        with _mock_identity_and_redis(identity, exchange_count=50):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_nudge_with_none_thread_id(self):
        """Returns '' when thread_id is None."""
        svc = _make_service_instance()
        result = svc._get_onboarding_nudge(None)
        assert result == ""

    def test_no_nudge_when_needs_tools(self):
        """Returns '' when classification signals tool use (needs_tools=True)."""
        svc = _make_service_instance()

        with _mock_identity_and_redis({}, exchange_count=10):
            result = svc._get_onboarding_nudge("t1", {'needs_tools': True})

        assert result == ""

    def test_no_nudge_when_urgency_high(self):
        """Returns '' when classification urgency is 'high'."""
        svc = _make_service_instance()

        with _mock_identity_and_redis({}, exchange_count=10):
            result = svc._get_onboarding_nudge("t1", {'urgency': 'high'})

        assert result == ""

    def test_returns_empty_on_redis_error(self):
        """Redis error → returns '', does not raise."""
        svc = _make_service_instance()

        with patch('services.redis_client.RedisClientService') as mock_redis_cls, \
             patch('services.identity_state_service.IdentityStateService'):
            mock_redis_cls.create_connection.side_effect = ConnectionError("Redis down")
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

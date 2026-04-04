"""Unit tests for FrontalCortexService._get_onboarding_nudge."""

import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service_instance():
    """Return a bare PromptAssemblyService instance with no __init__ side-effects."""
    from services.prompt_assembly_service import PromptAssemblyService
    return object.__new__(PromptAssemblyService)


def _mock_identity_and_store(
    identity_blob: dict,
    exchange_count: int,
    thread_id: str = "t1",
    user_traits: list = None,
):
    """
    Context manager that patches IdentityStateService, MemoryClientService,
    and KnowledgeService for onboarding nudge tests.

    All are imported locally inside _get_onboarding_nudge, so we patch them
    at their source modules.  The store is a real ``MemoryStore`` instance
    pre-populated with the exchange count hash so the service can read it
    without any mock side-effects.

    Args:
        identity_blob: dict returned by IdentityStateService.get_all().
        exchange_count: integer turn count written into the store hash for
            ``thread:<thread_id>`` / ``exchange_count`` field.
        thread_id: thread identifier used as the hash key prefix (default "t1").
        user_traits: list of knowledge dicts from KnowledgeService.get_by_kind().
                     Defaults to [] (no traits in permanent storage).

    Yields:
        tuple[MagicMock, MemoryStore]: the mocked identity service instance
        and the pre-populated real MemoryStore.
    """
    import contextlib
    from services.memory_store import MemoryStore

    if user_traits is None:
        user_traits = []

    @contextlib.contextmanager
    def _ctx():
        _store = MemoryStore()
        _store.hset(f"thread:{thread_id}", "exchange_count", str(exchange_count))

        with patch('services.identity_state_service.IdentityStateService') as mock_id_cls, \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=_store), \
             patch('services.knowledge_service.KnowledgeService') as mock_ks_cls:

            mock_id = MagicMock()
            mock_id.get_all.return_value = identity_blob
            mock_id.set_onboarding_state.return_value = True
            mock_id_cls.return_value = mock_id

            mock_ks = MagicMock()
            mock_ks.get_by_kind.return_value = user_traits
            mock_ks_cls.return_value = mock_ks

            yield mock_id, _store

    return _ctx()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOnboardingNudge:

    def test_no_nudge_below_min_turn(self, db):
        """No nudge when exchange_count < min_turn (3)."""
        svc = _make_service_instance()

        with _mock_identity_and_store({}, exchange_count=2):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_nudge_when_name_already_set(self, db):
        """No trait nudge when IdentityStateService already has a name value."""
        svc = _make_service_instance()
        identity = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []}
        }

        # exchange_count=4: only name (min_turn=3) is eligible, but it's set → no trait nudge.
        # Patch capabilities as connected so no capability nudge fires either.
        mock_cap = MagicMock()
        mock_cap._connected = True
        with _mock_identity_and_store(identity, exchange_count=4), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_first_nudge_at_min_turn(self, db):
        """Nudge emitted at exchange_count == min_turn (3) with no name set."""
        svc = _make_service_instance()

        with _mock_identity_and_store({}, exchange_count=3) as (mock_id, _):
            result = svc._get_onboarding_nudge("t1")

        assert result != ""
        assert "Onboarding note" in result
        # Onboarding state should be updated
        mock_id.set_onboarding_state.assert_called_once()
        call_arg = mock_id.set_onboarding_state.call_args[0][0]
        assert call_arg['name']['attempts'] == 1

    def test_no_nudge_during_cooldown(self, db):
        """No nudge at exchange_count=4 when last nudge was at turn 3 (cooldown=6, age_range min_turn=5 not yet eligible)."""
        svc = _make_service_instance()
        # name nudged at turn 3; cooldown=6 so next eligible at turn 9.
        # age_range has min_turn=5, so at turn 4 it isn't eligible yet either.
        identity = {
            '_onboarding': {'name': {'nudged_at_turn': 3, 'attempts': 1}},
        }

        # Patch capabilities as connected so no capability nudge fires
        mock_cap = MagicMock()
        mock_cap._connected = True
        with _mock_identity_and_store(identity, exchange_count=4), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_second_nudge_after_cooldown(self, db):
        """Second nudge fires when cooldown (6 turns) has elapsed."""
        svc = _make_service_instance()
        # name nudged at turn 3, cooldown=6, so turn 9 is the earliest eligible.
        # age_range (min_turn=5) is also not yet set → it fires first at turn 9.
        # Mark age_range as already nudged so name gets a chance.
        identity = {
            '_onboarding': {
                'name': {'nudged_at_turn': 3, 'attempts': 1},
                'age_range': {'nudged_at_turn': 5, 'attempts': 1},
            },
        }

        with _mock_identity_and_store(identity, exchange_count=9) as (mock_id, _):
            result = svc._get_onboarding_nudge("t1")

        assert result != ""
        call_arg = mock_id.set_onboarding_state.call_args[0][0]
        assert call_arg['name']['attempts'] == 2

    def test_no_nudge_after_max_attempts(self, db):
        """No nudge after max_attempts reached for all scheduled traits (with capability connected)."""
        svc = _make_service_instance()
        # All 9 traits in _ONBOARDING_SCHEDULE must be exhausted (attempts >= max_attempts).
        identity = {
            '_onboarding': {
                'name':                    {'nudged_at_turn': 9,  'attempts': 2},
                'age_range':               {'nudged_at_turn': 11, 'attempts': 1},
                'occupation':              {'nudged_at_turn': 14, 'attempts': 1},
                'timezone':                {'nudged_at_turn': 21, 'attempts': 1},
                'communication_preference':{'nudged_at_turn': 24, 'attempts': 1},
                'interests':               {'nudged_at_turn': 26, 'attempts': 1},
                'work_schedule':           {'nudged_at_turn': 28, 'attempts': 1},
                'primary_goal':            {'nudged_at_turn': 31, 'attempts': 1},
                'learning_style':          {'nudged_at_turn': 36, 'attempts': 1},
            },
        }

        # Patch capabilities as connected so no capability nudge fires
        mock_cap = MagicMock()
        mock_cap._connected = True
        with _mock_identity_and_store(identity, exchange_count=80), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_nudge_when_trait_in_permanent_storage(self, db):
        """No name nudge when IdentityStateService expired but knowledge table has the name."""
        svc = _make_service_instance()
        # IdentityStateService has no name (TTL expired), but knowledge table does
        user_traits = [
            {'key': 'name', 'value': 'Dylan', 'confidence': 0.95, 'kind': 'trait'},
        ]

        # exchange_count=4: name (min_turn=3) eligible but in knowledge → skipped
        # age_range (min_turn=5) not yet eligible → no trait nudge at all
        # Patch capabilities as connected so no capability nudge fires
        mock_cap = MagicMock()
        mock_cap._connected = True
        with _mock_identity_and_store({}, exchange_count=4, user_traits=user_traits), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_nudge_with_none_thread_id(self, db):
        """Returns '' when thread_id is None."""
        svc = _make_service_instance()
        result = svc._get_onboarding_nudge(None)
        assert result == ""

    def test_no_nudge_when_needs_tools(self, db):
        """Returns '' when classification signals tool use (needs_tools=True)."""
        svc = _make_service_instance()

        with _mock_identity_and_store({}, exchange_count=10):
            result = svc._get_onboarding_nudge("t1", {'needs_tools': True})

        assert result == ""

    def test_no_nudge_when_urgency_high(self, db):
        """Returns '' when classification urgency is 'high'."""
        svc = _make_service_instance()

        with _mock_identity_and_store({}, exchange_count=10):
            result = svc._get_onboarding_nudge("t1", {'urgency': 'high'})

        assert result == ""

    def test_returns_empty_on_store_error(self, db):
        """MemoryStore error -- returns '', does not raise."""
        svc = _make_service_instance()

        with patch('services.memory_client.MemoryClientService') as mock_store_cls, \
             patch('services.identity_state_service.IdentityStateService'):
            mock_store_cls.create_connection.side_effect = ConnectionError("MemoryStore down")
            result = svc._get_onboarding_nudge("t1")

        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# Capability Connect Nudge
# ─────────────────────────────────────────────────────────────────────────────


def _all_traits_exhausted():
    """Return onboarding state with all trait nudges at max_attempts."""
    return {
        '_onboarding': {
            'name':                    {'nudged_at_turn': 9,  'attempts': 2},
            'age_range':               {'nudged_at_turn': 11, 'attempts': 1},
            'occupation':              {'nudged_at_turn': 14, 'attempts': 1},
            'timezone':                {'nudged_at_turn': 21, 'attempts': 1},
            'communication_preference':{'nudged_at_turn': 24, 'attempts': 1},
            'interests':               {'nudged_at_turn': 26, 'attempts': 1},
            'work_schedule':           {'nudged_at_turn': 28, 'attempts': 1},
            'primary_goal':            {'nudged_at_turn': 31, 'attempts': 1},
            'learning_style':          {'nudged_at_turn': 36, 'attempts': 1},
        },
    }


class TestCapabilityConnectNudge:

    def test_capability_nudge_when_no_caps_connected(self, db):
        """Capability nudge fires when all traits exhausted and no capability connected."""
        svc = _make_service_instance()
        identity = _all_traits_exhausted()

        mock_cap = MagicMock()
        mock_cap._connected = False
        mock_cap.get_id.return_value = 'caldav'

        mock_tcs = MagicMock()
        mock_tcs.get_tool_config.return_value = {}

        with _mock_identity_and_store(identity, exchange_count=80) as (mock_id, _), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_tcs):
            result = svc._get_onboarding_nudge("t1")

        assert "Onboarding note" in result
        assert "life integrations" in result.lower() or "calendar" in result.lower()

    def test_no_capability_nudge_when_cap_connected(self, db):
        """No capability nudge when a capability is connected."""
        svc = _make_service_instance()
        identity = _all_traits_exhausted()

        mock_cap = MagicMock()
        mock_cap._connected = True

        with _mock_identity_and_store(identity, exchange_count=80), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_no_capability_nudge_when_cap_has_stored_credentials(self, db):
        """No capability nudge when credentials are stored (even if not connected yet)."""
        svc = _make_service_instance()
        identity = _all_traits_exhausted()

        mock_cap = MagicMock()
        mock_cap._connected = False
        mock_cap.get_id.return_value = 'caldav'

        mock_tcs = MagicMock()
        mock_tcs.get_tool_config.return_value = {'caldav:username': 'user@example.com'}

        with _mock_identity_and_store(identity, exchange_count=80), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_tcs):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_capability_nudge_respects_min_turn(self, db):
        """No capability nudge before min_turn even if no capabilities connected."""
        svc = _make_service_instance()

        mock_cap = MagicMock()
        mock_cap._connected = False
        mock_cap.get_id.return_value = 'caldav'

        mock_tcs = MagicMock()
        mock_tcs.get_tool_config.return_value = {}

        # exchange_count=2 is below both name (min_turn=3) and capability nudge (min_turn=4)
        with _mock_identity_and_store({}, exchange_count=2), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_tcs):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_capability_nudge_respects_max_attempts(self, db):
        """Capability nudge stops after max_attempts."""
        svc = _make_service_instance()
        identity = _all_traits_exhausted()
        identity['_onboarding']['_capability_connect'] = {
            'nudged_at_turn': 40, 'attempts': 2
        }

        mock_cap = MagicMock()
        mock_cap._connected = False
        mock_cap.get_id.return_value = 'caldav'

        mock_tcs = MagicMock()
        mock_tcs.get_tool_config.return_value = {}

        with _mock_identity_and_store(identity, exchange_count=80), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_tcs):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_capability_nudge_respects_cooldown(self, db):
        """Capability nudge doesn't fire during cooldown period."""
        svc = _make_service_instance()
        identity = _all_traits_exhausted()
        # Nudged at turn 40, cooldown=10, so next eligible at turn 50
        identity['_onboarding']['_capability_connect'] = {
            'nudged_at_turn': 40, 'attempts': 1
        }

        mock_cap = MagicMock()
        mock_cap._connected = False
        mock_cap.get_id.return_value = 'caldav'

        mock_tcs = MagicMock()
        mock_tcs.get_tool_config.return_value = {}

        with _mock_identity_and_store(identity, exchange_count=45), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_tcs):
            result = svc._get_onboarding_nudge("t1")

        assert result == ""

    def test_trait_nudge_takes_priority_over_capability_nudge(self, db):
        """Trait nudges fire before capability nudge."""
        svc = _make_service_instance()

        mock_cap = MagicMock()
        mock_cap._connected = False

        # exchange_count=5: name nudge (min_turn=3) fires since no name set
        with _mock_identity_and_store({}, exchange_count=5) as (mock_id, _), \
             patch('capabilities.load_capabilities', return_value={'caldav': mock_cap}):
            result = svc._get_onboarding_nudge("t1")

        # Should get name trait nudge, not capability nudge
        assert "Onboarding note" in result
        assert "name" in result.lower() or "called" in result.lower()

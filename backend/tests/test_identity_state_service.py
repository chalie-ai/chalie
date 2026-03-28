"""Unit tests for IdentityStateService — MemoryStore-backed identity authority."""

import json
import pytest
import time
from unittest.mock import MagicMock, patch

from services.memory_store import MemoryStore
from services.identity_state_service import IdentityStateService


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service(initial_blob: dict = None, broken_store=None):
    """
    Create an IdentityStateService backed by a real MemoryStore.

    Args:
        initial_blob: Optional dict to pre-populate ``identity_state`` key in
            the store before the service is constructed.  Pass ``None`` to
            start with an empty store.
        broken_store: Optional MagicMock to inject instead of a real
            MemoryStore.  Use only for Category-C (error-path) tests where a
            ``side_effect`` is needed to simulate a broken connection.

    Returns:
        tuple[IdentityStateService, MemoryStore | MagicMock, MagicMock]:
            ``(svc, store, mock_cls)`` — the service, the backing store, and
            the patched ``MemoryClientService`` class mock.  The caller must
            re-apply the patch with ``with patch(..., mock_cls):`` around any
            service method call so that ``MemoryClientService.create_connection``
            resolves to *store* inside each method.
    """
    if broken_store is not None:
        store = broken_store
    else:
        store = MemoryStore()
        if initial_blob is not None:
            store.set('identity_state', json.dumps(initial_blob))

    with patch('services.identity_state_service.MemoryClientService') as mock_cls:
        mock_cls.create_connection.return_value = store
        svc = IdentityStateService()

    return svc, store, mock_cls


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityStateServiceSetField:

    def test_set_field_stores_correct_values(self):
        """set_field('name', 'Dylan', 0.95) stores value, normalized, display."""
        svc, store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_field('name', 'Dylan', 0.95)

        assert result is True
        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert field['value'] == 'Dylan'
        assert field['normalized'] == 'dylan'
        assert field['display'] == 'Dylan'
        assert field['confidence'] == 0.95
        assert field['provisional'] is False
        assert field['previous'] == []

    def test_set_field_normalizes_all_lowercase(self):
        """set_field with all-lowercase input → title-case display, lowercase normalized."""
        svc, store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'dylan', 0.95)

        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert field['display'] == 'Dylan'
        assert field['normalized'] == 'dylan'

    def test_set_field_preserves_mixed_case(self):
        """Mixed-case input (e.g., O'Brien) stored as-is, not title-cased."""
        svc, store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', "O'Brien", 0.95)

        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert field['display'] == "O'Brien"
        assert field['normalized'] == "o'brien"

    def test_set_field_previous_populated_on_change(self):
        """On value change, old display value is prepended to previous[]."""
        existing = {
            'name': {
                'value': 'Alice',
                'normalized': 'alice',
                'display': 'Alice',
                'confidence': 0.9,
                'updated_at': 0.0,
                'provisional': False,
                'previous': [],
            }
        }
        svc, store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert 'Alice' in field['previous']
        assert field['display'] == 'Dylan'

    def test_set_field_no_previous_on_same_normalized_value(self):
        """Same normalized value does not add an entry to previous[]."""
        existing = {
            'name': {
                'value': 'Dylan',
                'normalized': 'dylan',
                'display': 'Dylan',
                'confidence': 0.9,
                'updated_at': 0.0,
                'provisional': False,
                'previous': [],
            }
        }
        svc, store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            # Set same name again with different casing
            svc.set_field('name', 'dylan', 0.95)

        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert field['previous'] == []

    def test_set_field_previous_capped_at_max(self):
        """previous[] is capped at MAX_PREVIOUS_HISTORY (5) entries."""
        existing_previous = ['Name1', 'Name2', 'Name3', 'Name4', 'Name5']
        existing = {
            'name': {
                'value': 'Name5',
                'normalized': 'name5',
                'display': 'Name5',
                'confidence': 0.9,
                'updated_at': 0.0,
                'provisional': False,
                'previous': existing_previous,
            }
        }
        svc, store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Name6', 0.95)

        written = json.loads(store.get('identity_state'))
        field = written['name']
        assert len(field['previous']) <= IdentityStateService.MAX_PREVIOUS_HISTORY

    def test_set_field_refreshes_ttl(self):
        """set_field always writes to MemoryStore with a positive TTL on every write."""
        svc, store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        # Key must be present in the store
        assert store.get('identity_state') is not None
        # setex was used, so the key has a positive TTL
        assert 0 < store.ttl('identity_state') <= IdentityStateService.STORE_TTL

    def test_set_field_store_error_returns_false_no_raise(self):
        """MemoryStore error → returns False, does not raise."""
        broken_store = MagicMock()
        broken_store.get.side_effect = ConnectionError("MemoryStore down")
        svc, _, mock_cls = _make_service(broken_store=broken_store)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_field('name', 'Dylan', 0.95)

        assert result is False

    def test_set_field_store_key_is_fixed(self):
        """MemoryStore key is fixed as 'identity_state'."""
        svc, store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        # The value must be stored under the fixed 'identity_state' key
        assert store.get('identity_state') is not None
        written = json.loads(store.get('identity_state'))
        assert 'name' in written


class TestIdentityStateServiceGetAll:

    def test_get_all_returns_empty_on_missing_key(self):
        """Missing MemoryStore key → get_all() returns {}."""
        svc, store, mock_cls = _make_service(initial_blob=None)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert result == {}

    def test_get_all_returns_blob(self):
        """Existing blob is returned as dict."""
        blob = {'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                         'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                         'previous': []}}
        svc, store, mock_cls = _make_service(initial_blob=blob)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert result['name']['display'] == 'Dylan'

    def test_get_all_returns_empty_on_store_error(self):
        """MemoryStore error → returns {}, does not raise."""
        broken_store = MagicMock()
        broken_store.get.side_effect = ConnectionError("MemoryStore down")
        svc, _, mock_cls = _make_service(broken_store=broken_store)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert result == {}

    def test_get_all_returns_onboarding_key(self):
        """_onboarding key is included in get_all() alongside identity fields."""
        blob = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []},
            '_onboarding': {'name': {'nudged_at_turn': 5, 'attempts': 1, 'backed_off': False}},
        }
        svc, store, mock_cls = _make_service(initial_blob=blob)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert '_onboarding' in result
        assert result['_onboarding']['name']['attempts'] == 1


class TestIdentityStateServiceClearField:

    def test_clear_field_removes_target_only(self):
        """clear_field removes only the specified field; others remain intact."""
        blob = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []},
            'timezone': {'value': 'UTC', 'normalized': 'utc', 'display': 'UTC',
                         'confidence': 0.8, 'updated_at': 0.0, 'provisional': False,
                         'previous': []},
        }
        svc, store, mock_cls = _make_service(initial_blob=blob)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.clear_field('name')

        assert result is True
        written = json.loads(store.get('identity_state'))
        assert 'name' not in written
        assert 'timezone' in written

    def test_clear_field_missing_key_returns_true(self):
        """clear_field on missing MemoryStore key returns True (idempotent)."""
        svc, store, mock_cls = _make_service(initial_blob=None)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.clear_field('name')

        assert result is True
        # No data should have been written — the store key stays absent
        assert store.get('identity_state') is None


class TestIdentityStateServiceOnboardingState:

    def test_set_onboarding_state_writes_to_blob(self):
        """set_onboarding_state writes _onboarding key; existing identity fields intact."""
        blob = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []},
        }
        svc, store, mock_cls = _make_service(initial_blob=blob)
        onboarding = {'name': {'nudged_at_turn': 5, 'attempts': 1}}

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_onboarding_state(onboarding)

        assert result is True
        written = json.loads(store.get('identity_state'))
        assert written['_onboarding'] == onboarding
        assert written['name']['display'] == 'Dylan'

    def test_user_state_is_pure_telemetry(self):
        """_get_user_state returns telemetry only, no identity data."""
        with patch('services.client_context_service.ClientContextService') as mock_cc:
            mock_cc.return_value.get.return_value = {
                'timezone': 'Europe/Malta',
                'location_name': 'Malta',
                'device': {'class': 'iPhone'},
            }

            from services.prompt_assembly_service import PromptAssemblyService

            svc = object.__new__(PromptAssemblyService)
            result = svc._get_user_state()

        assert 'Malta' in result
        assert 'iPhone' in result

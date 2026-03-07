"""Unit tests for IdentityStateService — MemoryStore-backed identity authority."""

import json
import pytest
import time
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service(initial_blob: dict = None):
    """Create an IdentityStateService with a mocked MemoryStore connection."""
    with patch('services.identity_state_service.MemoryClientService') as mock_cls:
        mock_store = MagicMock()
        mock_cls.create_connection.return_value = mock_store

        if initial_blob is not None:
            mock_store.get.return_value = json.dumps(initial_blob)
        else:
            mock_store.get.return_value = None

        from services.identity_state_service import IdentityStateService
        svc = IdentityStateService()
        return svc, mock_store, mock_cls


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityStateServiceSetField:

    def test_set_field_stores_correct_values(self):
        """set_field('name', 'Dylan', 0.95) stores value, normalized, display."""
        svc, mock_store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_field('name', 'Dylan', 0.95)

        assert result is True
        # Inspect what was written
        call_args = mock_store.setex.call_args
        written = json.loads(call_args[0][2])
        field = written['name']
        assert field['value'] == 'Dylan'
        assert field['normalized'] == 'dylan'
        assert field['display'] == 'Dylan'
        assert field['confidence'] == 0.95
        assert field['provisional'] is False
        assert field['previous'] == []

    def test_set_field_normalizes_all_lowercase(self):
        """set_field with all-lowercase input → title-case display, lowercase normalized."""
        svc, mock_store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'dylan', 0.95)

        written = json.loads(mock_store.setex.call_args[0][2])
        field = written['name']
        assert field['display'] == 'Dylan'
        assert field['normalized'] == 'dylan'

    def test_set_field_preserves_mixed_case(self):
        """Mixed-case input (e.g., O'Brien) stored as-is, not title-cased."""
        svc, mock_store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', "O'Brien", 0.95)

        written = json.loads(mock_store.setex.call_args[0][2])
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
        svc, mock_store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        written = json.loads(mock_store.setex.call_args[0][2])
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
        svc, mock_store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            # Set same name again with different casing
            svc.set_field('name', 'dylan', 0.95)

        written = json.loads(mock_store.setex.call_args[0][2])
        field = written['name']
        assert field['previous'] == []

    def test_set_field_previous_capped_at_max(self):
        """previous[] is capped at MAX_PREVIOUS_HISTORY (5) entries."""
        from services.identity_state_service import IdentityStateService
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
        svc, mock_store, mock_cls = _make_service(initial_blob=existing)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Name6', 0.95)

        written = json.loads(mock_store.setex.call_args[0][2])
        field = written['name']
        assert len(field['previous']) <= IdentityStateService.MAX_PREVIOUS_HISTORY

    def test_set_field_refreshes_ttl(self):
        """set_field always calls setex (refreshing TTL) on every write."""
        from services.identity_state_service import IdentityStateService
        svc, mock_store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        assert mock_store.setex.called
        call_args = mock_store.setex.call_args[0]
        assert call_args[1] == IdentityStateService.STORE_TTL

    def test_set_field_store_error_returns_false_no_raise(self):
        """MemoryStore error → returns False, does not raise."""
        svc, mock_store, mock_cls = _make_service()
        mock_store.get.side_effect = ConnectionError("MemoryStore down")

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_field('name', 'Dylan', 0.95)

        assert result is False

    def test_set_field_store_key_is_fixed(self):
        """MemoryStore key is fixed as 'identity_state'."""
        svc, mock_store, mock_cls = _make_service()

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            svc.set_field('name', 'Dylan', 0.95)

        call_args = mock_store.setex.call_args[0]
        assert call_args[0] == 'identity_state:primary'


class TestIdentityStateServiceGetAll:

    def test_get_all_returns_empty_on_missing_key(self):
        """Missing MemoryStore key → get_all() returns {}."""
        svc, mock_store, mock_cls = _make_service(initial_blob=None)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert result == {}

    def test_get_all_returns_blob(self):
        """Existing blob is returned as dict."""
        blob = {'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                         'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                         'previous': []}}
        svc, mock_store, mock_cls = _make_service(initial_blob=blob)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.get_all()

        assert result['name']['display'] == 'Dylan'

    def test_get_all_returns_empty_on_store_error(self):
        """MemoryStore error → returns {}, does not raise."""
        svc, mock_store, mock_cls = _make_service()
        mock_store.get.side_effect = ConnectionError("MemoryStore down")

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
        svc, mock_store, mock_cls = _make_service(initial_blob=blob)

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
        svc, mock_store, mock_cls = _make_service(initial_blob=blob)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.clear_field('name')

        assert result is True
        written = json.loads(mock_store.setex.call_args[0][2])
        assert 'name' not in written
        assert 'timezone' in written

    def test_clear_field_missing_key_returns_true(self):
        """clear_field on missing MemoryStore key returns True (idempotent)."""
        svc, mock_store, mock_cls = _make_service(initial_blob=None)

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.clear_field('name')

        assert result is True
        mock_store.setex.assert_not_called()


class TestIdentityStateServiceOnboardingState:

    def test_set_onboarding_state_writes_to_blob(self):
        """set_onboarding_state writes _onboarding key; existing identity fields intact."""
        blob = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []},
        }
        svc, mock_store, mock_cls = _make_service(initial_blob=blob)
        onboarding = {'name': {'nudged_at_turn': 5, 'attempts': 1}}

        with patch('services.identity_state_service.MemoryClientService', mock_cls):
            result = svc.set_onboarding_state(onboarding)

        assert result is True
        written = json.loads(mock_store.setex.call_args[0][2])
        assert written['_onboarding'] == onboarding
        assert written['name']['display'] == 'Dylan'

    def test_identity_context_renders_confirmed(self):
        """_get_identity_context renders 'Known user details' with (confirmed) qualifier."""
        blob = {
            'name': {'value': 'Dylan', 'normalized': 'dylan', 'display': 'Dylan',
                     'confidence': 0.95, 'updated_at': 0.0, 'provisional': False,
                     'previous': []},
        }
        # Patch MemoryClientService where IdentityStateService uses it
        with patch('services.identity_state_service.MemoryClientService') as mock_cls2:
            mock_store = MagicMock()
            mock_cls2.create_connection.return_value = mock_store
            mock_store.get.return_value = json.dumps(blob)

            from services.frontal_cortex_service import FrontalCortexService

            svc = object.__new__(FrontalCortexService)
            result = svc._get_identity_context(
                returning_from_silence=True,
                context_warmth=1.0,
            )

        assert 'Known user details' in result
        assert 'Dylan' in result
        assert '(confirmed)' in result

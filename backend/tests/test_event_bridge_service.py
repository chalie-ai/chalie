"""Tests for EventBridgeService — event pipeline with gating, cooldowns, and aggregation."""

import time
import json
import pytest
from unittest.mock import patch, MagicMock

from services.event_bridge_service import (
    BridgeEvent,
    EventBridgeService,
    _STABILIZED_EVENTS,
    _PRIORITY_ORDER,
    CONFIDENCE_PHRASING,
)


pytestmark = pytest.mark.unit


class TestBridgeEvent:
    """Tests for BridgeEvent dataclass defaults."""

    def test_default_confidence_is_half(self):
        """Default confidence should be 0.5."""
        event = BridgeEvent(event_type='test_event')
        assert event.confidence == 0.5

    def test_default_interrupt_priority_is_normal(self):
        """Default interrupt_priority should be 'normal'."""
        event = BridgeEvent(event_type='test_event')
        assert event.interrupt_priority == 'normal'

    def test_default_silent_is_false(self):
        """Default silent should be False."""
        event = BridgeEvent(event_type='test_event')
        assert event.silent is False

    def test_default_payload_is_empty_dict(self):
        """Default payload should be an empty dict."""
        event = BridgeEvent(event_type='test_event')
        assert event.payload == {}

    def test_default_states_are_none(self):
        """from_state and to_state should default to None."""
        event = BridgeEvent(event_type='test_event')
        assert event.from_state is None
        assert event.to_state is None

    def test_timestamp_auto_populated(self):
        """Timestamp should be auto-populated to roughly current time."""
        before = time.time()
        event = BridgeEvent(event_type='test_event')
        after = time.time()
        assert before <= event.timestamp <= after


class TestEventBridgeConfidenceGate:
    """Tests for confidence threshold gating."""

    def test_low_confidence_event_rejected(self, mock_redis):
        """Events below confidence threshold should be suppressed."""
        config = {'confidence_threshold': 0.6, 'cooldowns': {}}
        with patch('services.event_bridge_service._load_config', return_value=config):
            svc = EventBridgeService()

        event = BridgeEvent(event_type='session_start', confidence=0.3)
        result = svc.submit_event(event)
        assert result is False

    def test_high_confidence_event_passes_gate(self, mock_redis):
        """Events above confidence threshold should pass the gate."""
        config = {
            'confidence_threshold': 0.6,
            'cooldowns': {},
            'interrupt_priority': {},
            'silent_events': [],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = False
            svc = EventBridgeService()

        event = BridgeEvent(event_type='session_start', confidence=0.8)
        result = svc.submit_event(event)
        assert result is True


class TestEventBridgeCooldown:
    """Tests for per-event cooldown enforcement."""

    def test_cooldown_blocks_rapid_refire(self, mock_redis):
        """Second event of same type within cooldown should be blocked."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {'session_start': 300},
            'interrupt_priority': {},
            'silent_events': [],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = False
            svc = EventBridgeService()

        event = BridgeEvent(event_type='session_start', confidence=0.8)

        # First event succeeds and sets cooldown
        first = svc.submit_event(event)
        assert first is True

        # Second event should be blocked by cooldown
        second = svc.submit_event(event)
        assert second is False

    def test_no_cooldown_allows_rapid_refire(self, mock_redis):
        """Events with no configured cooldown should always pass."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {},  # No cooldowns
            'interrupt_priority': {},
            'silent_events': [],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = False
            svc = EventBridgeService()

        event = BridgeEvent(event_type='session_start', confidence=0.8)
        assert svc.submit_event(event) is True
        assert svc.submit_event(event) is True


class TestEventBridgeFocusGate:
    """Tests for focus gate blocking during deep focus."""

    def test_deep_focus_blocks_normal_priority(self, mock_redis):
        """Normal-priority events should be blocked during deep focus."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {},
            'interrupt_priority': {'test_event': 'normal'},
            'silent_events': [],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = True
            svc = EventBridgeService()

            event = BridgeEvent(event_type='test_event', confidence=0.8)
            result = svc.submit_event(event)
            assert result is False

    def test_deep_focus_allows_critical_priority(self, mock_redis):
        """Critical-priority events should bypass the focus gate."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {},
            'interrupt_priority': {'item_due': 'critical'},
            'silent_events': [],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = True
            svc = EventBridgeService()

            event = BridgeEvent(event_type='item_due', confidence=0.9)
            result = svc.submit_event(event)
            assert result is True

    def test_focus_gate_bypass_event_passes(self, mock_redis):
        """Events listed in focus_gate_bypass should pass even during deep focus."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {},
            'interrupt_priority': {},
            'silent_events': [],
            'focus_gate_bypass': ['item_due'],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config), \
             patch('services.ambient_inference_service.AmbientInferenceService') as mock_ambient:
            mock_ambient.return_value.is_user_deep_focus.return_value = True
            svc = EventBridgeService()

            event = BridgeEvent(event_type='item_due', confidence=0.9)
            result = svc.submit_event(event)
            assert result is True


class TestEventBridgeSilentEvents:
    """Tests for silent event handling."""

    def test_silent_event_returns_true_without_routing(self, mock_redis):
        """Silent events should return True but not go through the routing pipeline."""
        config = {
            'confidence_threshold': 0.5,
            'cooldowns': {},
            'interrupt_priority': {},
            'silent_events': ['attention_deepened'],
            'focus_gate_bypass': [],
            'aggregation_window_seconds': 9999,
        }
        with patch('services.event_bridge_service._load_config', return_value=config):
            svc = EventBridgeService()

        event = BridgeEvent(event_type='attention_deepened', confidence=0.8)
        result = svc.submit_event(event)
        assert result is True
        assert event.silent is True


class TestEventBridgePhrasing:
    """Tests for confidence-to-phrasing mapping."""

    def test_high_confidence_phrasing(self, mock_redis):
        """Confidence >= 0.8 should map to 'high' phrasing."""
        config = {}
        with patch('services.event_bridge_service._load_config', return_value=config):
            svc = EventBridgeService()

        assert svc._get_phrasing(0.9) == 'high'
        assert svc._get_phrasing(0.8) == 'high'

    def test_medium_confidence_phrasing(self, mock_redis):
        """Confidence in [0.6, 0.8) should map to 'medium' phrasing."""
        config = {}
        with patch('services.event_bridge_service._load_config', return_value=config):
            svc = EventBridgeService()

        assert svc._get_phrasing(0.7) == 'medium'
        assert svc._get_phrasing(0.6) == 'medium'

    def test_low_confidence_phrasing(self, mock_redis):
        """Confidence < 0.6 should map to 'low' (suppressed) phrasing."""
        config = {}
        with patch('services.event_bridge_service._load_config', return_value=config):
            svc = EventBridgeService()

        assert svc._get_phrasing(0.5) == 'low'
        assert svc._get_phrasing(0.1) == 'low'


class TestEventBridgeStabilization:
    """Tests for jitter-prone event stabilization."""

    def test_stabilized_events_set_is_correct(self):
        """The set of events requiring stabilization should contain expected types."""
        assert 'place_transition' in _STABILIZED_EVENTS
        assert 'attention_shift' in _STABILIZED_EVENTS
        assert 'energy_change' in _STABILIZED_EVENTS

    def test_priority_order_mapping(self):
        """Priority order should rank critical < normal < low."""
        assert _PRIORITY_ORDER['critical'] < _PRIORITY_ORDER['normal']
        assert _PRIORITY_ORDER['normal'] < _PRIORITY_ORDER['low']

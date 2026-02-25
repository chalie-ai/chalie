"""
Event Bridge Service — Connects context changes to autonomous actions.

Event sources:
  - Ambient Inference → place_transition, attention_shift, energy_change
  - Scheduler Service → item_approaching, item_due
  - Client Context → session_start, session_resume, long_idle_end
  - Spark State → phase_transition

Pipeline:
  1. State stabilization (90s window for jitter-prone events)
  2. Confidence threshold gating
  3. Per-event cooldown enforcement
  4. Silent event aggregation (bundle co-occurring events within 60s)
  5. Focus gate check (deep_focus blocks all except critical priority)
  6. Route through existing autonomous action system
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from services.redis_client import RedisClientService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EVENT BRIDGE]"

# Redis key prefixes
_STABILIZATION_PREFIX = "event_bridge:stabilize:"
_COOLDOWN_PREFIX = "event_bridge:cooldown:"
_AGGREGATION_KEY = "event_bridge:pending_bundle"
_AGGREGATION_TTL = 120  # 2min safety TTL

# Events that require stabilization (jitter-prone)
_STABILIZED_EVENTS = {'place_transition', 'attention_shift', 'energy_change'}

# Priority ordering for aggregation bundles
_PRIORITY_ORDER = {'critical': 0, 'normal': 1, 'low': 2}

# Confidence → language register mapping
CONFIDENCE_PHRASING = {
    'high': '',            # assertive, no hedge
    'medium': 'hedged',    # "looks like", "may have"
    'low': 'suppressed',   # don't message, log only
}


@dataclass
class BridgeEvent:
    """Normalized event from any source."""
    event_type: str
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    interrupt_priority: str = 'normal'
    silent: bool = False


def _load_config() -> dict:
    """Load event bridge configuration."""
    import os
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'configs', 'agents', 'event-bridge.json'
    )
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"{LOG_PREFIX} Config load failed: {e}, using defaults")
        return {}


class EventBridgeService:
    """Connects ambient context changes to autonomous actions."""

    def __init__(self):
        self._config = _load_config()
        self._redis = RedisClientService.create_connection()

    def submit_event(self, event: BridgeEvent) -> bool:
        """
        Submit an event through the bridge pipeline.

        Returns True if the event was routed for action.
        """
        logger.debug(f"{LOG_PREFIX} Received {event.event_type} (confidence={event.confidence:.2f})")

        # 1. Confidence threshold
        threshold = self._config.get('confidence_threshold', 0.6)
        if event.confidence < threshold:
            logger.debug(f"{LOG_PREFIX} Suppressed {event.event_type} (confidence {event.confidence:.2f} < {threshold})")
            return False

        # 2. State stabilization for jitter-prone events
        if event.event_type in _STABILIZED_EVENTS:
            if not self._is_stable(event):
                return False

        # 3. Cooldown enforcement
        if not self._check_cooldown(event.event_type):
            logger.debug(f"{LOG_PREFIX} Cooldown active for {event.event_type}")
            return False

        # 4. Classify interrupt priority from config
        priorities = self._config.get('interrupt_priority', {})
        event.interrupt_priority = priorities.get(event.event_type, 'normal')

        # 5. Check if this is a silent (internal-only) event
        silent_events = self._config.get('silent_events', [])
        if event.event_type in silent_events:
            event.silent = True
            self._handle_silent_event(event)
            return True

        # 6. Focus gate
        if not self._passes_focus_gate(event):
            logger.debug(f"{LOG_PREFIX} Focus gate blocked {event.event_type}")
            return False

        # 7. Set cooldown
        self._set_cooldown(event.event_type)

        # 8. Add to aggregation bundle
        self._add_to_bundle(event)

        # 9. Process bundle if aggregation window has passed
        self._try_flush_bundle()

        return True

    def flush_bundle(self) -> List[BridgeEvent]:
        """Force-flush the aggregation bundle. Returns events that were routed."""
        return self._try_flush_bundle(force=True)

    # ── Stabilization ────────────────────────────────────────────────

    def _is_stable(self, event: BridgeEvent) -> bool:
        """
        Check if a state has been stable for the stabilization window.

        Requires the same to_state for >= stabilization_window_seconds.
        """
        window = self._config.get('stabilization_window_seconds', 90)
        key = f"{_STABILIZATION_PREFIX}{event.event_type}"

        try:
            stored = self._redis.get(key)
            if stored:
                data = json.loads(stored)
                if data.get('to_state') == event.to_state:
                    elapsed = time.time() - data.get('first_seen', time.time())
                    if elapsed >= window:
                        # Stable — clear stabilization state and allow
                        self._redis.delete(key)
                        logger.debug(
                            f"{LOG_PREFIX} {event.event_type} stabilized "
                            f"({event.to_state}, {elapsed:.0f}s >= {window}s)"
                        )
                        return True
                    else:
                        # Still waiting
                        return False
                else:
                    # State changed — reset
                    pass

            # Record new state candidate
            self._redis.setex(key, window + 30, json.dumps({
                'to_state': event.to_state,
                'first_seen': time.time(),
            }))
            return False

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Stabilization check failed: {e}")
            return True  # Fail open

    # ── Cooldowns ────────────────────────────────────────────────────

    def _check_cooldown(self, event_type: str) -> bool:
        """Check if event type is past its cooldown period."""
        cooldowns = self._config.get('cooldowns', {})
        cooldown = cooldowns.get(event_type, 0)
        if cooldown <= 0:
            return True

        key = f"{_COOLDOWN_PREFIX}{event_type}"
        try:
            return not bool(self._redis.get(key))
        except Exception:
            return True

    def _set_cooldown(self, event_type: str):
        """Set cooldown for an event type."""
        cooldowns = self._config.get('cooldowns', {})
        cooldown = cooldowns.get(event_type, 0)
        if cooldown <= 0:
            return

        key = f"{_COOLDOWN_PREFIX}{event_type}"
        try:
            self._redis.setex(key, cooldown, "1")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Cooldown set failed: {e}")

    # ── Focus Gate ───────────────────────────────────────────────────

    def _passes_focus_gate(self, event: BridgeEvent) -> bool:
        """
        Check if event can pass the focus gate.

        Deep focus blocks all events except critical priority.
        """
        bypass_events = self._config.get('focus_gate_bypass', [])
        if event.event_type in bypass_events:
            return True

        try:
            from services.ambient_inference_service import AmbientInferenceService
            inference = AmbientInferenceService()
            if inference.is_user_deep_focus():
                if event.interrupt_priority == 'critical':
                    logger.info(f"{LOG_PREFIX} Critical event {event.event_type} bypasses focus gate")
                    return True
                return False
        except Exception:
            pass

        return True

    # ── Silent Events ────────────────────────────────────────────────

    def _handle_silent_event(self, event: BridgeEvent):
        """
        Handle silent internal events — no user message, just internal flags.

        attention_deepened → suppress suggestions
        energy_low → adjust suggestion complexity
        """
        try:
            if event.event_type == 'attention_deepened':
                self._redis.setex("event_bridge:suppress_suggestions", 1800, "1")
                logger.debug(f"{LOG_PREFIX} Attention deepened — suppressing suggestions for 30min")

            elif event.event_type == 'energy_low':
                self._redis.setex("event_bridge:low_energy_mode", 3600, "1")
                logger.debug(f"{LOG_PREFIX} Energy low — lighter suggestions for 1hr")

            # Always set a generic flag for the event
            self._redis.setex(f"event_bridge:flag:{event.event_type}", 1800, json.dumps({
                'confidence': event.confidence,
                'timestamp': event.timestamp,
            }))
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Silent event handling failed: {e}")

    # ── Aggregation ──────────────────────────────────────────────────

    def _add_to_bundle(self, event: BridgeEvent):
        """Add event to the pending aggregation bundle."""
        try:
            entry = json.dumps({
                'event_type': event.event_type,
                'from_state': event.from_state,
                'to_state': event.to_state,
                'confidence': event.confidence,
                'timestamp': event.timestamp,
                'payload': event.payload,
                'interrupt_priority': event.interrupt_priority,
            })
            self._redis.rpush(_AGGREGATION_KEY, entry)
            self._redis.expire(_AGGREGATION_KEY, _AGGREGATION_TTL)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Bundle add failed: {e}")

    def _try_flush_bundle(self, force: bool = False) -> List[BridgeEvent]:
        """
        Flush the aggregation bundle if the window has passed.

        Bundles co-occurring events within the aggregation window into a single
        contextual message, priority-ordered (critical > normal > low).
        """
        window = self._config.get('aggregation_window_seconds', 60)

        try:
            items = self._redis.lrange(_AGGREGATION_KEY, 0, -1)
            if not items:
                return []

            # Check if oldest item is past the aggregation window
            first = json.loads(items[0])
            elapsed = time.time() - first.get('timestamp', time.time())

            if not force and elapsed < window:
                return []

            # Flush all items
            self._redis.delete(_AGGREGATION_KEY)

            events = []
            for raw in items:
                data = json.loads(raw)
                events.append(BridgeEvent(
                    event_type=data['event_type'],
                    from_state=data.get('from_state'),
                    to_state=data.get('to_state'),
                    confidence=data.get('confidence', 0.5),
                    timestamp=data.get('timestamp', time.time()),
                    payload=data.get('payload', {}),
                    interrupt_priority=data.get('interrupt_priority', 'normal'),
                ))

            # Sort by priority
            events.sort(key=lambda e: _PRIORITY_ORDER.get(e.interrupt_priority, 1))

            # Route the bundle
            self._route_bundle(events)

            return events

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Bundle flush failed: {e}")
            return []

    def _route_bundle(self, events: List[BridgeEvent]):
        """
        Route a bundle of events through the autonomous action system.

        Builds a ThoughtContext with event metadata and passes it to the
        action decision router.
        """
        if not events:
            return

        primary = events[0]
        logger.info(
            f"{LOG_PREFIX} Routing bundle: {[e.event_type for e in events]} "
            f"(primary={primary.event_type}, confidence={primary.confidence:.2f})"
        )

        try:
            from services.autonomous_actions.base import ThoughtContext

            thought = ThoughtContext(
                thought_type='event',
                thought_content=self._compose_message(events),
                activation_energy=primary.confidence,
                seed_concept=primary.event_type,
                seed_topic=primary.to_state or primary.event_type,
                extra={
                    'event_type': primary.event_type,
                    'event_payload': {
                        'primary': {
                            'event_type': primary.event_type,
                            'from_state': primary.from_state,
                            'to_state': primary.to_state,
                            'confidence': primary.confidence,
                        },
                        'bundle_size': len(events),
                        'all_events': [e.event_type for e in events],
                    },
                    'confidence': primary.confidence,
                },
            )

            # Route through existing action decision router
            from services.autonomous_actions.action_decision_router import ActionDecisionRouter
            router = ActionDecisionRouter()
            router.evaluate_and_execute(thought)

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Event routing failed: {e}")

    def _compose_message(self, events: List[BridgeEvent]) -> str:
        """
        Compose a contextual message from bundled events.

        Uses confidence-weighted phrasing.
        """
        parts = []

        for event in events:
            phrasing = self._get_phrasing(event.confidence)
            msg = self._event_to_text(event, phrasing)
            if msg:
                parts.append(msg)

        return ' '.join(parts) if parts else ''

    def _get_phrasing(self, confidence: float) -> str:
        """Map confidence to phrasing register."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        return 'low'

    def _event_to_text(self, event: BridgeEvent, phrasing: str) -> str:
        """Convert a single event to natural text with confidence-appropriate phrasing."""
        if phrasing == 'low':
            return ''  # Suppressed

        templates = {
            'place_transition': {
                'high': f"You've arrived at {event.to_state}.",
                'medium': f"Looks like you may have arrived at {event.to_state}.",
            },
            'attention_shift': {
                'high': f"You seem to be in {event.to_state} mode.",
                'medium': f"Your attention may have shifted to {event.to_state}.",
            },
            'energy_change': {
                'high': f"Your energy seems {event.to_state}.",
                'medium': f"Energy levels may be {event.to_state}.",
            },
            'session_start': {
                'high': "Welcome back.",
                'medium': "Welcome back.",
            },
            'session_resume': {
                'high': "Welcome back.",
                'medium': "Welcome back.",
            },
            'item_approaching': {
                'high': f"{event.payload.get('item_title', 'Something')} is coming up.",
                'medium': f"{event.payload.get('item_title', 'Something')} may be coming up.",
            },
            'item_due': {
                'high': f"{event.payload.get('item_title', 'A scheduled item')} is due now.",
                'medium': f"{event.payload.get('item_title', 'A scheduled item')} is due now.",
            },
        }

        event_templates = templates.get(event.event_type, {})
        return event_templates.get(phrasing, '')

    # ── Public Helpers ───────────────────────────────────────────────

    def is_suggestions_suppressed(self) -> bool:
        """Check if suggestions are suppressed (attention_deepened active)."""
        try:
            return bool(self._redis.get("event_bridge:suppress_suggestions"))
        except Exception:
            return False

    def is_low_energy_mode(self) -> bool:
        """Check if low-energy mode is active."""
        try:
            return bool(self._redis.get("event_bridge:low_energy_mode"))
        except Exception:
            return False

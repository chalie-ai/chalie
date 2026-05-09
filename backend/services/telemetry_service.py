"""
Telemetry Service — in-process event collection and summarisation.

Provides a lightweight :class:`TelemetryCollector` that:

- Exposes a direct :meth:`TelemetryCollector.record` entry-point for services
  that want to emit telemetry events.
- Maintains an in-memory ring buffer (``collections.deque``, ``maxlen=1000``)
  so memory footprint is bounded even under high event rates.
- Provides :meth:`TelemetryCollector.get_summary` for the observability
  endpoint at ``/system/observability/telemetry``.

A process-level singleton is accessed via :func:`get_telemetry_collector`.

Typical usage by a service::

    from services.telemetry_service import get_telemetry_collector, MEMORY_RECALL

    get_telemetry_collector().record(MEMORY_RECALL, {"episode_count": 5})
"""

import datetime
import logging
import threading
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional

from utils.logger import get_correlation_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

#: Emitted when episodes are retrieved from memory for context assembly.
MEMORY_RECALL: str = "memory_recall"

#: Emitted after context assembly completes; payload should include episode,
#: concept, and trait counts.
CONTEXT_ASSEMBLY: str = "context_assembly"

#: Emitted when the proactive engine evaluates a potential outreach; payload
#: should include ``confidence``, ``social_cost``, and ``action_taken``.
PROACTIVE_TRIGGER: str = "proactive_trigger"

#: Emitted on goal create, decay, or evidence_added lifecycle transitions.
GOAL_LIFECYCLE: str = "goal_lifecycle"

#: Emitted on each iteration of the ACT reasoning loop.
ACT_LOOP_ITERATION: str = "act_loop_iteration"

#: Emitted when an ACT reasoning loop run completes (success or failure).
ACT_LOOP_COMPLETE: str = "act_loop_complete"

#: Emitted when a semantically surprising cross-topic connection is surfaced.
SURPRISING_CONNECTION: str = "surprising_connection"

#: Ordered tuple of all telemetry event type constants — used for bus
#: subscription and summary initialisation.
ALL_TELEMETRY_EVENT_TYPES: tuple = (
    MEMORY_RECALL,
    CONTEXT_ASSEMBLY,
    PROACTIVE_TRIGGER,
    GOAL_LIFECYCLE,
    ACT_LOOP_ITERATION,
    ACT_LOOP_COMPLETE,
    SURPRISING_CONNECTION,
)


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TelemetryCollector:
    """Thread-safe in-memory collector for structured telemetry events.

    Maintains a bounded ring buffer of the most-recent 1,000 events and
    per-type counters.  Designed to be used as a process singleton via
    :func:`get_telemetry_collector`.

    Attributes:
        _buffer: ``deque`` holding the most-recent events (maxlen=1000).
        _counts: Per-event-type occurrence counters.
        _lock: ``threading.Lock`` guarding all mutations.
    """

    def __init__(self) -> None:
        """Initialise an empty collector with an unbounded per-type counter and
        a ring buffer capped at 1,000 entries.
        """
        self._buffer: deque = deque(maxlen=1000)
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record a telemetry event into the ring buffer.

        Stamps the event with an ISO 8601 UTC timestamp and, if a
        correlation ID is active in the current execution context (set via
        :func:`~utils.logger.set_correlation_id`), attaches it to the
        stored entry.

        Args:
            event_type: One of the module-level event type constants (e.g.
                :data:`MEMORY_RECALL`).  Arbitrary strings are accepted but
                only the canonical constants are exposed in
                :meth:`get_summary`.
            payload: Caller-supplied mapping of event-specific fields.  The
                dict is *shallow-copied* so the caller may mutate their
                original dict after the call without affecting the stored
                entry.
        """
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        correlation_id: Optional[str] = get_correlation_id()

        entry: Dict[str, Any] = {
            "event_type": event_type,
            "timestamp": timestamp,
            "payload": dict(payload),
        }
        if correlation_id is not None:
            entry["correlation_id"] = correlation_id

        with self._lock:
            self._buffer.append(entry)
            self._counts[event_type] += 1

        logger.debug(
            "Telemetry event recorded",
            extra={"event_type": event_type, "correlation_id": correlation_id},
        )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all recorded telemetry events.

        The summary is keyed by event type and includes:

        - ``count`` — total number of events recorded since process start
          (not bounded by the ring buffer size).
        - ``recent`` — up to the 3 most-recent entries for that type,
          in reverse-chronological order (newest first).

        All canonical event types (:data:`ALL_TELEMETRY_EVENT_TYPES`) are
        present in the result even if no events of that type have been
        recorded yet (``count`` will be 0, ``recent`` will be an empty
        list).

        Returns:
            A ``dict`` of the form::

                {
                    "memory_recall": {
                        "count": 42,
                        "recent": [
                            {"event_type": "memory_recall", "timestamp": "...",
                             "payload": {...}},
                            ...
                        ],
                    },
                    ...
                }
        """
        with self._lock:
            # Snapshot the buffer to avoid holding the lock while building
            # the per-type recent lists.
            snapshot: List[Dict[str, Any]] = list(self._buffer)
            counts_snapshot: Dict[str, int] = dict(self._counts)

        # Build per-type recent lists from the snapshot (newest-first)
        recent_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in reversed(snapshot):
            etype = entry["event_type"]
            if len(recent_by_type[etype]) < 3:
                recent_by_type[etype].append(entry)

        # Assemble the final summary, ensuring all canonical types are present
        summary: Dict[str, Any] = {}
        all_types = set(ALL_TELEMETRY_EVENT_TYPES) | set(counts_snapshot.keys())
        for etype in sorted(all_types):
            summary[etype] = {
                "count": counts_snapshot.get(etype, 0),
                "recent": recent_by_type.get(etype, []),
            }

        return summary



# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector_instance: Optional[TelemetryCollector] = None
_singleton_lock: threading.Lock = threading.Lock()


def get_telemetry_collector() -> TelemetryCollector:
    """Return the process-level :class:`TelemetryCollector` singleton.

    Creates the instance on first call (thread-safe double-checked locking).
    Subsequent calls return the same object.

    Returns:
        The shared :class:`TelemetryCollector` instance for this process.
    """
    global _collector_instance
    if _collector_instance is None:
        with _singleton_lock:
            if _collector_instance is None:
                _collector_instance = TelemetryCollector()
                logger.debug("TelemetryCollector singleton created.")
    return _collector_instance

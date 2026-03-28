"""
Bridge between capability findings and Chalie's goal ecology.

Capabilities detect changes in the external world (calendar conflicts,
actionable emails, etc.) but those findings only affect the LLM prompt
via WorldStateService.  This module routes them into the goal ecology
so they can form or strengthen goals autonomously.

Usage inside a capability monitor()/understand():
    from capabilities.signal_bridge import emit_capability_signal
    emit_capability_signal("caldav", "schedule_fired",
                           "Conflict: standup overlaps design review")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Capability findings → cognitive signal types understood by
# GoalSignalService.route_cognitive_signal().
#
# "schedule_fired"     → temporal evidence  (calendar events, reminders)
# "new_knowledge"      → behavioral evidence (emails, new facts)
# "novel_observation"  → behavioral evidence (conflicts, anomalies)
# "task_state_changed" → behavioral evidence (task updates)
VALID_SIGNAL_TYPES = frozenset({
    "schedule_fired",
    "new_knowledge",
    "novel_observation",
    "task_state_changed",
})


def emit_capability_signal(
    capability_id: str,
    signal_type: str,
    content: str,
    source: Optional[str] = None,
) -> bool:
    """Route a capability finding to the goal ecology.

    Args:
        capability_id: Capability identifier, e.g. ``"caldav"`` or ``"imap"``.
        signal_type:   One of :data:`VALID_SIGNAL_TYPES`.
        content:       Human-readable description of the finding (max 500 chars).
        source:        Optional source tag, e.g. ``"caldav:conflict"``.

    Returns:
        True if the signal was routed, False if skipped or errored.
    """
    if (signal_type not in VALID_SIGNAL_TYPES
            or not content or len(content.strip()) < 10):
        return False
    try:
        from services.goal_signal_service import route_cognitive_signal

        route_cognitive_signal({
            "signal_type": signal_type,
            "payload": {"content": content[:500], "capability": capability_id},
            "source_id": source or f"{capability_id}:monitor",
        })
        return True
    except Exception:
        return False

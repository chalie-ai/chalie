"""
Engagement Signal Service — Read-only peer to EngagementTracker.

Boundary with EngagementTracker
--------------------------------
EngagementTracker (services/autonomous_actions/engagement_tracker.py) scores
individual proactive-message responses and writes ``proactive:engagement_score``
to MemoryStore.  EngagementSignalService reads that score and formats it as a
WorldState item when engagement is noteworthy (< 0.35 or > 0.88).  It does
**not** write to MemoryStore or score messages.

Usage
-----
Instantiate once and call ``get_engagement_items()`` to obtain a (possibly
empty) list of world-state dicts suitable for inclusion in
``WorldStateService.get_world_state()``.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
LOG_PREFIX = "[ENGAGEMENT SIGNAL]"

# MemoryStore key written by EngagementTracker._recompute_engagement_score()
_ENGAGEMENT_SCORE_KEY = "proactive:engagement_score"

# Thresholds that define the "unremarkable" band — scores inside this range
# are skipped so the context window is not spammed with routine values.
_LOW_THRESHOLD = 0.35   # Below this → engagement is low
_HIGH_THRESHOLD = 0.88  # Above this → engagement is unusually high


class EngagementSignalService:
    """
    Read-only service that surfaces the rolling engagement score as a
    WorldState item when the score is outside the unremarkable band.

    The service lazily initialises a MemoryStore connection on first use so it
    is safe to instantiate at import time.
    """

    def __init__(self) -> None:
        """
        Initialise the service.

        The MemoryStore connection is deferred to the first call of
        ``get_engagement_items()`` to avoid touching infrastructure at
        import/instantiation time.
        """
        self._store = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_store(self):
        """
        Lazily create and cache the shared MemoryStore connection.

        Returns:
            MemoryStore: The shared in-memory store singleton.
        """
        if self._store is None:
            from services.memory_client import MemoryClientService
            self._store = MemoryClientService.create_connection()
        return self._store

    # ── Public API ────────────────────────────────────────────────────────────

    def get_engagement_items(self) -> List[Dict[str, Any]]:
        """
        Return world-state items that reflect notable engagement levels.

        Reads ``proactive:engagement_score`` from MemoryStore (written by
        :class:`~services.autonomous_actions.engagement_tracker.EngagementTracker`).

        Decision table:

        +-----------------+-----------------------------------------------+
        | Score range     | Behaviour                                     |
        +=================+===============================================+
        | None / missing  | Return ``[]`` (no data yet)                   |
        +-----------------+-----------------------------------------------+
        | 0.35 – 0.88     | Return ``[]`` (unremarkable, skip context)    |
        +-----------------+-----------------------------------------------+
        | < 0.35          | Return item: low-engagement advisory          |
        +-----------------+-----------------------------------------------+
        | > 0.88          | Return item: high-engagement advisory         |
        +-----------------+-----------------------------------------------+

        Any exception (e.g. MemoryStore unavailable) is caught and logged;
        the method returns ``[]`` so callers are never blocked (fail-open).

        Returns:
            list[dict]: Zero or one dict with keys ``type`` (str),
            ``label`` (str), and ``salience`` (float 0.0–1.0).
        """
        try:
            raw = self._get_store().get(_ENGAGEMENT_SCORE_KEY)
            if raw is None:
                return []

            score = float(raw)

            if score < _LOW_THRESHOLD:
                logger.debug(
                    "%s Low engagement score %.3f — surfacing advisory",
                    LOG_PREFIX,
                    score,
                )
                return [
                    {
                        "type": "engagement",
                        "label": (
                            "User engagement is low — consider closing naturally"
                        ),
                        "salience": 0.3,
                    }
                ]

            if score > _HIGH_THRESHOLD:
                logger.debug(
                    "%s High engagement score %.3f — surfacing advisory",
                    LOG_PREFIX,
                    score,
                )
                return [
                    {
                        "type": "engagement",
                        "label": "User is highly engaged",
                        "salience": 0.4,
                    }
                ]

            # Score is in the unremarkable band — nothing to surface
            return []

        except Exception:
            logger.debug(
                "%s Failed to read engagement score — returning []",
                LOG_PREFIX,
                exc_info=True,
            )
            return []

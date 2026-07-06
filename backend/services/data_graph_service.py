"""DataGraphService — the one gateway to ``data_graph`` structured user-context.

Scope is exactly the spine's §3.11 clusters A/C: the *system* (user_summary) and
*user_specific* (traits) lanes the prompt builder reads at assembly time, plus
the post-turn write back into them. The *behavioral_pattern* lane has its own
gateway (:class:`services.behavioral_pattern_service.BehavioralPatternService`).
Episodic recall, FTS/vec search, concept-LUT canonicalization and the off-spine
decay cycle are NOT here — they stay on their own (off-spine) layers and are
migrated onto this gateway under a follow-up ticket, not this rewrite.

Every method is a thin wrapper over :class:`models.data_graph.DataGraph` — the
sole home of ``data_graph`` SQL (I6). Beyond the turn-scoped ``source``
provenance tag (``self.mp.config.channel``) on write, this service's only added
behavior is :meth:`store`'s kind-validation and never-raise write guard —
legacy semantics with no SQL of their own (I6-compliant).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models.behavioral_pattern import BehavioralPattern
from models.data_graph import DataGraph

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

# The full data_graph kind universe (unchanged from the legacy service). store()
# validates against this before writing (legacy `_KIND_POLICY` gate) — kept here,
# not on the model, since it is a service-layer write guard, not row SQL (I6).
# Off-spine callers still importing these names directly (place.py, save_graph.py,
# durable_timestamp.py, …) get an app-level handle under a follow-up ticket
# (REWRITE_SPEC.md §4.2 D10); the constants stay put in the meantime. The pattern
# kind is sourced from BehavioralPattern so there is one literal for it.
KIND_USER_SPECIFIC = "user_specific"
KIND_SYSTEM = "system"
KIND_MISC = "misc"
KIND_DOCUMENT = "document"
KIND_BEHAVIORAL_PATTERN = BehavioralPattern.KIND
KIND_PLACE = "place"
KIND_CONTACT = "contact"
KIND_DISCOVERY = "discovery"
VALID_KINDS = frozenset({
    KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_DOCUMENT,
    KIND_BEHAVIORAL_PATTERN, KIND_PLACE, KIND_CONTACT, KIND_DISCOVERY,
})


class DataGraphService:
    """Read the system/traits lanes; write facts back into the graph."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def traits(self) -> list[DataGraph]:
        """Live ``user_specific`` rows, most-reinforced first — the
        user-summary channel's facts section (§3.11 cluster B)."""
        return DataGraph.traits().get()

    def store(self, kind: str, key: str, value: str) -> DataGraph | None:
        """Upsert one fact, sourced from this turn's channel (§3.11 cluster C).

        Rejects an unknown ``kind`` and swallows a write failure — both logged,
        never raised — so one bad fact never crashes the post-turn pipeline
        (legacy ``DataGraphService.store`` semantics, preserved per
        REWRITE_SPEC.md §4.2 D10)."""
        if kind not in VALID_KINDS:
            logger.warning("data_graph_service.store: unknown kind=%r key=%r", kind, key)
            return None
        try:
            return DataGraph.store(kind, key, value, source=self.mp.config.channel)
        except Exception as exc:  # noqa: BLE001 — a bad write must not crash the turn
            logger.exception("data_graph_service.store failed kind=%r key=%r: %s", kind, key, exc)
            return None

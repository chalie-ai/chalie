"""DataGraphService — the one gateway to ``data_graph`` structured user-context.

Scope is exactly the spine's §3.11 clusters A/B/C: the *system* (user_summary),
*user_specific* (traits) and *behavioral_pattern* lanes the prompt builder reads
at assembly time, plus the post-turn write/decay back into them. Episodic
recall, FTS/vec search, concept-LUT canonicalization and the off-spine decay
cycle are NOT here — they stay on their own (off-spine) layers and are migrated
onto this gateway under a follow-up ticket, not this rewrite.

Every method is a thin wrapper over :class:`models.data_graph.DataGraph` — the
sole home of ``data_graph`` SQL (I6). Beyond the turn-scoped ``source``
provenance tag (``self.mp.config.channel``) on write, this service's only added
behavior is :meth:`store`'s kind-validation and never-raise write guard —
legacy semantics with no SQL of their own (I6-compliant).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from models.data_graph import DataGraph

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

# The full data_graph kind universe (unchanged from the legacy service). store()
# validates against this before writing (legacy `_KIND_POLICY` gate) — kept here,
# not on the model, since it is a service-layer write guard, not row SQL (I6).
# Off-spine callers still importing these names directly (place.py, save_graph.py,
# durable_timestamp.py, …) get an app-level handle under a follow-up ticket
# (REWRITE_SPEC.md §4.2 D10); the constants stay put in the meantime.
KIND_USER_SPECIFIC = "user_specific"
KIND_SYSTEM = "system"
KIND_MISC = "misc"
KIND_DOCUMENT = "document"
KIND_BEHAVIORAL_PATTERN = "behavioral_pattern"
KIND_PLACE = "place"
KIND_CONTACT = "contact"
KIND_DISCOVERY = "discovery"
VALID_KINDS = frozenset({
    KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_DOCUMENT,
    KIND_BEHAVIORAL_PATTERN, KIND_PLACE, KIND_CONTACT, KIND_DISCOVERY,
})


class DataGraphService:
    """Read the system/traits/pattern lanes; write + decay the graph."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def user_summary(self) -> str | None:
        """The live ``system`` row's value at key ``user_summary`` — the
        user-definition prompt fact (§3.11 cluster A)."""
        row = DataGraph.active_by_key("system", "user_summary")
        return row.value if row else None

    def user_summary_long(self) -> str | None:
        """The live ``system`` row's value at key ``user_summary_long`` — DMN's
        richer reflection context (§3.11 cluster A). Written by
        :meth:`persist_user_summary`; callers fall back to :meth:`user_summary`
        when absent."""
        row = DataGraph.active_by_key("system", "user_summary_long")
        return row.value if row else None

    def traits(self) -> list[DataGraph]:
        """Live ``user_specific`` rows, most-reinforced first — the
        user-summary channel's facts section (§3.11 cluster B)."""
        return DataGraph.traits().get()

    def patterns(self) -> list[DataGraph]:
        """Live ``behavioral_pattern`` rows, most-recently-confirmed first —
        the user-summary channel's active-patterns section (§3.11 cluster B)."""
        return DataGraph.patterns().get()

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

    def decay(self, touched_keys: set[str]) -> None:
        """Run the behavioural-pattern confidence sweep, exempting the
        pattern names written this turn (§3.11 cluster C)."""
        DataGraph.decay(touched_keys)

    def ids_for_touched(self, touched_keys: set[str]) -> set[int]:
        """Live ``behavioral_pattern`` row ids sourced from this turn's channel
        whose key was written this turn — the skill-association pass's input
        (legacy ``PatternSkillSyncHook``). Empty in / empty out."""
        if not touched_keys:
            return set()
        keys = sorted(touched_keys)
        placeholders = ", ".join("?" * len(keys))
        rows = (
            DataGraph.live("behavioral_pattern")
            .filter("source = ?", self.mp.config.channel)
            .filter(f"key IN ({placeholders})", *keys)
            .get()
        )
        return {row.id for row in rows if row.id is not None}

    def persist_user_summary(self, text: str) -> None:
        """Parse the user-summary channel's JSON response and upsert the long
        then short ``system`` facts — long FIRST so a crash mid-write leaves the
        richer fact in place (legacy ``PersistUserSummaryHook``). Every reject is
        logged, never raised; the two writes ride :meth:`store`'s own guard."""
        stripped = (text or "").strip()
        if not stripped:
            logger.warning("persist_user_summary: empty response — skipping")
            return
        body = stripped.removeprefix("```json").removeprefix("```").lstrip()
        body = body.removesuffix("```").rstrip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("persist_user_summary: JSON parse failed (%s) — skipping", exc)
            return
        if not isinstance(parsed, dict):
            logger.warning("persist_user_summary: parsed value is not a dict — skipping")
            return
        short = (parsed.get("short") or "").strip()
        long_ = (parsed.get("long") or "").strip()
        if not short or not long_:
            logger.warning("persist_user_summary: 'short'/'long' missing/empty — skipping")
            return
        self.store(KIND_SYSTEM, "user_summary_long", long_)
        self.store(KIND_SYSTEM, "user_summary", short)

"""BehavioralPatternService — the one gateway to the ``behavioral_pattern`` lane
of ``data_graph``.

Every method is a thin wrapper over
:class:`models.behavioral_pattern.BehavioralPattern` (the sole home of the
pattern SQL); the service's only added behaviour is the turn-scoped ``source``
provenance tag (``self.mp.config.channel``) on write and the Python confidence
ranking, which needs no SQL of its own. Error style mirrors
:class:`services.data_graph_service.DataGraphService`: reads and the
``save_pattern`` write propagate, the post-turn decay is self-guarding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from models.behavioral_pattern import BehavioralPattern

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

# The confidence-ranked existing-patterns block is capped at this many rows —
# the pattern/geo channels' prompt budget (legacy ``_TOP_PATTERN_CAP``).
_TOP_PATTERN_CAP = 50


class BehavioralPatternService:
    """Read, rank, write and decay the behavioural-pattern lane."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def patterns(self) -> list[BehavioralPattern]:
        """Live patterns, most-recently-confirmed first — the user-summary
        channel's active-patterns section."""
        return BehavioralPattern.recent().get()

    def top_patterns(self) -> dict[str, str]:
        """The top :data:`_TOP_PATTERN_CAP` patterns by confidence as an ordered
        ``name -> summary`` mapping — the pattern/geo channels' existing-patterns
        block. Ranked in Python over the live rows (skipping any without a valid
        numeric confidence and a name), so a malformed row can never break the
        ordering."""
        scored: list[tuple[float, dict[str, object]]] = []
        for row in self.patterns():
            content = BehavioralPattern.parse(row.value)
            if content is None or "confidence" not in content or not content.get("name"):
                continue
            try:
                confidence = float(cast("str | float", content["confidence"]))
            except (TypeError, ValueError):
                continue
            scored.append((confidence, content))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {
            cast("str", content["name"]): cast("str", content.get("summary", ""))
            for _, content in scored[:_TOP_PATTERN_CAP]
        }

    def store(self, validated: dict[str, object]) -> tuple[int, float, bool]:
        """Insert or reinforce the validated pattern, sourced from this turn's
        channel. Returns ``(row_id, confidence, reinforced)``. Grouped in one
        transaction so the exact-key read and the write are atomic (legacy
        ``_upsert_pattern`` semantics)."""
        with self.mp.db.transaction():
            return BehavioralPattern.upsert(validated, self.mp.config.channel)

    def decay_untouched(self, touched_keys: set[str]) -> None:
        """Run the on-turn confidence sweep, exempting the pattern names written
        this turn. Named ``decay_untouched`` (not ``decay``) because this is the
        on-turn, ``mp``-bound sweep — incompatible with the off-turn
        :class:`~orchestrators.decayable.Decayable` shape (``decay() -> int``),
        which this service does not implement (T1)."""
        BehavioralPattern.decay(touched_keys)

    def ids_for_touched(self, touched_keys: set[str]) -> set[int]:
        """Live pattern row ids sourced from this turn's channel whose key was
        written this turn — the skill-association pass's input. Empty in / empty
        out."""
        if not touched_keys:
            return set()
        rows = (
            BehavioralPattern.live()
            .filter("source", self.mp.config.channel)
            .filter_in("key", sorted(touched_keys))
            .get()
        )
        return {cast("int", row.id) for row in rows if row.id is not None}

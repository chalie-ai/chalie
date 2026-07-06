"""FactService — the one gateway to the ``user_specific`` (FACTS) lane of
``data_graph``. A thin wrapper over :class:`~models.fact.FactRow` (the sole home
of the SQL): it adds the typed status envelope the memory tool / worker / save
tool render, and registers as the off-turn FACTS :class:`~orchestrators.decayable.Decayable`.
mp-optional (T2): store/forget take their ``source`` provenance explicitly, and
decay is off-turn and mp-less, so no method needs ``mp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.fact import FactRow

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class FactService:
    """Store, forget, read and decay the FACTS (user_specific) lane."""

    def __init__(self, mp: MessageProcessor | None = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None) -> dict[str, object]:
        """Upsert one fact; return the typed status envelope
        (status ∈ {created, reinforced, superseded})."""
        row, status, old_value = FactRow.store(key, value, source=source)
        return self._store_envelope(status, key, value, old_value, row)

    def forget(self, key: str, value: str | None = None) -> dict[str, object]:
        """Invalidate a fact by exact key (optionally a specific value); return the
        typed forget envelope (status ∈ {forgotten, value_not_found, not_found})."""
        closed = FactRow.forget(key, value)
        if closed:
            return {"action": "forget", "status": "forgotten", "canonical_key": key,
                    "provided_key": key, "value": value, "old_value": value, "versions_removed": closed}
        status = "value_not_found" if value is not None else "not_found"
        return {"action": "forget", "status": status, "canonical_key": key,
                "provided_key": key, "value": value, "remaining_values": []}

    def traits(self) -> list[FactRow]:
        """Live FACTS, most-reinforced first — the user-summary channel's facts
        section. (E9 repoints the prompt's traits read here from the gateway.)"""
        return FactRow.live().order_by("retrieval_weight DESC").get()

    def decay(self) -> int:
        """Off-turn Decayable entry point: power-law rw-decay of live FACTS."""
        return FactRow.decay()

    @staticmethod
    def _store_envelope(status: str, key: str, value: str,
                        old_value: str | None, row: FactRow) -> dict[str, object]:
        """The store status envelope the memory tool / save_graph / worker read —
        ports ``_make_store_result``'s shape, trimmed to the fields this slice's
        statuses use (no LUT/coexist fields — those arrive in E1b)."""
        return {"action": "store", "status": status,
                "canonical_key": key, "provided_key": key,
                "value": value, "old_value": old_value,
                "date": row.last_confirmed_at, "row": row}

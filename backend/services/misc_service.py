"""MiscService — the one gateway to the ``misc`` lane of ``data_graph``. A
thin wrapper over :class:`~models.misc.MiscRow` (the sole home of the *misc*
SQL): it adds the typed status envelope the memory tool renders, and
registers as the off-turn MISC :class:`~orchestrators.decayable.Decayable`.
Misc has no concept-LUT canonicalization, so ``canonical_key`` always equals
``provided_key`` and ``rule``/``all_values`` are always ``None``. mp-optional
(T2): store takes its ``source`` provenance explicitly, and decay is off-turn
and mp-less, so no method needs ``mp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.misc import MiscRow

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class MiscService:
    """Store and decay the misc lane of the data graph."""

    def __init__(self, mp: MessageProcessor | None = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None) -> dict[str, object]:
        """Upsert one misc scratch note by exact key; return the status
        envelope (status ∈ {created, reinforced}). Misc has no concept LUT,
        so ``canonical_key`` always equals ``provided_key`` and
        ``rule``/``all_values`` are always ``None``."""
        row, status, old_value = MiscRow.store(key, value, source=source)
        return {"action": "store", "status": status,
                "canonical_key": key, "provided_key": key,
                "value": value, "old_value": old_value,
                "rule": None, "all_values": None,
                "date": row.last_confirmed_at, "row": row}

    def decay(self) -> int:
        """Off-turn Decayable entry point: power-law rw-decay of live MISC
        rows plus the 2-day hard-purge of expired ones."""
        return MiscRow.decay()

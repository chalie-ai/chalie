"""SystemService — the one gateway to the ``system`` lane of ``data_graph``.
A thin wrapper over :class:`~models.system.SystemRow` (the sole home of the
*data_graph* SQL for this kind): it adds the typed status envelope the memory
tool renders. Unlike the FACTS lane, system keys are exact and unstructured —
there is no concept-LUT canonicalizer, so ``canonical_key`` and ``provided_key``
are always the same string and ``rule``/``all_values`` are always ``None``.
mp-optional: store/forget take their ``source`` provenance explicitly, so no
method needs ``mp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.system import SystemRow

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class SystemService:
    """Store and forget the system lane of the data graph."""

    def __init__(self, mp: MessageProcessor | None = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None) -> dict[str, object]:
        """Upsert one system value by exact key; return the typed status
        envelope (status ∈ {created, reinforced, superseded}). System has no
        concept LUT, so ``canonical_key`` always equals ``provided_key`` and
        ``rule``/``all_values`` are always ``None``."""
        row, status, old_value = SystemRow.store(key, value, source=source)
        return {"action": "store", "status": status,
                "canonical_key": key, "provided_key": key,
                "value": value, "old_value": old_value,
                "rule": None, "all_values": None,
                "date": row.last_confirmed_at, "row": row}

    def forget(self, key: str, value: str | None = None) -> dict[str, object]:
        """Invalidate a system value by exact key (optionally a specific
        value); return the typed forget envelope (status ∈ {forgotten,
        value_not_found, not_found})."""
        closed = SystemRow.forget(key, value)
        if closed:
            return {"action": "forget", "status": "forgotten", "canonical_key": key,
                    "provided_key": key, "value": value, "old_value": value, "versions_removed": closed}
        status = "value_not_found" if value is not None else "not_found"
        return {"action": "forget", "status": status, "canonical_key": key,
                "provided_key": key, "value": value, "remaining_values": []}

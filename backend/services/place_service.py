"""PlaceService — the one gateway to the ``place`` lane of ``data_graph``. A
thin wrapper over :class:`~models.place.PlaceRow` (the sole home of the
*place* SQL): places have no concept-LUT canonicalization, so this just
surfaces the store status and the delete outcome — the place ability renders
its own :class:`~abilities._result.ToolResult`. mp-optional: ``store``/
``delete`` take their provenance/id explicitly, so no method needs ``mp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.place import PlaceRow

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class PlaceService:
    """Store and delete the place lane of the data graph."""

    def __init__(self, mp: MessageProcessor | None = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None) -> dict[str, object]:
        """Upsert one place by exact key; return the status envelope
        (status ∈ {created, reinforced, superseded})."""
        row, status, old_value = PlaceRow.store(key, value, source=source)
        return {"status": status, "key": key, "value": value,
                "old_value": old_value, "row": row}

    def delete(self, row_id: int) -> bool:
        """Soft-delete the live place row with this id; ``False`` if none."""
        return PlaceRow.delete_by_id(row_id)

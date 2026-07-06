"""ContactService — the one gateway to the ``contact`` lane of ``data_graph``. A
thin wrapper over :class:`~models.contact.ContactRow` (the sole home of the
*contact* SQL): contacts render no status envelope (callers ignore the return),
so this stays a bare pass-through to the store primitive. mp-optional (T2):
``store`` takes its ``source`` provenance explicitly, and contacts carry no
decay, so no method needs ``mp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.contact import ContactRow

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class ContactService:
    """Store, or reinforce, one contact row."""

    def __init__(self, mp: "MessageProcessor | None" = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None) -> ContactRow:
        """Upsert one contact row; contacts render no status envelope, callers
        ignore the return."""
        return ContactRow.store(key, value, source=source)

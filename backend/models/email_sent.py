"""EmailSent — one row of the ``emails_sent`` ledger.

Active-record row-model backed by the ``emails_sent`` table defined in
``backend/schema.sql``. The table stores the RFC 5322 Message-ID stamped on
each outgoing SMTP send together with the timestamp of the send (filled by
the ``value`` column's ``DEFAULT(datetime('now'))``). Lookups go through the
natural key (``message_id``), not the ``id`` PK, so the public API is a pair of
bespoke, distinctly-named ``@classmethod``\\ s rather than shadowing
:func:`model.Model.get`/save/delete (which would violate LSP), mirroring the
layout of :class:`~models.setting.Setting`.

No vault, no encryption, no service class."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from models.model import Model


class EmailSent(Model):
    """One ``emails_sent`` row: an archived Message-ID of an outgoing send."""

    __columns__: ClassVar[tuple[str, ...]] = ("id", "key", "value")

    @classmethod
    def get_table(cls) -> str:
        return "emails_sent"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access). ``id`` keeps the
    # base Model annotation.
    key: str
    value: str

    @classmethod
    def record(cls, message_id: str) -> None:
        """Insert the given Message-ID into the ledger if it isn't already
        present. Idempotent: duplicate IDs are dropped on INSERT. Skips the
        ``value`` column so the column default fills the send timestamp."""
        conn = cls._bound_connection()
        conn.execute(
            "INSERT OR IGNORE INTO emails_sent (key) VALUES (?)",
            (message_id,),
        )

    @classmethod
    def sent_ids(cls, message_ids: Iterable[str]) -> set[str]:
        """Return the subset of ``message_ids`` that exist in the ledger."""
        # Materialize first: message_ids may be a generator, which is truthy
        # even when empty and is consumed by the first iteration.
        ids = [str(mid) for mid in message_ids]
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT key FROM emails_sent WHERE key IN ({placeholders})"
        cursor = cls._bound_connection().execute(sql, ids)
        return {row[0] for row in cursor.fetchall()}

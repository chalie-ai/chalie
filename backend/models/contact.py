"""One ``data_graph`` row of kind ``contact`` — the passive people-index lane
behind IMAP sender indexing and CardDAV contact sync (see
``capabilities/contact_resolver.py``).

Same table, columns and SQL semantics as :class:`models.data_graph.DataGraph`;
this model is the SOLE home of the *contact* slice of that table. ``"contact"``
is not in ``DataGraph._SUPERSEDING_KINDS``, so it is a non-superseding kind: a
contradicting value at the same key inserts a fresh row rather than temporally
superseding the old one (no demote, no ``valid_from``). Contacts also carry no
decay (ttl=None) — they are a passive identity index, not a decaying fact — so
this model declares no ``decay`` method."""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class ContactRow(DataGraphRow):
    KIND: ClassVar[str] = "contact"

    @classmethod
    def active_by_key(cls, key: str) -> Self | None:
        """The single active row for this kind's exact ``key`` — the store
        lookup (mirrors ``_SELECT_ACTIVE_BY_KIND_KEY_SQL``: ``active = 1`` only,
        NOT ``deleted_at``-filtered)."""
        return (cls.filter("kind = ?", cls.KIND).filter("key = ?", key)
                   .filter("active = 1").first())

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> Self:
        """Exact-key upsert, ported from the non-superseding branch of
        ``DataGraph.store`` (``"contact"`` is not in ``_SUPERSEDING_KINDS``): no
        active row → insert. Same value (case-folded, trimmed) → :meth:`reinforce`
        (:meth:`~models.data_graph.DataGraphRow.reinforce`). A contradicting value
        inserts a FRESH row — no demote, no supersede, no ``valid_from`` — since
        contacts never temporally supersede."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            return cls(
                kind=cls.KIND, key=key, value=value, source=source,
                first_seen_at=now, last_confirmed_at=now,
            ).save()
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce()
        return cls(
            kind=cls.KIND, key=key, value=value, source=source,
            first_seen_at=now, last_confirmed_at=now,
        ).save()

"""The ``document`` vertical of ``data_graph`` — chunked fragment text for
uploaded/created documents. Subclasses the shared
:class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec
write-sync, kind-scoped live reads); adds only the DOCUMENT kind, its
plain-insert fragment write path, and the source-scoped bulk read/activation/
purge primitives this two-table vertical needs. Sole home of this lane's SQL
(I6).

Two-table, joined by string convention (no FK, no schema change): document
*metadata* lives in the separate ``documents`` table (owned by
``DocumentService``); this model owns only the ``data_graph`` fragment side,
joined to its parent by ``source = f"document:{doc_id}"`` (shared across every
fragment of one document) and ``key = f"doc:{doc_id}:{i:03d}"`` (unique per
chunk).

CRITICAL DIVERGENCE from every other vertical: document fragments are NEVER
re-confirmed by key — each ingest mints a brand-new ``doc_id``, so brand-new
keys. There is no contradiction/reinforce/supersede branch here;
:meth:`create_fragment` is a plain per-fragment INSERT. Document
*supersession* lives entirely on the ``documents.supersedes_id`` column (a
separate metadata mechanism, out of scope for this model); on re-ingest the
OLD fragments are hard-purged via :meth:`purge_by_document_id`, not
superseded. Fragments carry no independent TTL — they live and die with their
parent ``documents`` row, so this vertical declares no ``decay()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

from models.data_graph import DataGraphRow
from models.query import Query
from services.time_utils import utc_now

if TYPE_CHECKING:
    from models.document_meta import DocumentMetaData


class DocumentRow(DataGraphRow):
    KIND: ClassVar[str] = "document"

    @classmethod
    def create_fragment(cls, doc_id: str, index: int, text: str) -> Self:
        """Plain-insert one chunk of ``doc_id`` at position ``index`` — NOT an
        upsert: every ingest mints a fresh ``doc_id``, so ``key`` is always
        brand new and there is nothing to reinforce or supersede against."""
        now = utc_now().isoformat()
        return cls(
            kind=cls.KIND, key=f"doc:{doc_id}:{index:03d}", value=text,
            source=f"document:{doc_id}", first_seen_at=now, last_confirmed_at=now,
        ).save()

    @classmethod
    def for_document_id(cls, doc_id: str) -> Query[Self]:
        """Live fragments for one document, chunk (``key``) order — the join
        every read (view/preview/full-content) resolves through."""
        return cls.live().filter("source", f"document:{doc_id}").order_by("key")

    def get_meta_data(self) -> "DocumentMetaData | None":
        """This fragment's parent ``documents`` row — the metadata half of
        the two-table join documented above, resolved off ``source``
        (``"document:{doc_id}"``, the same convention :meth:`for_document_id`
        builds). ``None`` if the fragment carries no source (shouldn't happen
        for a real row, but ``source`` is nullable on the shared
        ``data_graph`` shell) or the parent row is gone.

        Deferred import: ``models.document_meta`` imports THIS module at top
        level for the inverse relationship
        (``DocumentMetaData.get_document``), so importing it back at module
        scope here would be circular."""
        from models.document_meta import DocumentMetaData  # noqa: PLC0415

        if self.source is None:
            return None
        doc_id = self.source.split(":", 1)[1]
        return DocumentMetaData.filter("id", doc_id).first()

    @classmethod
    def set_fragments_active(cls, doc_id: str, active: int) -> None:
        """Flip every fragment's ``active`` flag for one document. ``active=0``
        (soft-delete) stops them surfacing in
        :meth:`~models.data_graph.DataGraphRow.live` and in cross-kind recall;
        ``active=1`` (restore) reverses it."""
        cls.filter("kind", cls.KIND).filter("source", f"document:{doc_id}").update(active=active)

    @classmethod
    def purge_by_document_id(cls, doc_id: str) -> int:
        """Hard-delete every fragment for ``doc_id`` — the row plus its FTS
        posting and key/value-vec shadows — mirroring
        ``models.misc.MiscRow._purge_expired``'s per-row FTS-then-DELETE
        idiom (external-content FTS5 must be told the indexed values before
        the source row disappears). The primitive that fixes the
        orphaned-fragment bug on hard-delete and folder-watcher re-ingest.
        Returns the count purged."""
        conn = cls._bound_connection()
        dead = conn.execute(
            "SELECT rowid, key, value, kind, search_queries FROM data_graph "
            "WHERE kind = ? AND source = ?",
            (cls.KIND, f"document:{doc_id}"),
        ).fetchall()
        for rowid, key, value, kind, search_queries in dead:
            cls._purge_search_index(
                conn, rowid,
                {"key": key, "value": value, "kind": kind, "search_queries": search_queries},
            )
            conn.execute("DELETE FROM data_graph WHERE rowid = ?", (rowid,))
        return len(dead)

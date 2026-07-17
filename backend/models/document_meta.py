"""DocumentMetaData — one ``documents`` row: the metadata half of the
two-table document vertical (Rule 5 / §4.1). ``id`` is a caller-minted hex
token (:func:`secrets.token_hex`), not model-generated — the on-disk path is
built from it *before* the row exists — so :meth:`create` is a bespoke INSERT
rather than a fresh instance's :meth:`~models.model.Model.save`, whose ``id
is None`` insert-detection can never fire for an id supplied up front.

The rest of this model's writes are targeted column UPDATEs, ported verbatim
(same SQL, same ``datetime('now')`` stamps) off the dissolved raw-SQL bodies
in ``services/document_service.py``, via the same ``update_fields`` idiom
:meth:`~models.episode.Episode.update_fields` established: arbitrary columns
plus ``updated_at`` in one statement, values already sqlite-bindable (JSON
columns are serialised by the service). A handful of reads/writes stay
hand-rolled SQL because the query builder cannot express them: an
explicit-id INSERT, OR-across-columns search, a GROUP BY aggregate, or a
``documents_vec`` join/MATCH — the builder has no INSERT/UPDATE/aggregate/
join support at all (see ``models/query.py``).

Two-table vertical: fragment text lives in the separate ``data_graph`` table,
owned by :class:`~models.document.DocumentRow` (the ``document`` kind of that
vertical), joined to this row by string convention
(``source = f"document:{doc_id}"``) — see that module's docstring for the
full two-table contract. :meth:`get_document` is the metadata→fragments
relationship helper; the inverse (``DocumentRow.get_meta_data``) lives on
that model.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import ClassVar, Self

from models.document import DocumentRow
from models.model import Model
from models.query import Query
from services._vec_upsert import vec0_upsert


class DocumentMetaData(Model):
    """One ``documents`` row: upload/creation metadata, classification,
    lifecycle (soft-delete/purge), and the ``documents_vec`` embedding
    join-key."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "original_name", "mime_type", "file_size_bytes", "file_path",
        "file_hash", "page_count", "status", "error_message", "chunk_count",
        "source_type", "watched_folder_id", "tags", "summary",
        "extracted_metadata", "supersedes_id", "clean_text", "language",
        "fingerprint", "doc_category", "doc_project", "doc_date",
        "meta_locked", "created_at", "updated_at", "deleted_at", "purge_after",
    )

    @classmethod
    def get_table(cls) -> str:
        return "documents"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    original_name: str
    mime_type: str
    file_size_bytes: int | None
    file_path: str
    file_hash: str | None
    page_count: int | None
    status: str
    error_message: str | None
    chunk_count: int
    source_type: str
    watched_folder_id: str | None
    tags: str
    summary: str | None
    extracted_metadata: str
    supersedes_id: str | None
    clean_text: str | None
    language: str | None
    fingerprint: str | None
    doc_category: str | None
    doc_project: str | None
    doc_date: str | None
    meta_locked: int
    created_at: str
    updated_at: str
    deleted_at: str | None
    purge_after: str | None

    #: The three classification columns :meth:`classification_groups` may
    #: group by — validated before the field name reaches an f-string
    #: (mirrors ``Query._validate_identifier``'s belt-and-braces posture,
    #: even though the service already whitelists ``field`` first).
    _CLASSIFICATION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"doc_category", "doc_project", "doc_date"}
    )

    # ── relationship helper ─────────────────────────────────────────────

    def get_document(self) -> Query[DocumentRow]:
        """This document's live fragments — the data_graph half of the
        two-table join. Returns the un-executed query (mirrors
        :meth:`~models.document.DocumentRow.for_document_id`) so a caller can
        further chain (``.get()``, ``.limit()``, ``.count()``, ...)."""
        return DocumentRow.for_document_id(str(self.id))

    # ── scopes ───────────────────────────────────────────────────────────

    @classmethod
    def live(cls) -> Query[Self]:
        """The live-document scope — ``deleted_at IS NULL`` — the shared
        predicate under every non-deleted metadata read."""
        return cls.filter("deleted_at", None, "IS")

    # ── create ───────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        doc_id: str,
        original_name: str,
        mime_type: str,
        file_size_bytes: int,
        file_path: str,
        file_hash: str,
        source_type: str,
        watched_folder_id: str | None,
    ) -> None:
        """Insert one new document row. Bespoke INSERT (see module
        docstring): ``doc_id`` is a caller-supplied hex token minted before
        this row exists, so ``Model.save()``'s ``id is None`` insert
        detection never applies here."""
        cls._bound_connection().execute(
            "INSERT INTO documents "
            "(id, original_name, mime_type, file_size_bytes, file_path, file_hash, "
            " source_type, watched_folder_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
            (doc_id, original_name, mime_type, file_size_bytes, file_path, file_hash,
             source_type, watched_folder_id),
        )

    # ── search (OR-across-columns + json_each — the builder can't) ────────

    @classmethod
    def search(cls, query: str) -> list[Self]:
        """Live-document metadata search: name/category/project ``LIKE``
        ORed with tag-array membership (``EXISTS(json_each(tags))``) — a
        predicate the structured builder cannot express (OR across columns,
        an EXISTS subquery). Newest first."""
        like_query = f"%{query}%"
        rows = cls._bound_connection().execute(
            "SELECT * FROM documents "
            "WHERE deleted_at IS NULL "
            "  AND (LOWER(original_name) LIKE LOWER(?) "
            "       OR EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?) "
            "       OR LOWER(doc_category) LIKE LOWER(?) "
            "       OR LOWER(doc_project) LIKE LOWER(?)) "
            "ORDER BY created_at DESC",
            (like_query, query, like_query, like_query),
        ).fetchall()
        return [cls.hydrate(row) for row in rows]

    # ── targeted updates (no builder UPDATE support at all) ────────────────

    @classmethod
    def update_fields(cls, doc_id: str, fields: dict[str, object]) -> None:
        """Targeted UPDATE of arbitrary columns plus ``updated_at`` — the
        same idiom as ``Episode.update_fields``. Values must already be
        sqlite-bindable (JSON columns are serialised by the service). Empty
        ``fields`` is a no-op."""
        if not fields:
            return
        set_clauses = [f"{key} = ?" for key in fields]
        set_clauses.append("updated_at = datetime('now')")
        values = list(fields.values())
        values.append(doc_id)
        cls._bound_connection().execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )

    # ── lifecycle (soft-delete / restore / hard-delete / purge) ────────────

    @classmethod
    def soft_delete(cls, doc_id: str, purge_after: datetime) -> int:
        """Mark one live document deleted (idempotent — already-deleted rows
        are left alone) and schedule its purge date. Returns the affected
        row count so the caller knows whether anything actually changed."""
        cursor = cls._bound_connection().execute(
            "UPDATE documents SET deleted_at = datetime('now'), purge_after = ?, "
            "updated_at = datetime('now') WHERE id = ? AND deleted_at IS NULL",
            (purge_after, doc_id),
        )
        return cursor.rowcount

    @classmethod
    def restore(cls, doc_id: str) -> int:
        """Reactivate one soft-deleted document (idempotent — already-live
        rows are left alone). Returns the affected row count."""
        cursor = cls._bound_connection().execute(
            "UPDATE documents SET deleted_at = NULL, purge_after = NULL, "
            "updated_at = datetime('now') WHERE id = ? AND deleted_at IS NOT NULL",
            (doc_id,),
        )
        return cursor.rowcount

    @classmethod
    def hard_delete(cls, doc_id: str) -> int:
        """Hard-delete one document row and its ``documents_vec`` embedding
        (embedding first — vec0 virtual tables don't support FK cascades).
        Returns the affected ``documents`` row count."""
        cls._bound_connection().execute(
            "DELETE FROM documents_vec WHERE rowid = "
            "(SELECT rowid FROM documents WHERE id = ?)",
            (doc_id,),
        )
        return cls.filter("id", doc_id).delete()

    @classmethod
    def expired_ids(cls) -> list[str]:
        """IDs of every document past its scheduled purge date. Raw SQL: the
        builder binds only Python values, so it can't compare a column
        against the live ``datetime('now')`` SQL function."""
        rows = cls._bound_connection().execute(
            "SELECT id FROM documents "
            "WHERE purge_after IS NOT NULL AND purge_after < datetime('now')"
        ).fetchall()
        return [str(row[0]) for row in rows]

    # ── documents_vec embedding join (delete/insert; computation stays in
    #    the service) ───────────────────────────────────────────────────────

    @classmethod
    def set_embedding(cls, doc_id: str, blob: bytes) -> None:
        """Replace one document's ``documents_vec`` embedding row, keyed by
        the ``rowid`` borrowed from the parent ``documents`` row. A missing
        document is a silent no-op."""
        connection = cls._bound_connection()
        row = connection.execute(
            "SELECT rowid FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return
        vec0_upsert(connection, "documents_vec", row[0], blob)

    # ── duplicate detection (exact hash + semantic MATCH join) ─────────────

    @classmethod
    def by_file_hash(cls, file_hash: str, exclude_id: str | None) -> list[sqlite3.Row]:
        """Exact-hash duplicate candidates among live documents — the
        ``exclude_id IS NULL OR id != exclude_id`` branch is an OR the
        structured builder can't express as one predicate."""
        return cls._bound_connection().execute(
            "SELECT id, original_name, created_at FROM documents "
            "WHERE file_hash = ? AND deleted_at IS NULL "
            "  AND (? IS NULL OR id != ?)",
            (file_hash, exclude_id, exclude_id),
        ).fetchall()

    @classmethod
    def semantic_matches(cls, embedding_blob: bytes, k: int = 5) -> list[sqlite3.Row]:
        """Nearest-neighbour ``documents_vec`` MATCH joined back to
        ``(id, original_name, created_at, distance)`` — a join the builder
        can't express. Nearest-first, live documents only."""
        return cls._bound_connection().execute(
            "SELECT d.id, d.original_name, d.created_at, v.distance "
            "FROM documents_vec v JOIN documents d ON d.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? AND d.deleted_at IS NULL "
            "ORDER BY v.distance",
            (embedding_blob, k),
        ).fetchall()

    # ── classification (GROUP BY aggregate — the builder has none) ─────────

    @classmethod
    def classification_groups(cls, field: str) -> list[tuple[str, int]]:
        """``COUNT(*)`` grouped by one classification column, among ready,
        live documents — the builder has no GROUP BY/aggregate support.
        ``field`` must be one of :attr:`_CLASSIFICATION_FIELDS` (the service
        already whitelists it before calling; this re-check is
        defense-in-depth against the f-string below)."""
        if field not in cls._CLASSIFICATION_FIELDS:
            raise ValueError(f"unsupported classification field: {field!r}")
        connection = cls._bound_connection()
        if field == "doc_date":
            rows = connection.execute(
                "SELECT COALESCE(SUBSTR(doc_date, 1, 4), 'Unknown') AS grp, COUNT(*) AS cnt "
                "FROM documents WHERE deleted_at IS NULL AND status = 'ready' "
                "GROUP BY grp ORDER BY grp DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT COALESCE({field}, 'Uncategorized') AS grp, COUNT(*) AS cnt "
                "FROM documents WHERE deleted_at IS NULL AND status = 'ready' "
                "GROUP BY grp ORDER BY cnt DESC"
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

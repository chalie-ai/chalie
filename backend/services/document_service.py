"""
Document Service — Data access layer for document storage, retrieval, and search.

Manages document metadata, soft delete/purge lifecycle, duplicate detection
(hash + semantic), and metadata search. Document content is stored as data_graph
artifacts and retrieved via DocumentSkill.

Documents are reference material — retrieved via the document skill (ACT loop),
NOT injected into context assembly.
"""

import hashlib
import json
import logging
import os
import secrets
import shutil
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import paths
from services.embedding_utils import pack_embedding as _pack_embedding
from services.log_utils import safe
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)

# Default similarity thresholds for semantic dedup
DEDUP_EXACT_THRESHOLD = 0.15       # cosine distance < 0.15 = likely same document
DEDUP_REVISION_THRESHOLD = 0.35    # cosine distance < 0.35 = likely revision/update
DEDUP_MIN_TEXT_LENGTH = 200        # skip semantic dedup for very short docs

# Purge window (days after soft delete)
PURGE_WINDOW_DAYS = 30

# Document storage root — single hard-coded layout (paths.py).
DOCUMENTS_ROOT = str(paths.DOCUMENTS_DIR)


class DocumentService:
    """Manages document storage, chunk retrieval, and hybrid search."""

    def __init__(self, db_service):
        """Initialize the document service with a database connection.

        Args:
            db_service: :class:`~services.database_service.DatabaseService`
                instance used for all document storage and retrieval operations.
        """
        self.db = db_service
        self._write_queue = get_write_queue()

    # ─────────────────────────────────────────────
    # Document CRUD
    # ─────────────────────────────────────────────

    def create_document(
        self,
        original_name: str,
        mime_type: str,
        file_size: int,
        file_path: str,
        file_hash: str,
        source_type: str = 'upload',
        watched_folder_id: str = None,
        doc_id: str = None,
    ) -> str:
        """Create a new document record in the database.

        Args:
            original_name: Human-readable filename displayed in the UI.
            mime_type: MIME type string (e.g. ``'application/pdf'``).
            file_size: File size in bytes.
            file_path: Path relative to ``DOCUMENTS_ROOT`` (or absolute for
                watched-folder documents).
            file_hash: SHA-256 hex digest of the file for deduplication.
            source_type: Origin label (``'upload'``, ``'watched_folder'``,
                or ``'conversation'``).
            watched_folder_id: Optional ID of the originating watched folder.
            doc_id: Optional explicit document ID; generated if not provided.

        Returns:
            Eight-character hex document ID.

        Raises:
            Exception: Propagates any database error to the caller.
        """
        doc_id = doc_id or secrets.token_hex(4)

        try:
            _params = (
                doc_id, original_name, mime_type, file_size, file_path,
                file_hash, source_type, watched_folder_id,
            )

            def _insert(params=_params, db=self.db):
                """Insert the document row; executed on the write-queue thread."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """INSERT INTO documents
                               (id, original_name, mime_type, file_size_bytes, file_path,
                                file_hash, source_type, watched_folder_id,
                                created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))""",
                        params,
                    )
                    cursor.close()

            # submit_sync so any DB error propagates via raise below
            self._write_queue.submit_sync(_insert)

            logger.info(f"[DOCS] Created document '{original_name}' (id={doc_id})")
            return doc_id

        except Exception as e:
            logger.error(f"[DOCS] create_document failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document record by its ID.

        Args:
            doc_id: Eight-character hex document identifier.

        Returns:
            Document dict, or ``None`` if not found or on database error.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           doc_category, doc_project, doc_date, meta_locked,
                           watched_folder_id,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents WHERE id = ?
                """, (doc_id,))
                row = cursor.fetchone()
                cursor.close()

            if not row:
                return None
            return self._row_to_dict(row)

        except Exception as e:
            logger.error(f"[DOCS] get_document failed: {e}")
            return None

    def get_all_documents(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """Retrieve all document records, newest first.

        Args:
            include_deleted: When ``True``, soft-deleted documents are included in
                the results.  Defaults to ``False``.

        Returns:
            List of document dicts ordered by ``created_at`` descending.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if include_deleted:
                    cursor.execute("""
                        SELECT id, original_name, mime_type, file_size_bytes, file_path,
                               file_hash, page_count, status, error_message, chunk_count,
                               source_type, tags, summary, extracted_metadata, supersedes_id,
                               clean_text, language, fingerprint,
                               doc_category, doc_project, doc_date, meta_locked,
                               watched_folder_id,
                               created_at, updated_at, deleted_at, purge_after
                        FROM documents
                        ORDER BY created_at DESC
                    """)
                else:
                    cursor.execute("""
                        SELECT id, original_name, mime_type, file_size_bytes, file_path,
                               file_hash, page_count, status, error_message, chunk_count,
                               source_type, tags, summary, extracted_metadata, supersedes_id,
                               clean_text, language, fingerprint,
                               doc_category, doc_project, doc_date, meta_locked,
                               watched_folder_id,
                               created_at, updated_at, deleted_at, purge_after
                        FROM documents
                        WHERE deleted_at IS NULL
                        ORDER BY created_at DESC
                    """)
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"[DOCS] get_all_documents failed: {e}")
            return []

    def search_documents_metadata(self, query: str) -> List[Dict[str, Any]]:
        """Search non-deleted documents by name, tags, category, and project.

        Args:
            query: Search string matched case-insensitively against
                ``original_name``, ``doc_category``, and ``doc_project``;
                also matched exactly against tag values.

        Returns:
            List of matching document dicts ordered by ``created_at`` descending.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                like_query = f"%{query}%"
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           doc_category, doc_project, doc_date, meta_locked,
                           watched_folder_id,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents
                    WHERE deleted_at IS NULL
                      AND (LOWER(original_name) LIKE LOWER(?)
                           OR EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)
                           OR LOWER(doc_category) LIKE LOWER(?)
                           OR LOWER(doc_project) LIKE LOWER(?))
                    ORDER BY created_at DESC
                """, (like_query, query, like_query, like_query))
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"[DOCS] search_documents_metadata failed: {e}")
            return []

    def create_document_from_text(
        self,
        original_name: str,
        text_content: str,
        source_type: str = 'conversation',
    ) -> str:
        """
        Create a document from raw text without a file upload.

        Writes the text to disk as a Markdown file, computes a SHA-256
        hash, sizes the content, and creates a database record via
        :meth:`create_document`.

        Args:
            original_name: Filename to use (should end in ``.md``).
            text_content: Raw text content to write and store.
            source_type: Origin label (default ``'conversation'``).

        Returns:
            Eight-character hex document ID.
        """
        doc_id = secrets.token_hex(4)

        # Write markdown to disk — strip any path components from the filename
        # to prevent directory traversal via a crafted original_name.
        safe_name = os.path.basename(original_name) or "document.md"
        doc_dir = os.path.join(DOCUMENTS_ROOT, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        file_path = os.path.join(doc_dir, safe_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        file_hash = hashlib.sha256(text_content.encode()).hexdigest()
        file_size = len(text_content.encode('utf-8'))

        self.create_document(
            original_name=safe_name,
            mime_type='text/markdown',
            file_size=file_size,
            file_path=f"{doc_id}/{safe_name}",
            file_hash=file_hash,
            source_type=source_type,
            doc_id=doc_id,
        )

        def _set_clean(did=doc_id, txt=text_content, db=self.db):
            with db.connection() as conn:
                conn.execute(
                    "UPDATE documents SET clean_text = ? WHERE id = ?",
                    (txt, did),
                )
        self._write_queue.submit_sync(_set_clean)

        logger.info(f"[DOCS] Created text document '{original_name}' (id={doc_id})")
        return doc_id

    # ─────────────────────────────────────────────
    # Status & metadata updates
    # ─────────────────────────────────────────────

    def update_status(
        self,
        doc_id: str,
        status: str,
        error_message: Optional[str] = None,
        chunk_count: int = 0,
    ) -> None:
        """Update the processing status of a document.

        Args:
            doc_id: Eight-character hex document identifier.
            status: New status string (``'processing'``, ``'awaiting_confirmation'``,
                ``'ready'``, or ``'failed'``).
            error_message: Optional error detail stored when status is ``'failed'``.
            chunk_count: Number of chunks produced (0 for non-terminal states).

        Raises:
            Exception: Propagates any database error to the caller.
        """
        try:
            _params = (status, error_message, chunk_count, doc_id)

            def _update(params=_params, db=self.db):
                """Persist document status change on the write-queue thread."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """UPDATE documents
                           SET status = ?, error_message = ?, chunk_count = ?,
                               updated_at = datetime('now')
                           WHERE id = ?""",
                        params,
                    )
                    cursor.close()

            self._write_queue.submit_sync(_update)
            logger.info("[DOCS] Updated status for %s: %s", safe(doc_id), safe(status))
        except Exception as e:
            logger.error("[DOCS] update_status failed for %s: %s", safe(doc_id), e)
            raise

    def update_summary(self, doc_id: str, summary: str) -> None:
        try:
            def _update(did=doc_id, s=summary, db=self.db):
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE documents SET summary = ?, updated_at = datetime('now') WHERE id = ?",
                        (s, did),
                    )
            self._write_queue.submit_sync(_update)
        except Exception as e:
            logger.error(f"[DOCS] update_summary failed: {e}")

    def update_extracted_metadata(
        self,
        doc_id: str,
        metadata: dict,
        summary: str,
        summary_embedding: list = None,
        clean_text: str = None,
        language: str = None,
        fingerprint: str = None,
        page_count: int = None,
    ) -> None:
        """Store extracted metadata, summary, embedding, and text signals.

        Builds a dynamic SET clause — only updates columns whose values are
        provided (non-None).  This keeps the query minimal and avoids issues
        with the sqlite-vec virtual table (embeddings are stored separately
        in documents_vec).

        Raises on failure so the caller (process_document) can mark status='failed'.
        """
        set_parts = ["extracted_metadata = ?", "summary = ?", "updated_at = datetime('now')"]
        params: list = [json.dumps(metadata), summary]

        if clean_text is not None:
            set_parts.append("clean_text = ?")
            params.append(clean_text)
        if language is not None:
            set_parts.append("language = ?")
            params.append(language)
        if fingerprint is not None:
            set_parts.append("fingerprint = ?")
            params.append(fingerprint)
        if page_count is not None:
            set_parts.append("page_count = ?")
            params.append(page_count)

        params.append(doc_id)

        packed_emb = _pack_embedding(summary_embedding) if summary_embedding is not None else None
        sql = f"UPDATE documents SET {', '.join(set_parts)} WHERE id = ?"

        def _update_meta(
            stmt=sql,
            p=params,
            did=doc_id,
            emb=packed_emb,
            db=self.db,
        ):
            """Persist extracted metadata and optionally upsert the summary embedding."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(stmt, p)

                if emb is not None:
                    cursor.execute("SELECT rowid FROM documents WHERE id = ?", (did,))
                    row = cursor.fetchone()
                    if row:
                        rowid = row[0]
                        cursor.execute("DELETE FROM documents_vec WHERE rowid = ?", (rowid,))
                        cursor.execute(
                            "INSERT INTO documents_vec (rowid, embedding) VALUES (?, ?)",
                            (rowid, emb),
                        )

                cursor.close()

        self._write_queue.submit_sync(_update_meta)

    def set_supersedes(self, doc_id: str, supersedes_id: str) -> None:
        """Mark a document as replacing (superseding) an older version.

        Args:
            doc_id: ID of the newer document.
            supersedes_id: ID of the older document being replaced.
        """
        try:
            def _supersede(did=doc_id, sid=supersedes_id, db=self.db):
                """Set supersedes_id on the document row."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """UPDATE documents SET supersedes_id = ?,
                           updated_at = datetime('now') WHERE id = ?""",
                        (sid, did),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_supersede)
            logger.info("[DOCS] Document %s supersedes %s", safe(doc_id), safe(supersedes_id))
        except Exception as e:
            logger.error(f"[DOCS] set_supersedes failed: {e}")

    # ─────────────────────────────────────────────
    # Duplicate detection
    # ─────────────────────────────────────────────

    def find_duplicates(
        self,
        file_hash: str,
        summary_embedding: Optional[list] = None,
        text_length: int = 0,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dual-layer duplicate detection:
        1. Exact hash match (SHA-256)
        2. Semantic similarity of summary_embedding (if text is long enough)

        Args:
            exclude_id: Document ID to exclude from results (typically the
                        document being checked, to avoid self-matching).

        Returns list of dicts with 'doc', 'match_type', and 'distance' keys.
        """
        results = []

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Layer 1: exact hash match
                if file_hash:
                    cursor.execute("""
                        SELECT id, original_name, created_at
                        FROM documents
                        WHERE file_hash = ? AND deleted_at IS NULL
                          AND (? IS NULL OR id != ?)
                    """, (file_hash, exclude_id, exclude_id))
                    for row in cursor.fetchall():
                        results.append({
                            'id': row[0],
                            'original_name': row[1],
                            'created_at': row[2],
                            'match_type': 'exact',
                            'distance': 0.0,
                        })

                # Layer 2: semantic similarity (skip for short docs)
                if (summary_embedding
                        and text_length >= DEDUP_MIN_TEXT_LENGTH
                        and not results):
                    packed = _pack_embedding(summary_embedding)
                    cursor.execute("""
                        SELECT d.id, d.original_name, d.created_at,
                               v.distance
                        FROM documents_vec v
                        JOIN documents d ON d.rowid = v.rowid
                        WHERE v.embedding MATCH ? AND k = 5
                          AND d.deleted_at IS NULL
                        ORDER BY v.distance
                    """, (packed,))
                    for row in cursor.fetchall():
                        dist = float(row[3])
                        doc_id = row[0]
                        if exclude_id and doc_id == exclude_id:
                            continue
                        if dist < DEDUP_EXACT_THRESHOLD:
                            results.append({
                                'id': doc_id,
                                'original_name': row[1],
                                'created_at': row[2],
                                'match_type': 'semantic_exact',
                                'distance': dist,
                            })
                        elif dist < DEDUP_REVISION_THRESHOLD:
                            results.append({
                                'id': doc_id,
                                'original_name': row[1],
                                'created_at': row[2],
                                'match_type': 'semantic_revision',
                                'distance': dist,
                            })

                cursor.close()

        except Exception as e:
            logger.error(f"[DOCS] find_duplicates failed: {e}")

        return results

    # ─────────────────────────────────────────────
    # Soft delete / restore / purge
    # ─────────────────────────────────────────────

    def soft_delete(self, doc_id: str) -> bool:
        """Soft-delete a document by setting its deletion timestamp.

        Sets ``deleted_at`` to now and schedules ``purge_after`` 30 days from now.

        Args:
            doc_id: Eight-character hex document identifier.

        Returns:
            ``True`` if the document was found and deleted, ``False`` otherwise.
        """
        try:
            purge_after = datetime.now(timezone.utc) + timedelta(days=PURGE_WINDOW_DAYS)

            def _soft_delete(did=doc_id, pa=purge_after, db=self.db):
                """Set deleted_at and schedule purge; return rowcount."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """UPDATE documents
                           SET deleted_at = datetime('now'), purge_after = ?,
                               updated_at = datetime('now')
                           WHERE id = ? AND deleted_at IS NULL""",
                        (pa, did),
                    )
                    affected = cursor.rowcount
                    cursor.close()
                return affected

            updated = self._write_queue.submit_sync(_soft_delete) > 0

            if updated:
                logger.info("[DOCS] Soft-deleted document %s", safe(doc_id))
                # Deactivate data_graph artifacts so they stop surfacing in recall.
                try:
                    from services.data_graph_service import get_data_graph_service
                    dgs = get_data_graph_service()
                    with dgs.db.connection() as conn:
                        conn.execute(
                            "UPDATE data_graph SET active=0 WHERE source LIKE ?",
                            (f'document:{doc_id}%',),
                        )
                except Exception as exc:
                    logger.warning("[DOCS] Failed to deactivate artifacts for %s: %s", safe(doc_id), exc)
            return updated

        except Exception as e:
            logger.error(f"[DOCS] soft_delete failed: {e}")
            return False

    def restore(self, doc_id: str) -> bool:
        """Restore a previously soft-deleted document.

        Clears ``deleted_at`` and ``purge_after`` so the document re-appears in
        normal queries.

        Args:
            doc_id: Eight-character hex document identifier.

        Returns:
            ``True`` if the document was found and restored, ``False`` otherwise.
        """
        try:
            def _restore(did=doc_id, db=self.db):
                """Clear deleted_at and purge_after; return rowcount."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """UPDATE documents
                           SET deleted_at = NULL, purge_after = NULL,
                               updated_at = datetime('now')
                           WHERE id = ? AND deleted_at IS NOT NULL""",
                        (did,),
                    )
                    affected = cursor.rowcount
                    cursor.close()
                return affected

            updated = self._write_queue.submit_sync(_restore) > 0

            if updated:
                logger.info("[DOCS] Restored document %s", safe(doc_id))
                # Reactivate data_graph artifacts so they surface in recall again.
                try:
                    from services.data_graph_service import get_data_graph_service
                    dgs = get_data_graph_service()
                    with dgs.db.connection() as conn:
                        conn.execute(
                            "UPDATE data_graph SET active=1 WHERE source LIKE ?",
                            (f'document:{doc_id}%',),
                        )
                except Exception as exc:
                    logger.warning("[DOCS] Failed to reactivate artifacts for %s: %s", safe(doc_id), exc)
            return updated

        except Exception as e:
            logger.error(f"[DOCS] restore failed: {e}")
            return False

    def hard_delete(self, doc_id: str) -> bool:
        """Permanently delete a document record and its file from disk.

        Removes the entry from the ``documents_vec`` virtual table before deleting
        the parent row so that FK-unaware virtual tables are cleaned up correctly.

        Args:
            doc_id: Eight-character hex document identifier.

        Returns:
            ``True`` if the document was found and deleted, ``False`` otherwise.
        """
        try:
            doc = self.get_document(doc_id)
            if not doc:
                return False

            def _hard_delete(did=doc_id, db=self.db):
                """Delete virtual-table rows then the document row; return rowcount."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    # Clean up virtual tables BEFORE the document delete —
                    # sqlite-vec virtual tables don't support FK cascades.
                    cursor.execute(
                        "DELETE FROM documents_vec WHERE rowid = "
                        "(SELECT rowid FROM documents WHERE id = ?)",
                        (did,),
                    )
                    # Delete document record.
                    cursor.execute("DELETE FROM documents WHERE id = ?", (did,))
                    affected = cursor.rowcount
                    cursor.close()
                return affected

            deleted = self._write_queue.submit_sync(_hard_delete) > 0

            # Cascade-delete data_graph artifacts for this document.
            if deleted:
                try:
                    from services.data_graph_service import get_data_graph_service
                    get_data_graph_service().hard_delete_by_source_prefix(f'document:{doc_id}')
                except Exception as exc:
                    logger.warning("[DOCS] Failed to cascade-delete data_graph artifacts for %s: %s", safe(doc_id), exc)

            # Delete file from disk (skip for watched folder docs — source files are not ours)
            if deleted and doc.get('file_path') and not doc.get('watched_folder_id'):
                # Validate doc_id is a safe hex token before using in path construction.
                safe_doc_id = os.path.basename(doc_id)
                file_dir = os.path.join(DOCUMENTS_ROOT, safe_doc_id)
                resolved = os.path.realpath(file_dir)
                if resolved.startswith(os.path.realpath(DOCUMENTS_ROOT)) and os.path.exists(resolved):
                    shutil.rmtree(resolved, ignore_errors=True)
            if deleted:
                logger.info("[DOCS] Hard-deleted document %s", safe(doc_id))

            return deleted

        except Exception as e:
            logger.error(f"[DOCS] hard_delete failed: {e}")
            return False

    def purge_expired(self) -> int:
        """Hard-delete all documents that have passed their scheduled purge date.

        Returns:
            Number of documents permanently deleted.
        """
        try:
            # Find expired docs first (need file paths for disk cleanup)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM documents
                    WHERE purge_after IS NOT NULL AND purge_after < datetime('now')
                """)
                expired_ids = [row[0] for row in cursor.fetchall()]
                cursor.close()

            count = 0
            for doc_id in expired_ids:
                if self.hard_delete(doc_id):
                    count += 1

            if count > 0:
                logger.info(f"[DOCS] Purged {count} expired documents")
            return count

        except Exception as e:
            logger.error(f"[DOCS] purge_expired failed: {e}")
            return 0

    def search_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query documents by extracted_metadata JSON fields.
        Supported filters: document_type, company, has_expiration.
        """
        try:
            conditions = ["deleted_at IS NULL", "status = 'ready'"]
            params = []

            if 'document_type' in filters:
                conditions.append("json_extract(extracted_metadata, '$.document_type.value') = ?")
                params.append(filters['document_type'])

            if 'company' in filters:
                conditions.append(
                    "EXISTS (SELECT 1 FROM json_each(json_extract(extracted_metadata, '$.companies')) c "
                    "WHERE json_extract(c.value, '$.name') LIKE ?)"
                )
                params.append(f"%{filters['company']}%")

            if filters.get('has_expiration'):
                conditions.append("json_array_length(COALESCE(json_extract(extracted_metadata, '$.expiration_dates'), '[]')) > 0")

            where = " AND ".join(conditions)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           doc_category, doc_project, doc_date, meta_locked,
                           watched_folder_id,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents
                    WHERE {where}
                    ORDER BY created_at DESC
                """, params)
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"[DOCS] search_by_metadata failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Watched folder helpers
    # ─────────────────────────────────────────────

    def get_documents_by_watched_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get all documents (including soft-deleted) for a watched folder."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           doc_category, doc_project, doc_date, meta_locked,
                           watched_folder_id,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents
                    WHERE watched_folder_id = ?
                    ORDER BY created_at DESC
                """, (folder_id,))
                rows = cursor.fetchall()
                cursor.close()
            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[DOCS] get_documents_by_watched_folder failed: {e}")
            return []

    def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a non-deleted document by its file_path."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           doc_category, doc_project, doc_date, meta_locked,
                           watched_folder_id,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents
                    WHERE file_path = ? AND deleted_at IS NULL
                """, (file_path,))
                row = cursor.fetchone()
                cursor.close()
            if not row:
                return None
            return self._row_to_dict(row)
        except Exception as e:
            logger.error(f"[DOCS] get_document_by_path failed: {e}")
            return None

    def update_tags(self, doc_id: str, tags: list) -> None:
        """Update the tags JSON array for a document.

        Args:
            doc_id: Eight-character hex document identifier.
            tags: New list of tag strings to store.
        """
        try:
            def _update_tags(did=doc_id, t=json.dumps(tags), db=self.db):
                """Persist updated tags on the write-queue thread."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE documents SET tags = ?, updated_at = datetime('now') WHERE id = ?",
                        (t, did),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_update_tags)
        except Exception as e:
            logger.error(f"[DOCS] update_tags failed: {e}")

    def update_classification(
        self, doc_id: str,
        category: str = None, project: str = None,
        doc_date: str = None, lock: bool = False,
    ) -> None:
        """Update document classification metadata, optionally locking it.

        Args:
            doc_id: Eight-character hex document identifier.
            category: New ``doc_category`` value, or ``None`` to leave unchanged.
            project: New ``doc_project`` value, or ``None`` to leave unchanged.
            doc_date: New ``doc_date`` value (``YYYY-MM-DD``), or ``None`` to
                leave unchanged.
            lock: When ``True``, sets ``meta_locked = 1`` so automatic
                classification cannot overwrite user edits.
        """
        try:
            set_parts = ["updated_at = datetime('now')"]
            params: list = []
            if category is not None:
                set_parts.append("doc_category = ?")
                params.append(category)
            if project is not None:
                set_parts.append("doc_project = ?")
                params.append(project)
            if doc_date is not None:
                set_parts.append("doc_date = ?")
                params.append(doc_date)
            if lock:
                set_parts.append("meta_locked = 1")
            params.append(doc_id)

            def _update_class(
                stmt=f"UPDATE documents SET {', '.join(set_parts)} WHERE id = ?",
                p=params,
                db=self.db,
            ):
                """Persist classification metadata on the write-queue thread."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(stmt, p)
                    cursor.close()

            self._write_queue.submit_sync(_update_class)
        except Exception as e:
            logger.error(f"[DOCS] update_classification failed: {e}")

    def get_classification_groups(self, field: str) -> List[Dict[str, Any]]:
        """Get unique values and counts for a classification field (doc_category, doc_project, doc_date).

        For doc_date, groups by year. Returns [{value, count}] sorted by count desc.
        """
        if field not in ('doc_category', 'doc_project', 'doc_date'):
            return []
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if field == 'doc_date':
                    # Group by year
                    cursor.execute("""
                        SELECT COALESCE(SUBSTR(doc_date, 1, 4), 'Unknown') as grp,
                               COUNT(*) as cnt
                        FROM documents
                        WHERE deleted_at IS NULL AND status = 'ready'
                        GROUP BY grp ORDER BY grp DESC
                    """)
                else:
                    cursor.execute(f"""
                        SELECT COALESCE({field}, 'Uncategorized') as grp,
                               COUNT(*) as cnt
                        FROM documents
                        WHERE deleted_at IS NULL AND status = 'ready'
                        GROUP BY grp ORDER BY cnt DESC
                    """)
                rows = cursor.fetchall()
                cursor.close()
            return [{'value': r[0], 'count': r[1]} for r in rows]
        except Exception as e:
            logger.error(f"[DOCS] get_classification_groups failed: {e}")
            return []

    def update_file_path(self, doc_id: str, new_path: str) -> None:
        """Update the stored ``file_path`` for a document.

        Used during folder-watcher scans when a rename is detected (same hash,
        new path).

        Args:
            doc_id: Eight-character hex document identifier.
            new_path: New absolute or relative file path string.
        """
        try:
            def _update_path(did=doc_id, path=new_path, db=self.db):
                """Persist the renamed file path on the write-queue thread."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE documents SET file_path = ?, updated_at = datetime('now') WHERE id = ?",
                        (path, did),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_update_path)
        except Exception as e:
            logger.error(f"[DOCS] update_file_path failed: {e}")

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a documents table row tuple to a document dict.

        Args:
            row: sqlite3 row object with exactly 27 positional columns matching
                the SELECT column order used throughout this service.

        Returns:
            Document dict with keys matching the ``documents`` table schema plus
            boolean ``meta_locked``.
        """
        return {
            'id': row[0],
            'original_name': row[1],
            'mime_type': row[2],
            'file_size_bytes': row[3],
            'file_path': row[4],
            'file_hash': row[5],
            'page_count': row[6],
            'status': row[7],
            'error_message': row[8],
            'chunk_count': row[9],
            'source_type': row[10],
            'tags': row[11] or [],
            'summary': row[12],
            'extracted_metadata': row[13] or {},
            'supersedes_id': row[14],
            'clean_text': row[15],
            'language': row[16],
            'fingerprint': row[17],
            'doc_category': row[18],
            'doc_project': row[19],
            'doc_date': row[20],
            'meta_locked': bool(row[21]),
            'watched_folder_id': row[22],
            'created_at': row[23],
            'updated_at': row[24],
            'deleted_at': row[25],
            'purge_after': row[26],
        }

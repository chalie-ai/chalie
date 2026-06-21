
import hashlib
import json
import logging
import os
import secrets
import shutil
import sqlite3
from datetime import timedelta
from typing import Optional, List, Dict, cast

from services.database_service import DatabaseService
from services.file_mapper_service import FileMapperService
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



class DocumentService:

    def __init__(self, db_service: DatabaseService) -> None:
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
        watched_folder_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        doc_id = doc_id or secrets.token_hex(4)

        try:
            _params = (
                doc_id, original_name, mime_type, file_size, file_path,
                file_hash, source_type, watched_folder_id,
            )

            def _insert(params: tuple[object, ...] = _params, db: DatabaseService = self.db) -> None:
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

    def get_document(self, doc_id: str) -> Optional[Dict[str, object]]:
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

    def get_all_documents(self, include_deleted: bool = False) -> List[Dict[str, object]]:
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

    def search_documents_metadata(self, query: str) -> List[Dict[str, object]]:
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
        doc_id = secrets.token_hex(4)

        # Write markdown to disk — strip any path components from the filename
        # to prevent directory traversal via a crafted original_name.
        safe_name = os.path.basename(original_name) or "document.md"
        doc_dir = FileMapperService.get_documents_path(doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        file_path = str(FileMapperService.get_documents_path(doc_id, safe_name))
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

        def _set_clean(did: str = doc_id, txt: str = text_content, db: DatabaseService = self.db) -> None:
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
        try:
            _params = (status, error_message, chunk_count, doc_id)

            def _update(params: tuple[object, ...] = _params, db: DatabaseService = self.db) -> None:
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

    def update_clean_text(self, doc_id: str, clean_text: str) -> None:
        try:
            def _update(did: str = doc_id, txt: str = clean_text, db: DatabaseService = self.db) -> None:
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE documents SET clean_text = ?, updated_at = datetime('now') WHERE id = ?",
                        (txt, did),
                    )
            self._write_queue.submit_sync(_update)
        except Exception as e:
            logger.error(f"[DOCS] update_clean_text failed: {e}")

    def update_summary(self, doc_id: str, summary: str) -> None:
        try:
            def _update(did: str = doc_id, s: str = summary, db: DatabaseService = self.db) -> None:
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
        metadata: dict[str, object],
        summary: str,
        summary_embedding: Optional[list[float]] = None,
        clean_text: Optional[str] = None,
        language: Optional[str] = None,
        fingerprint: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> None:
        set_parts = ["extracted_metadata = ?", "summary = ?", "updated_at = datetime('now')"]
        params: list[object] = [json.dumps(metadata), summary]

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
            stmt: str = sql,
            p: list[object] = params,
            did: str = doc_id,
            emb: Optional[bytes] = packed_emb,
            db: DatabaseService = self.db,
        ) -> None:
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
        try:
            def _supersede(did: str = doc_id, sid: str = supersedes_id, db: DatabaseService = self.db) -> None:
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
        summary_embedding: Optional[list[float]] = None,
        text_length: int = 0,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
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
        try:
            from services.time_utils import utc_now
            purge_after = utc_now() + timedelta(days=PURGE_WINDOW_DAYS)

            def _soft_delete(did: str = doc_id, pa: object = purge_after, db: DatabaseService = self.db) -> int:
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

            updated = cast(int, self._write_queue.submit_sync(_soft_delete)) > 0

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
        try:
            def _restore(did: str = doc_id, db: DatabaseService = self.db) -> int:
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

            updated = cast(int, self._write_queue.submit_sync(_restore)) > 0

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
        try:
            doc = self.get_document(doc_id)
            if not doc:
                return False

            def _hard_delete(did: str = doc_id, db: DatabaseService = self.db) -> int:
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

            deleted = cast(int, self._write_queue.submit_sync(_hard_delete)) > 0

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
                file_dir = str(FileMapperService.get_documents_path(safe_doc_id))
                resolved = os.path.realpath(file_dir)
                if FileMapperService.validate_document_path(resolved) and os.path.exists(resolved):
                    shutil.rmtree(resolved, ignore_errors=True)
            if deleted:
                logger.info("[DOCS] Hard-deleted document %s", safe(doc_id))

            return deleted

        except Exception as e:
            logger.error(f"[DOCS] hard_delete failed: {e}")
            return False

    def purge_expired(self) -> int:
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

    # ─────────────────────────────────────────────
    # Watched folder helpers
    # ─────────────────────────────────────────────

    def get_documents_by_watched_folder(self, folder_id: str) -> List[Dict[str, object]]:
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

    def update_tags(self, doc_id: str, tags: list[str]) -> None:
        try:
            def _update_tags(did: str = doc_id, t: str = json.dumps(tags), db: DatabaseService = self.db) -> None:
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
        category: Optional[str] = None, project: Optional[str] = None,
        doc_date: Optional[str] = None, lock: bool = False,
    ) -> None:
        try:
            set_parts = ["updated_at = datetime('now')"]
            params: list[object] = []
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
                stmt: str = f"UPDATE documents SET {', '.join(set_parts)} WHERE id = ?",
                p: list[object] = params,
                db: DatabaseService = self.db,
            ) -> None:
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(stmt, p)
                    cursor.close()

            self._write_queue.submit_sync(_update_class)
        except Exception as e:
            logger.error(f"[DOCS] update_classification failed: {e}")

    def get_classification_groups(self, field: str) -> List[Dict[str, object]]:
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
        try:
            def _update_path(did: str = doc_id, path: str = new_path, db: DatabaseService = self.db) -> None:
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

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, object]:
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

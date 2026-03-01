"""
Document Service — Data access layer for document storage, retrieval, and search.

Manages document metadata, chunk storage, soft delete/purge lifecycle,
duplicate detection (hash + semantic), and hybrid search (vector + full-text + keyword boost).

Documents are reference material — retrieved via the document skill (ACT loop),
NOT injected into context assembly.
"""

import logging
import os
import secrets
import shutil
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default similarity thresholds for semantic dedup
DEDUP_EXACT_THRESHOLD = 0.15       # cosine distance < 0.15 = likely same document
DEDUP_REVISION_THRESHOLD = 0.35    # cosine distance < 0.35 = likely revision/update
DEDUP_MIN_TEXT_LENGTH = 200        # skip semantic dedup for very short docs

# Purge window (days after soft delete)
PURGE_WINDOW_DAYS = 30

# Document storage root (inside Docker volume)
DOCUMENTS_ROOT = os.environ.get('DOCUMENTS_ROOT', '/app/data/documents')


class DocumentService:
    """Manages document storage, chunk retrieval, and hybrid search."""

    def __init__(self, db_service):
        self.db = db_service

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
    ) -> str:
        """Create a new document record. Returns doc_id (8-char hex)."""
        doc_id = secrets.token_hex(4)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO documents
                        (id, original_name, mime_type, file_size_bytes, file_path,
                         file_hash, source_type, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (doc_id, original_name, mime_type, file_size, file_path,
                      file_hash, source_type))
                cursor.close()

            logger.info(f"[DOCS] Created document '{original_name}' (id={doc_id})")
            return doc_id

        except Exception as e:
            logger.error(f"[DOCS] create_document failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents WHERE id = %s
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
        """Get all documents, optionally including soft-deleted ones."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if include_deleted:
                    cursor.execute("""
                        SELECT id, original_name, mime_type, file_size_bytes, file_path,
                               file_hash, page_count, status, error_message, chunk_count,
                               source_type, tags, summary, extracted_metadata, supersedes_id,
                               clean_text, language, fingerprint,
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
        """Text search on original_name and tags."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                like_query = f"%{query}%"
                cursor.execute("""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
                           created_at, updated_at, deleted_at, purge_after
                    FROM documents
                    WHERE deleted_at IS NULL
                      AND (LOWER(original_name) LIKE LOWER(%s)
                           OR %s = ANY(tags))
                    ORDER BY created_at DESC
                """, (like_query, query))
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"[DOCS] search_documents_metadata failed: {e}")
            return []

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
        """Update document processing status."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET status = %s, error_message = %s, chunk_count = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (status, error_message, chunk_count, doc_id))
                cursor.close()
            logger.info(f"[DOCS] Updated status for {doc_id}: {status}")
        except Exception as e:
            logger.error(f"[DOCS] update_status failed for {doc_id}: {e}")
            raise

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
        provided (non-None).  This avoids COALESCE casting issues with the
        pgvector ``vector`` type, which rejects ``numeric[]::vector``.

        Raises on failure so the caller (process_document) can mark status='failed'.
        """
        import json
        set_parts = ["extracted_metadata = %s", "summary = %s", "updated_at = NOW()"]
        params = [json.dumps(metadata), summary]

        if summary_embedding is not None:
            set_parts.append("summary_embedding = %s")
            params.append(summary_embedding)
        if clean_text is not None:
            set_parts.append("clean_text = %s")
            params.append(clean_text)
        if language is not None:
            set_parts.append("language = %s")
            params.append(language)
        if fingerprint is not None:
            set_parts.append("fingerprint = %s")
            params.append(fingerprint)
        if page_count is not None:
            set_parts.append("page_count = %s")
            params.append(page_count)

        params.append(doc_id)

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE documents SET {', '.join(set_parts)} WHERE id = %s",
                params,
            )
            cursor.close()

    def set_supersedes(self, doc_id: str, supersedes_id: str) -> None:
        """Mark a document as replacing an older version."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents SET supersedes_id = %s, updated_at = NOW()
                    WHERE id = %s
                """, (supersedes_id, doc_id))
                cursor.close()
            logger.info(f"[DOCS] Document {doc_id} supersedes {supersedes_id}")
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
                        WHERE file_hash = %s AND deleted_at IS NULL
                          AND (%s IS NULL OR id != %s)
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
                    cursor.execute("""
                        SELECT id, original_name, created_at,
                               summary_embedding <=> %s::vector AS distance
                        FROM documents
                        WHERE deleted_at IS NULL
                          AND summary_embedding IS NOT NULL
                          AND (%s IS NULL OR id != %s)
                        ORDER BY distance ASC
                        LIMIT 5
                    """, (summary_embedding, exclude_id, exclude_id))
                    for row in cursor.fetchall():
                        dist = float(row[3])
                        if dist < DEDUP_EXACT_THRESHOLD:
                            results.append({
                                'id': row[0],
                                'original_name': row[1],
                                'created_at': row[2],
                                'match_type': 'semantic_exact',
                                'distance': dist,
                            })
                        elif dist < DEDUP_REVISION_THRESHOLD:
                            results.append({
                                'id': row[0],
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
        """Soft-delete a document (30-day purge window)."""
        try:
            purge_after = datetime.now(timezone.utc) + timedelta(days=PURGE_WINDOW_DAYS)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET deleted_at = NOW(), purge_after = %s, updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (purge_after, doc_id))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                logger.info(f"[DOCS] Soft-deleted document {doc_id}")
            return updated

        except Exception as e:
            logger.error(f"[DOCS] soft_delete failed: {e}")
            return False

    def restore(self, doc_id: str) -> bool:
        """Restore a soft-deleted document."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET deleted_at = NULL, purge_after = NULL, updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NOT NULL
                """, (doc_id,))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                logger.info(f"[DOCS] Restored document {doc_id}")
            return updated

        except Exception as e:
            logger.error(f"[DOCS] restore failed: {e}")
            return False

    def hard_delete(self, doc_id: str) -> bool:
        """Permanently delete a document and its file from disk."""
        try:
            doc = self.get_document(doc_id)
            if not doc:
                return False

            # Delete from database (CASCADE removes chunks)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                deleted = cursor.rowcount > 0
                cursor.close()

            # Delete file from disk
            if deleted and doc.get('file_path'):
                file_dir = os.path.join(DOCUMENTS_ROOT, doc_id)
                if os.path.exists(file_dir):
                    shutil.rmtree(file_dir, ignore_errors=True)
                logger.info(f"[DOCS] Hard-deleted document {doc_id}")

            return deleted

        except Exception as e:
            logger.error(f"[DOCS] hard_delete failed: {e}")
            return False

    def purge_expired(self) -> int:
        """Hard-delete all documents past their purge window."""
        try:
            # Find expired docs first (need file paths for disk cleanup)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM documents
                    WHERE purge_after IS NOT NULL AND purge_after < NOW()
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
    # Chunk operations
    # ─────────────────────────────────────────────

    def store_chunks(self, doc_id: str, chunks: List[Dict]) -> None:
        """Bulk insert chunks into document_chunks."""
        if not chunks:
            return

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for chunk in chunks:
                    cursor.execute("""
                        INSERT INTO document_chunks
                            (document_id, chunk_index, content, page_number,
                             section_title, token_count, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        doc_id,
                        chunk['chunk_index'],
                        chunk['content'],
                        chunk.get('page_number'),
                        chunk.get('section_title'),
                        chunk.get('token_count'),
                        chunk['embedding'],
                    ))
                cursor.close()

            logger.info(f"[DOCS] Stored {len(chunks)} chunks for document {doc_id}")

        except Exception as e:
            logger.error(f"[DOCS] store_chunks failed: {e}")
            raise

    def get_chunks_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document, ordered by chunk_index."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, document_id, chunk_index, content, page_number,
                           section_title, token_count
                    FROM document_chunks
                    WHERE document_id = %s
                    ORDER BY chunk_index ASC
                """, (doc_id,))
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    'id': row[0],
                    'document_id': row[1],
                    'chunk_index': row[2],
                    'content': row[3],
                    'page_number': row[4],
                    'section_title': row[5],
                    'token_count': row[6],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"[DOCS] get_chunks_for_document failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Hybrid search (vector + full-text + keyword)
    # ─────────────────────────────────────────────

    def search_chunks(
        self,
        query_embedding: list,
        query_text: str,
        limit: int = 5,
        distance_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid 3-signal search across document chunks:
        1. Semantic vector search (HNSW cosine)
        2. Full-text search (tsvector/tsquery)
        3. Exact keyword/numeric boosting

        Results merged via Reciprocal Rank Fusion (RRF, k=60).
        Two-stage: coarse doc-level → fine chunk-level within top docs.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Stage 1: Coarse — find relevant documents via summary_embedding
                cursor.execute("""
                    SELECT id, original_name, created_at,
                           summary_embedding <=> %s::vector AS distance
                    FROM documents
                    WHERE deleted_at IS NULL
                      AND status = 'ready'
                      AND summary_embedding IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT 10
                """, (query_embedding,))
                candidate_docs = cursor.fetchall()

                if not candidate_docs:
                    cursor.close()
                    return []

                doc_ids = [row[0] for row in candidate_docs]
                doc_map = {row[0]: {'name': row[1], 'created_at': row[2]} for row in candidate_docs}

                # Stage 2: Fine — semantic search within candidate docs
                cursor.execute("""
                    SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                           dc.page_number, dc.section_title, dc.token_count,
                           dc.embedding <=> %s::vector AS distance
                    FROM document_chunks dc
                    WHERE dc.document_id = ANY(%s)
                    ORDER BY distance ASC
                    LIMIT %s
                """, (query_embedding, doc_ids, limit * 3))
                semantic_results = cursor.fetchall()

                # Stage 2b: Full-text search within candidate docs
                cursor.execute("""
                    SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                           dc.page_number, dc.section_title, dc.token_count,
                           ts_rank(to_tsvector('english', dc.content),
                                   plainto_tsquery('english', %s)) AS rank
                    FROM document_chunks dc
                    WHERE dc.document_id = ANY(%s)
                      AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (query_text, doc_ids, query_text, limit * 3))
                text_results = cursor.fetchall()

                cursor.close()

            # RRF merge (k=60)
            rrf_scores = {}
            chunk_data = {}

            # Semantic signal
            for rank, row in enumerate(semantic_results):
                chunk_id = row[0]
                distance = float(row[7])
                if distance > distance_threshold:
                    continue
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (60 + rank + 1)
                if chunk_id not in chunk_data:
                    doc_info = doc_map.get(row[1], {})
                    chunk_data[chunk_id] = {
                        'chunk_id': chunk_id,
                        'document_id': row[1],
                        'document_name': doc_info.get('name', ''),
                        'document_created_at': doc_info.get('created_at'),
                        'chunk_index': row[2],
                        'content': row[3],
                        'page_number': row[4],
                        'section_title': row[5],
                        'token_count': row[6],
                        'distance': distance,
                    }

            # Full-text signal
            for rank, row in enumerate(text_results):
                chunk_id = row[0]
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (60 + rank + 1)
                if chunk_id not in chunk_data:
                    doc_info = doc_map.get(row[1], {})
                    chunk_data[chunk_id] = {
                        'chunk_id': chunk_id,
                        'document_id': row[1],
                        'document_name': doc_info.get('name', ''),
                        'document_created_at': doc_info.get('created_at'),
                        'chunk_index': row[2],
                        'content': row[3],
                        'page_number': row[4],
                        'section_title': row[5],
                        'token_count': row[6],
                        'distance': None,
                    }

            # Keyword boost: bonus for chunks containing exact query terms
            query_terms = [t.lower() for t in query_text.split() if len(t) > 2]
            for chunk_id, data in chunk_data.items():
                content_lower = data['content'].lower()
                for term in query_terms:
                    if term in content_lower:
                        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 0.005

            # Sort by RRF score and return top results
            sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
            results = []
            for chunk_id in sorted_ids[:limit]:
                data = chunk_data[chunk_id]
                data['rrf_score'] = rrf_scores[chunk_id]
                results.append(data)

            return results

        except Exception as e:
            logger.error(f"[DOCS] search_chunks failed: {e}")
            return []

    def search_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query documents by extracted_metadata JSONB fields.
        Supported filters: document_type, company, has_expiration.
        """
        try:
            conditions = ["deleted_at IS NULL", "status = 'ready'"]
            params = []

            if 'document_type' in filters:
                conditions.append("extracted_metadata->'document_type'->>'value' = %s")
                params.append(filters['document_type'])

            if 'company' in filters:
                conditions.append(
                    "EXISTS (SELECT 1 FROM jsonb_array_elements(extracted_metadata->'companies') c "
                    "WHERE c->>'name' ILIKE %s)"
                )
                params.append(f"%{filters['company']}%")

            if filters.get('has_expiration'):
                conditions.append("jsonb_array_length(COALESCE(extracted_metadata->'expiration_dates', '[]'::jsonb)) > 0")

            where = " AND ".join(conditions)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT id, original_name, mime_type, file_size_bytes, file_path,
                           file_hash, page_count, status, error_message, chunk_count,
                           source_type, tags, summary, extracted_metadata, supersedes_id,
                           clean_text, language, fingerprint,
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
    # Internal helpers
    # ─────────────────────────────────────────────

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row tuple to a document dict."""
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
            'created_at': row[18],
            'updated_at': row[19],
            'deleted_at': row[20],
            'purge_after': row[21],
        }


import hashlib
import json
import logging
import os
import secrets
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Dict, cast

from models.document import DocumentRow
from models.document_meta import DocumentMetaData
from services.database import Database
from services.embedding_utils import pack_embedding as _pack_embedding
from services.file_mapper_service import FileMapperService
from services.log_utils import safe
from services.time_utils import utc_now
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)

# Default similarity thresholds for semantic dedup
DEDUP_EXACT_THRESHOLD = 0.15       # cosine distance < 0.15 = likely same document
DEDUP_REVISION_THRESHOLD = 0.35    # cosine distance < 0.35 = likely revision/update
DEDUP_MIN_TEXT_LENGTH = 200        # skip semantic dedup for very short docs

# Purge window (days after soft delete)
PURGE_WINDOW_DAYS = 30



class DocumentService:

    def __init__(self) -> None:
        self._write_queue = get_write_queue()

    @staticmethod
    def resolve_file_path(doc: "dict[str, object]") -> "str | None":
        """Resolve the on-disk path for a document, validating uploads stay in-root."""
        if doc.get("watched_folder_id"):
            full_path = cast(str, doc["file_path"])
            return full_path if os.path.isfile(os.path.realpath(full_path)) else None
        full_path = str(FileMapperService.get_documents_path(cast(str, doc["file_path"])))
        return full_path if FileMapperService.validate_document_path(full_path) else None

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
            def _insert(
                did: str = doc_id,
                name: str = original_name,
                mime: str = mime_type,
                size: int = file_size,
                path: str = file_path,
                fhash: str = file_hash,
                stype: str = source_type,
                wfid: Optional[str] = watched_folder_id,
            ) -> None:
                with Database.transaction():
                    DocumentMetaData.create(did, name, mime, size, path, fhash, stype, wfid)

            # submit_sync so any DB error propagates via raise below
            self._write_queue.submit_sync(_insert)

            logger.info(f"[DOCS] Created document '{original_name}' (id={doc_id})")
            return doc_id

        except Exception as e:
            logger.error(f"[DOCS] create_document failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, object]]:
        try:
            doc = DocumentMetaData.filter("id", doc_id).first()
            if doc is None:
                return None
            return self._doc_to_dict(doc)

        except Exception as e:
            logger.error(f"[DOCS] get_document failed: {e}")
            return None

    def get_all_documents(self, include_deleted: bool = False) -> List[Dict[str, object]]:
        try:
            query = DocumentMetaData.all() if include_deleted else DocumentMetaData.live()
            docs = query.order_by("created_at DESC").get()
            return [self._doc_to_dict(doc) for doc in docs]

        except Exception as e:
            logger.error(f"[DOCS] get_all_documents failed: {e}")
            return []

    def search_documents_metadata(self, query: str) -> List[Dict[str, object]]:
        try:
            docs = DocumentMetaData.search(query)
            return [self._doc_to_dict(doc) for doc in docs]

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

        self.update_clean_text(doc_id, text_content)

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
            def _update(
                did: str = doc_id,
                s: str = status,
                err: Optional[str] = error_message,
                cc: int = chunk_count,
            ) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(
                        did, {"status": s, "error_message": err, "chunk_count": cc}
                    )

            self._write_queue.submit_sync(_update)
            logger.info("[DOCS] Updated status for %s: %s", safe(doc_id), safe(status))
        except Exception as e:
            logger.error("[DOCS] update_status failed for %s: %s", safe(doc_id), e)
            raise

    def update_clean_text(self, doc_id: str, clean_text: str) -> None:
        try:
            def _update(did: str = doc_id, txt: str = clean_text) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, {"clean_text": txt})
            self._write_queue.submit_sync(_update)
        except Exception as e:
            logger.error(f"[DOCS] update_clean_text failed: {e}")

    def update_summary(self, doc_id: str, summary: str) -> None:
        try:
            def _update(did: str = doc_id, s: str = summary) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, {"summary": s})
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
        fields: dict[str, object] = {
            "extracted_metadata": json.dumps(metadata),
            "summary": summary,
        }
        if clean_text is not None:
            fields["clean_text"] = clean_text
        if language is not None:
            fields["language"] = language
        if fingerprint is not None:
            fields["fingerprint"] = fingerprint
        if page_count is not None:
            fields["page_count"] = page_count

        packed_emb = _pack_embedding(summary_embedding) if summary_embedding is not None else None

        def _update_meta(
            did: str = doc_id,
            f: dict[str, object] = fields,
            emb: Optional[bytes] = packed_emb,
        ) -> None:
            with Database.transaction():
                DocumentMetaData.update_fields(did, f)
                if emb is not None:
                    DocumentMetaData.set_embedding(did, emb)

        self._write_queue.submit_sync(_update_meta)

    def set_supersedes(self, doc_id: str, supersedes_id: str) -> None:
        try:
            def _supersede(did: str = doc_id, sid: str = supersedes_id) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, {"supersedes_id": sid})

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
            # Layer 1: exact hash match
            if file_hash:
                for row in DocumentMetaData.by_file_hash(file_hash, exclude_id):
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
                for row in DocumentMetaData.semantic_matches(cast(bytes, packed), k=5):
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

        except Exception as e:
            logger.error(f"[DOCS] find_duplicates failed: {e}")

        return results

    # ─────────────────────────────────────────────
    # Soft delete / restore / purge
    # ─────────────────────────────────────────────

    def soft_delete(self, doc_id: str) -> bool:
        try:
            purge_after = utc_now() + timedelta(days=PURGE_WINDOW_DAYS)

            def _soft_delete(did: str = doc_id, pa: datetime = purge_after) -> int:
                with Database.transaction():
                    return DocumentMetaData.soft_delete(did, pa)

            updated = cast(int, self._write_queue.submit_sync(_soft_delete)) > 0

            if updated:
                logger.info("[DOCS] Soft-deleted document %s", safe(doc_id))
                # Deactivate data_graph artifacts so they stop surfacing in recall.
                try:
                    DocumentRow.set_fragments_active(doc_id, 0)
                except Exception as exc:
                    logger.warning("[DOCS] Failed to deactivate artifacts for %s: %s", safe(doc_id), exc)
            return updated

        except Exception as e:
            logger.error(f"[DOCS] soft_delete failed: {e}")
            return False

    def restore(self, doc_id: str) -> bool:
        try:
            def _restore(did: str = doc_id) -> int:
                with Database.transaction():
                    return DocumentMetaData.restore(did)

            updated = cast(int, self._write_queue.submit_sync(_restore)) > 0

            if updated:
                logger.info("[DOCS] Restored document %s", safe(doc_id))
                # Reactivate data_graph artifacts so they surface in recall again.
                try:
                    DocumentRow.set_fragments_active(doc_id, 1)
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

            def _hard_delete(did: str = doc_id) -> int:
                with Database.transaction():
                    return DocumentMetaData.hard_delete(did)

            deleted = cast(int, self._write_queue.submit_sync(_hard_delete)) > 0

            # Cascade-delete data_graph artifacts for this document.
            if deleted:
                try:
                    DocumentRow.purge_by_document_id(doc_id)
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
            expired_ids = DocumentMetaData.expired_ids()

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
            docs = (
                DocumentMetaData.filter("watched_folder_id", folder_id)
                .order_by("created_at DESC")
                .get()
            )
            return [self._doc_to_dict(doc) for doc in docs]
        except Exception as e:
            logger.error(f"[DOCS] get_documents_by_watched_folder failed: {e}")
            return []

    def update_tags(self, doc_id: str, tags: list[str]) -> None:
        try:
            def _update_tags(did: str = doc_id, t: str = json.dumps(tags)) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, {"tags": t})

            self._write_queue.submit_sync(_update_tags)
        except Exception as e:
            logger.error(f"[DOCS] update_tags failed: {e}")

    def update_classification(
        self, doc_id: str,
        category: Optional[str] = None, project: Optional[str] = None,
        doc_date: Optional[str] = None, lock: bool = False,
    ) -> None:
        try:
            fields: dict[str, object] = {}
            if category is not None:
                fields["doc_category"] = category
            if project is not None:
                fields["doc_project"] = project
            if doc_date is not None:
                fields["doc_date"] = doc_date
            if lock:
                fields["meta_locked"] = 1

            def _update_class(did: str = doc_id, f: dict[str, object] = fields) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, f)

            self._write_queue.submit_sync(_update_class)
        except Exception as e:
            logger.error(f"[DOCS] update_classification failed: {e}")

    def get_classification_groups(self, field: str) -> List[Dict[str, object]]:
        if field not in ('doc_category', 'doc_project', 'doc_date'):
            return []
        try:
            groups = DocumentMetaData.classification_groups(field)
            return [{'value': grp, 'count': cnt} for grp, cnt in groups]
        except Exception as e:
            logger.error(f"[DOCS] get_classification_groups failed: {e}")
            return []

    def update_file_path(self, doc_id: str, new_path: str) -> None:
        try:
            def _update_path(did: str = doc_id, path: str = new_path) -> None:
                with Database.transaction():
                    DocumentMetaData.update_fields(did, {"file_path": path})

            self._write_queue.submit_sync(_update_path)
        except Exception as e:
            logger.error(f"[DOCS] update_file_path failed: {e}")

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _doc_to_dict(self, doc: DocumentMetaData) -> Dict[str, object]:
        return {
            'id': doc.id,
            'original_name': doc.original_name,
            'mime_type': doc.mime_type,
            'file_size_bytes': doc.file_size_bytes,
            'file_path': doc.file_path,
            'file_hash': doc.file_hash,
            'page_count': doc.page_count,
            'status': doc.status,
            'error_message': doc.error_message,
            'chunk_count': doc.chunk_count,
            'source_type': doc.source_type,
            'tags': json.loads(doc.tags) if doc.tags else [],
            'summary': doc.summary,
            'extracted_metadata': json.loads(doc.extracted_metadata) if doc.extracted_metadata else {},
            'supersedes_id': doc.supersedes_id,
            'clean_text': doc.clean_text,
            'language': doc.language,
            'fingerprint': doc.fingerprint,
            'doc_category': doc.doc_category,
            'doc_project': doc.doc_project,
            'doc_date': doc.doc_date,
            'meta_locked': bool(doc.meta_locked),
            'watched_folder_id': doc.watched_folder_id,
            'created_at': doc.created_at,
            'updated_at': doc.updated_at,
            'deleted_at': doc.deleted_at,
            'purge_after': doc.purge_after,
        }

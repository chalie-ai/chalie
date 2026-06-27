"""
Documents API — Upload, search, and manage documents.

Routes (all require session auth):
  POST   /documents/upload         — multipart file upload
  GET    /documents                — list all documents
  GET    /documents/<id>           — document metadata + first N chunks
  GET    /documents/<id>/content   — full extracted text (paginated)
  GET    /documents/<id>/download  — download original file
  DELETE /documents/<id>           — soft delete
  POST   /documents/<id>/restore  — undo soft delete
  DELETE /documents/<id>/purge    — immediate hard delete
  GET    /documents/search         — semantic search across chunks
  POST   /documents/<id>/confirm  — confirm document after synthesis review
  POST   /documents/<id>/augment  — add user context and confirm

  GET    /documents/watched-folders           — list watched folders
  POST   /documents/watched-folders           — add watched folder
  PUT    /documents/watched-folders/<id>      — update watched folder
  DELETE /documents/watched-folders/<id>      — remove watched folder
  POST   /documents/watched-folders/<id>/scan — trigger immediate scan
  POST   /documents/watched-folders/browse    — browse host directories
"""

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask import request, send_file
from flask.typing import ResponseReturnValue

from flask_restx import Namespace, Resource
from services.file_mapper_service import FileMapperService
from services.filename_utils import safe_filename
from .auth import require_session

if TYPE_CHECKING:
    from services.document_service import DocumentService
    from services.folder_watcher_service import FolderWatcherService

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"
_ERR_FILE_NOT_FOUND = "File not found on disk"

documents_bp = Namespace("documents", description="Document operations", path="/documents")

# Fallback when an uploaded name reduces to nothing safe (e.g. ".." or
# non-ASCII-only). secure_filename returns '' in those cases.
_FALLBACK_DOCUMENT_NAME = 'unnamed_document'

# Max upload size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed MIME types
ALLOWED_MIMES = {
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/html',
    'text/plain',
    'text/markdown',
    'text/css',
    'text/csv',
    'text/xml',
    'application/json',
    'application/xml',
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
}

# Allowed extensions (fallback for MIME detection)
ALLOWED_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.html', '.htm', '.txt', '.md',
    '.css', '.csv', '.xml', '.json',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.rb',
    '.jpg', '.jpeg', '.png', '.webp', '.gif',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_document_service() -> "DocumentService":
    from services.database_service import get_shared_db_service
    from services.document_service import DocumentService
    return DocumentService(get_shared_db_service())


def _serialize_dt(val: object) -> object:
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize_doc(doc: "dict[str, object]") -> "dict[str, object]":
    """Strips ``clean_text`` from list responses (too large)."""
    out = dict(doc)
    for field in ('created_at', 'updated_at', 'deleted_at', 'purge_after'):
        if field in out:
            out[field] = _serialize_dt(out[field])
    # Don't send clean_text in list responses (too large)
    out.pop('clean_text', None)
    return out


def _sanitize_filename(name: str) -> str:
    return safe_filename(name) or _FALLBACK_DOCUMENT_NAME


def _validate_file_path(full_path: str) -> bool:
    """Ensure resolved path is within the documents root (prevent symlink attacks)."""
    return FileMapperService.validate_document_path(full_path)


def _read_existing_metadata(svc: "DocumentService", doc_id: str) -> "dict[str, object]":
    """Read prior extracted_metadata so concurrent writes are not clobbered."""
    existing = svc.get_document(doc_id) or {}
    meta = existing.get('extracted_metadata') or {}
    if isinstance(meta, str):
        try:
            return cast("dict[str, object]", json.loads(meta))
        except Exception:
            return {}
    return cast("dict[str, object]", meta)


def _derive_summary(text: str) -> str:
    """Extract up to a 500-char summary, truncated at the last sentence boundary after 200 chars."""
    summary = text[:500]
    dot_pos = summary.rfind('. ')
    if dot_pos > 200:
        return summary[:dot_pos + 1]
    return summary


def _mark_upload_failed(doc_id: str, error: str) -> None:
    """Best-effort status update to 'failed' — swallow errors since we're already in a failure path."""
    try:
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service
        DocumentService(get_shared_db_service()).update_status(doc_id, 'failed', error[:500])
    except Exception:
        logger.exception(f"[DOCS API] Could not mark {doc_id} as failed")


def _run_upload_extraction(doc_id: str) -> None:
    """Extract text + write artifacts + mark ready. Raises on unrecoverable errors."""
    from services.document_service import DocumentService
    from services.database_service import get_shared_db_service
    from services.text_extractor import extract_text
    from services.document_chunking import create_document_artifacts

    svc = DocumentService(get_shared_db_service())
    doc = svc.get_document(doc_id)
    if not doc:
        return

    file_path = cast(str, doc.get('file_path', ''))
    if not file_path:
        svc.update_status(doc_id, 'failed', 'No file path')
        return

    text = extract_text(str(FileMapperService.get_documents_path(file_path)))
    is_image = cast(str, doc.get('mime_type') or '').startswith('image/')
    if not text:
        if is_image:
            # A textless image with no vision provider (e.g. a photo with no
            # words): persist 'ready' so it stays viewable / re-queryable via the
            # vision tool; there is simply nothing to index. NOT a failure.
            svc.update_status(doc_id, 'ready', chunk_count=0)
            return
        svc.update_status(doc_id, 'failed', 'Text extraction returned empty')
        return

    artifact_count = create_document_artifacts(doc_id, text)

    # Write clean_text so document(action='view') can return full text.
    # Merge with prior metadata to avoid clobbering concurrent
    # synthesis/classification writes.
    svc.update_extracted_metadata(
        doc_id,
        metadata=_read_existing_metadata(svc, doc_id),
        summary=_derive_summary(text),
        clean_text=text,
    )
    svc.update_status(doc_id, 'ready', chunk_count=artifact_count)
    logger.info(f"[DOCS API] Processed upload {doc_id}: {artifact_count} artifacts")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@documents_bp.route("/upload")
@documents_bp.response(201, "Document uploaded")
class UploadDocumentResource(Resource):
    @require_session
    def post(self) -> ResponseReturnValue:
        """Upload and ingest a document."""
        if 'file' not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files['file']
        if not file.filename:
            return {"error": "No filename provided"}, 400

        original_name = _sanitize_filename(file.filename)

        ext = os.path.splitext(original_name)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            return {"error": f"File type '{ext}' is not supported"}, 400

        import uuid
        from services.tmp_storage import new_tmp_path
        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{original_name}")
        try:
            file.save(tmp_path)

            size = os.path.getsize(tmp_path)
            if size > MAX_FILE_SIZE:
                return {"error": f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit"}, 400
            if size == 0:
                return {"error": "File is empty"}, 400

            from abilities.document import ingest_file
            svc = _get_document_service()
            result = ingest_file(svc, tmp_path, name=original_name)
            if result.get("error"):
                logger.error(f"[DOCS API] upload error: {result['error']}")
                return {"error": "Upload failed"}, 500

            doc_id = cast(str, result["id"])
            file_hash = cast(str, result["hash"])

            duplicates = svc.find_duplicates(file_hash, None, 0, exclude_id=doc_id)

            response: dict[str, object] = {
                "id": doc_id,
                "original_name": result["name"],
                "status": result.get("status") or "pending",
                "file_size": result["size"],
                "file_hash": file_hash,
            }

            if duplicates:
                response["duplicates"] = [
                    {
                        "id": d["id"],
                        "original_name": d["original_name"],
                        "match_type": d["match_type"],
                        "created_at": _serialize_dt(d.get("created_at")),
                    }
                    for d in duplicates
                ]

            return response, 201

        except Exception as e:
            logger.error(f"[DOCS API] upload error: {e}")
            return {"error": "Upload failed"}, 500
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@documents_bp.route("")
@documents_bp.response(200, "List of documents")
class ListDocumentsResource(Resource):
    @require_session
    def get(self) -> ResponseReturnValue:
        include_deleted = request.args.get('include_deleted', 'false').lower() == 'true'
        try:
            svc = _get_document_service()
            docs = svc.get_all_documents(include_deleted=include_deleted)
            return {"items": [_serialize_doc(d) for d in docs]}
        except Exception as e:
            logger.error(f"[DOCS API] list error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>")
@documents_bp.response(200, "Document details")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentResource(Resource):
    @require_session
    def get(self, doc_id: str) -> ResponseReturnValue:
        """Get document metadata + first N data_graph artifact previews."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            with dgs.db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, substr(value, 1, 200) as preview FROM data_graph WHERE source=? AND active=1 ORDER BY key LIMIT 5",
                    (f'document:{doc_id}',),
                ).fetchall()
            result = _serialize_doc(doc)
            result['artifacts'] = [{'key': r[0], 'preview': r[1]} for r in rows]
            return {"item": result}
        except Exception as e:
            logger.error(f"[DOCS API] get_document error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def delete(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            ok = svc.soft_delete(doc_id)
            if not ok:
                return {"error": _ERR_NOT_FOUND}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[DOCS API] delete error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/content")
@documents_bp.response(200, "Document content")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentContentResource(Resource):
    @require_session
    def get(self, doc_id: str) -> ResponseReturnValue:
        """Get full document text reconstructed from data_graph artifacts."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            with dgs.db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph WHERE source=? AND active=1 ORDER BY key",
                    (f'document:{doc_id}',),
                ).fetchall()

            artifacts = [{'key': r[0], 'content': r[1]} for r in rows]
            return {
                "document_id": doc_id,
                "total_artifacts": len(artifacts),
                "artifacts": artifacts,
            }
        except Exception as e:
            logger.error(f"[DOCS API] get_content error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/download")
@documents_bp.response(200, "File download")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentDownloadResource(Resource):
    @require_session
    def get(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            if doc.get('watched_folder_id'):
                full_path = cast(str, doc['file_path'])
                if not os.path.isfile(os.path.realpath(full_path)):
                    return {"error": _ERR_FILE_NOT_FOUND}, 404
            else:
                full_path = str(FileMapperService.get_documents_path(cast(str, doc['file_path'])))
                if not _validate_file_path(full_path) or not os.path.exists(full_path):
                    return {"error": _ERR_FILE_NOT_FOUND}, 404

            return send_file(
                full_path,
                mimetype=cast(str, doc['mime_type']),
                as_attachment=True,
                download_name=cast(str, doc['original_name']),
            )
        except Exception as e:
            logger.error(f"[DOCS API] download error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/preview")
@documents_bp.response(200, "File preview")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentPreviewResource(Resource):
    @require_session
    def get(self, doc_id: str) -> ResponseReturnValue:
        """Stream file for inline browser preview."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            if doc.get('watched_folder_id'):
                full_path = cast(str, doc['file_path'])
                if not os.path.isfile(os.path.realpath(full_path)):
                    return {"error": _ERR_FILE_NOT_FOUND}, 404
            else:
                full_path = str(FileMapperService.get_documents_path(cast(str, doc['file_path'])))
                if not _validate_file_path(full_path) or not os.path.exists(full_path):
                    return {"error": _ERR_FILE_NOT_FOUND}, 404

            return send_file(
                full_path,
                mimetype=cast(str, doc['mime_type']),
                as_attachment=False,
                download_name=cast(str, doc['original_name']),
            )
        except Exception as e:
            logger.error(f"[DOCS API] preview error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/classify")
@documents_bp.response(200, "Classification updated")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentClassifyResource(Resource):
    @require_session
    def put(self, doc_id: str) -> ResponseReturnValue:
        """Update document classification metadata."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            data = request.get_json() or {}
            svc.update_classification(
                doc_id,
                category=data.get('category'),
                project=data.get('project'),
                doc_date=data.get('date'),
                lock=True,
            )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"[DOCS API] classify update error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/groups/<field>")
@documents_bp.response(200, "Classification groups")
@documents_bp.param("field", "str", "Classification field name")
class DocumentGroupsResource(Resource):
    @require_session
    def get(self, field: str) -> ResponseReturnValue:
        """Get unique classification groups for a field."""
        try:
            svc = _get_document_service()
            groups = svc.get_classification_groups(field)
            return {"groups": groups}
        except Exception as e:
            logger.error(f"[DOCS API] groups error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/restore")
@documents_bp.response(200, "Document restored")
@documents_bp.response(404, "Not found or not deleted")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentRestoreResource(Resource):
    @require_session
    def post(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            ok = svc.restore(doc_id)
            if not ok:
                return {"error": "Not found or not deleted"}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[DOCS API] restore error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/purge")
@documents_bp.response(200, "Document purged")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentPurgeResource(Resource):
    @require_session
    def delete(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            ok = svc.hard_delete(doc_id)
            if not ok:
                return {"error": _ERR_NOT_FOUND}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[DOCS API] purge error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/search")
@documents_bp.response(200, "Search results")
class DocumentSearchResource(Resource):
    @require_session
    def get(self) -> ResponseReturnValue:
        """Search across document artifacts in data_graph."""
        q_raw = request.args.get('q', None)
        if q_raw is None:
            return {"error": "Query parameter 'q' is required"}, 400
        query = q_raw.strip()
        if not query:
            return {"error": "Query cannot be empty"}, 400

        limit = min(request.args.get('limit', 5, type=int), 20)

        try:
            from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT

            dgs = get_data_graph_service()
            results = dgs.recall(query, kinds=[KIND_DOCUMENT], limit=limit)

            serialized = []
            for row in results:
                source = cast(str, row.get('source', '') or '')
                doc_id = source.split(':', 1)[1] if source.startswith('document:') else ''
                serialized.append({
                    'document_id': doc_id,
                    'key': row.get('key', ''),
                    'content': row.get('value', ''),
                    'source': source,
                })

            return {"results": serialized, "query": query}
        except Exception as e:
            logger.error(f"[DOCS API] search error: {e}")
            return {"error": "Search failed"}, 500


@documents_bp.route("/<doc_id>/confirm")
@documents_bp.response(200, "Document confirmed")
@documents_bp.response(400, "Not awaiting confirmation")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentConfirmResource(Resource):
    @require_session
    def post(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            if doc['status'] != 'awaiting_confirmation':
                return {"error": "Document is not awaiting confirmation"}, 400

            svc.update_status(doc_id, 'ready', chunk_count=cast(int, doc.get('chunk_count', 0)))
            return {"ok": True, "status": "ready"}
        except Exception as e:
            logger.error(f"[DOCS API] confirm error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/augment")
@documents_bp.response(200, "Document augmented")
@documents_bp.response(400, "Invalid state or missing context")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentAugmentResource(Resource):
    @require_session
    def post(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return {"error": _ERR_NOT_FOUND}, 404

            if doc['status'] not in ('awaiting_confirmation', 'ready'):
                return {"error": "Document cannot be augmented in its current state"}, 400

            data = request.get_json(silent=True) or {}
            context = (data.get('context') or '').strip()
            if not context:
                return {"error": "Field 'context' is required"}, 400

            metadata = cast("dict[str, object]", doc.get('extracted_metadata') or {})
            metadata['_user_context'] = context

            svc.update_extracted_metadata(
                doc_id,
                metadata=metadata,
                summary=cast(str, doc.get('summary', '')),
                summary_embedding=cast("list[float] | None", doc.get('summary_embedding')),
            )

            if doc['status'] != 'ready':
                svc.update_status(doc_id, 'ready', chunk_count=cast(int, doc.get('chunk_count', 0)))
            return {"ok": True, "status": "ready"}
        except Exception as e:
            logger.error(f"[DOCS API] augment error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/<doc_id>/supersede")
@documents_bp.response(200, "Document superseded")
@documents_bp.response(400, "Missing old_id")
@documents_bp.response(404, "Not found")
@documents_bp.param("doc_id", "str", "Document identifier")
class DocumentSupersedeResource(Resource):
    @require_session
    def post(self, doc_id: str) -> ResponseReturnValue:
        try:
            svc = _get_document_service()

            new_doc = svc.get_document(doc_id)
            if not new_doc:
                return {"error": "New document not found"}, 404

            data = request.get_json(silent=True) or {}
            old_id = (data.get('old_id') or '').strip()
            if not old_id:
                return {"error": "Field 'old_id' is required"}, 400

            old_doc = svc.get_document(old_id)
            if not old_doc:
                return {"error": "Old document not found"}, 404

            svc.set_supersedes(doc_id, old_id)
            svc.soft_delete(old_id)

            return {"ok": True, "supersedes_id": old_id}
        except Exception as e:
            logger.error(f"[DOCS API] supersede error: {e}")
            return {"error": _ERR_INTERNAL}, 500


# ---------------------------------------------------------------------------
# Watched Folders
# ---------------------------------------------------------------------------

def _get_watcher_service() -> "FolderWatcherService":
    from services.database_service import get_shared_db_service
    from services.folder_watcher_service import FolderWatcherService
    return FolderWatcherService(get_shared_db_service())


@documents_bp.route("/watched-folders")
@documents_bp.response(200, "List of watched folders")
@documents_bp.response(201, "Watched folder created")
class WatchedFoldersResource(Resource):
    @require_session
    def get(self) -> ResponseReturnValue:
        try:
            svc = _get_watcher_service()
            folders = svc.get_all_folders()
            return {"items": folders}
        except Exception as e:
            logger.error(f"[DOCS API] list watched folders error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def post(self) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        folder_path = (data.get('folder_path') or '').strip()

        if not folder_path:
            return {"error": "Field 'folder_path' is required"}, 400

        try:
            svc = _get_watcher_service()
            folder = svc.create_folder(
                folder_path=folder_path,
                label=data.get('label'),
                file_patterns=data.get('file_patterns'),
                ignore_patterns=data.get('ignore_patterns'),
                recursive=data.get('recursive', True),
                scan_interval=data.get('scan_interval', 300),
            )
            return {"item": folder}, 201
        except ValueError as e:
            return {"error": str(e)}, 400
        except PermissionError as e:
            return {"error": str(e)}, 403
        except Exception as e:
            if 'UNIQUE constraint' in str(e):
                return {"error": "This folder is already being watched"}, 409
            logger.error(f"[DOCS API] create watched folder error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/watched-folders/<folder_id>")
@documents_bp.response(200, "Watched folder updated")
@documents_bp.response(404, "Not found")
@documents_bp.param("folder_id", "str", "Folder identifier")
class WatchedFolderResource(Resource):
    @require_session
    def put(self, folder_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        try:
            svc = _get_watcher_service()
            folder = svc.get_folder(folder_id)
            if not folder:
                return {"error": _ERR_NOT_FOUND}, 404

            updated = svc.update_folder(folder_id, **data)
            return {"item": updated}
        except (ValueError, PermissionError) as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"[DOCS API] update watched folder error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def delete(self, folder_id: str) -> ResponseReturnValue:
        delete_documents = request.args.get('delete_documents', 'false').lower() == 'true'
        try:
            svc = _get_watcher_service()
            ok = svc.delete_folder(folder_id, delete_documents=delete_documents)
            if not ok:
                return {"error": _ERR_NOT_FOUND}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[DOCS API] delete watched folder error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/watched-folders/<folder_id>/scan")
@documents_bp.response(200, "Scan triggered")
@documents_bp.response(404, "Not found")
@documents_bp.param("folder_id", "str", "Folder identifier")
class WatchedFolderScanResource(Resource):
    @require_session
    def post(self, folder_id: str) -> ResponseReturnValue:
        try:
            svc = _get_watcher_service()
            folder = svc.get_folder(folder_id)
            if not folder:
                return {"error": _ERR_NOT_FOUND}, 404

            svc.trigger_scan(folder_id)
            return {"ok": True, "message": "Scan requested"}
        except Exception as e:
            logger.error(f"[DOCS API] trigger scan error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@documents_bp.route("/watched-folders/browse")
@documents_bp.response(200, "Directory listing")
class WatchedFolderBrowseResource(Resource):
    @require_session
    def post(self) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        path = data.get('path')

        try:
            svc = _get_watcher_service()
            result = svc.browse_directory(path)
            return result
        except ValueError as e:
            return {"error": str(e)}, 400
        except PermissionError as e:
            return {"error": str(e)}, 403
        except Exception as e:
            logger.error(f"[DOCS API] browse error: {e}")
            return {"error": _ERR_INTERNAL}, 500

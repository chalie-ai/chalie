"""Documents namespace — upload, search, lifecycle, and watched-folder CRUD.

Two resource families (documents, watched-folders) plus non-CRUD operations
(file I/O, semantic search, lifecycle transitions) that stay as named endpoints
but are DTO-typed through the foundation boundary decorators
(``@expects``/``@responds``). Mutation/lifecycle endpoints return 204 no-body on
success; errors normalize to 404/403/409/422. No hand-woven JSON envelopes or
local datetime serializers — DTOs own the wire shape.

Routes (all require session auth):
  POST   /api/documents/upload                  — multipart file upload
  GET    /api/documents                         — list all documents
  GET    /api/documents/<id>                    — document metadata + artifact previews
  DELETE /api/documents/<id>                    — soft delete
  GET    /api/documents/<id>/content            — full extracted text
  GET    /api/documents/<id>/download           — download original file
  GET    /api/documents/<id>/preview            — inline file preview
  PUT    /api/documents/<id>/classify           — partial classification update
  GET    /api/documents/groups/<field>          — classification groupings
  POST   /api/documents/<id>/restore           — undo soft delete
  DELETE /api/documents/<id>/purge             — immediate hard delete
  GET    /api/documents/search                  — semantic search across artifacts
  POST   /api/documents/<id>/confirm           — confirm after synthesis review
  POST   /api/documents/<id>/augment           — add user context and confirm
  POST   /api/documents/<id>/supersede         — mark as superseding a prior doc
  GET    /api/documents/watched-folders         — list watched folders
  POST   /api/documents/watched-folders         — add watched folder
  PUT    /api/documents/watched-folders/<id>    — update watched folder
  DELETE /api/documents/watched-folders/<id>    — remove watched folder
  POST   /api/documents/watched-folders/<id>/scan — trigger immediate scan
  POST   /api/documents/watched-folders/browse  — browse host directories
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias, cast

from flask import request, send_file
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.file_mapper_service import FileMapperService
from services.filename_utils import safe_filename
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.document import (
    ArtifactContent,
    ArtifactPreview,
    AugmentRequest,
    ClassifyRequest,
    ClassificationGroup,
    Document,
    DocumentContent,
    DocumentDetail,
    DuplicateRef,
    SearchQuery,
    SearchResponse,
    SearchResult,
    SupersedeRequest,
    SupersedeResponse,
    UploadRequest,
    UploadResponse,
)
from .dto.watched_folder import (
    BrowseRequest,
    BrowseResponse,
    WatchedFolder,
    WatchedFolderCreate,
    WatchedFolderUpdate,
)

if TYPE_CHECKING:
    from services.document_service import DocumentService
    from services.folder_watcher_service import FolderWatcherService

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"
_ERR_FILE_NOT_FOUND = "File not found on disk"
_ERR_ALREADY_WATCHED = "This folder is already being watched"

_OptDatetime: TypeAlias = "datetime | None"
_OptStr: TypeAlias = "str | None"
_StrList: TypeAlias = "list[str]"

documents_ns = Namespace("documents", description="Document operations", path="/api/documents")

register_dto(
    documents_ns,
    Document,
    DocumentDetail,
    ArtifactPreview,
    ArtifactContent,
    DocumentContent,
    DuplicateRef,
    UploadResponse,
    UploadRequest,
    ClassifyRequest,
    AugmentRequest,
    SupersedeRequest,
    SupersedeResponse,
    ClassificationGroup,
    SearchResult,
    SearchResponse,
    SearchQuery,
    WatchedFolder,
    WatchedFolderCreate,
    WatchedFolderUpdate,
    BrowseRequest,
    BrowseResponse,
    Error,
)

_D = documents_ns.models

# Fallback when an uploaded name reduces to nothing safe (e.g. ".." or
# non-ASCII-only). secure_filename returns '' in those cases.
_FALLBACK_DOCUMENT_NAME = "unnamed_document"

# Max upload size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md",
    ".css", ".csv", ".xml", ".json",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb",
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
}


# ---------------------------------------------------------------------------
# Service accessors
# ---------------------------------------------------------------------------

def _get_document_service() -> "DocumentService":
    from services.database_service import get_shared_db_service
    from services.document_service import DocumentService
    return DocumentService(get_shared_db_service())


def _get_watcher_service() -> "FolderWatcherService":
    from services.database_service import get_shared_db_service
    from services.folder_watcher_service import FolderWatcherService
    return FolderWatcherService(get_shared_db_service())



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return safe_filename(name) or _FALLBACK_DOCUMENT_NAME


def _validate_file_path(full_path: str) -> bool:
    """Ensure resolved path is within the documents root (prevent symlink attacks)."""
    return FileMapperService.validate_document_path(full_path)


def _read_existing_metadata(svc: "DocumentService", doc_id: str) -> "dict[str, object]":
    """Read prior extracted_metadata so concurrent writes are not clobbered."""
    existing = svc.get_document(doc_id) or {}
    meta = existing.get("extracted_metadata") or {}
    if isinstance(meta, str):
        try:
            return cast("dict[str, object]", json.loads(meta))
        except Exception:
            return {}
    return cast("dict[str, object]", meta)


def _derive_summary(text: str) -> str:
    """Extract up to a 500-char summary, truncated at the last sentence boundary after 200 chars."""
    summary = text[:500]
    dot_pos = summary.rfind(". ")
    if dot_pos > 200:
        return summary[:dot_pos + 1]
    return summary


def _mark_upload_failed(doc_id: str, error: str) -> None:
    """Best-effort status update to 'failed' — swallow errors since we're already in a failure path."""
    try:
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service
        DocumentService(get_shared_db_service()).update_status(doc_id, "failed", error[:500])
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

    file_path = cast(str, doc.get("file_path", ""))
    if not file_path:
        svc.update_status(doc_id, "failed", "No file path")
        return

    text = extract_text(str(FileMapperService.get_documents_path(file_path)))
    is_image = cast(str, doc.get("mime_type") or "").startswith("image/")
    if not text:
        if is_image:
            # A textless image with no vision provider (e.g. a photo with no
            # words): persist 'ready' so it stays viewable / re-queryable via the
            # vision tool; there is simply nothing to index. NOT a failure.
            svc.update_status(doc_id, "ready", chunk_count=0)
            return
        svc.update_status(doc_id, "failed", "Text extraction returned empty")
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
    svc.update_status(doc_id, "ready", chunk_count=artifact_count)
    logger.info(f"[DOCS API] Processed upload {doc_id}: {artifact_count} artifacts")


# ---------------------------------------------------------------------------
# Documents resource
# ---------------------------------------------------------------------------

@documents_ns.route("/upload")
class UploadDocumentResource(Resource):
    @require_session
    @documents_ns.expect(_D["UploadRequest"])
    @documents_ns.response(201, "Document uploaded", model=_D["UploadResponse"])
    @documents_ns.response(400, "No file / unsupported type / empty / too large", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(UploadResponse, code=201)
    @expects(UploadRequest, source="form")
    def post(self, dto: UploadRequest) -> UploadResponse | ResponseReturnValue:
        """Upload and ingest a document."""
        original_name = _sanitize_filename(dto.file.filename or "")
        ext = os.path.splitext(original_name)[1].lower()
        if not ext:
            return error("File has no extension", 400)
        if ext not in ALLOWED_EXTENSIONS:
            return error(f"File type '{ext}' is not supported", 400)

        from services.tmp_storage import new_tmp_path
        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{original_name}")
        try:
            dto.file.save(tmp_path)

            size = os.path.getsize(tmp_path)
            if size > MAX_FILE_SIZE:
                return error(f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit", 400)
            if size == 0:
                return error("File is empty", 400)

            from abilities.document import ingest_file
            svc = _get_document_service()
            result = ingest_file(svc, tmp_path, name=original_name)
            if result.get("error"):
                logger.error(f"[DOCS API] upload error: {result['error']}")
                return error("Upload failed", 500)

            doc_id = cast(str, result["id"])
            file_hash = cast(str, result["hash"])
            duplicates = svc.find_duplicates(file_hash, None, 0, exclude_id=doc_id)

            return UploadResponse(
                id=doc_id,
                original_name=cast(str, result["name"]),
                status=cast(str, result.get("status") or "pending"),
                file_size=cast(int, result["size"]),
                file_hash=file_hash,
                duplicates=[
                    DuplicateRef(
                        id=cast(str, d["id"]),
                        original_name=cast(str, d["original_name"]),
                        match_type=cast(str, d["match_type"]),
                        created_at=cast(_OptDatetime, d.get("created_at")),
                    )
                    for d in duplicates
                ] or None,
            )
        except Exception as e:
            logger.error(f"[DOCS API] upload error: {e}")
            return error("Upload failed", 500)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@documents_ns.route("")
class ListDocumentsResource(Resource):
    @require_session
    @documents_ns.param("include_deleted", "Include soft-deleted documents", type="boolean")
    @documents_ns.response(200, "All documents", model=_D["Document"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(Document, code=200)
    def get(self) -> list[Document] | ResponseReturnValue:
        """List all documents, optionally including soft-deleted ones."""
        include_deleted = request.args.get("include_deleted", "false").lower() == "true"
        try:
            svc = _get_document_service()
            return [_document_dto(row) for row in svc.get_all_documents(include_deleted=include_deleted)]
        except Exception as e:
            logger.error(f"[DOCS API] list error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>")
class DocumentResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(200, "Document details", model=_D["DocumentDetail"])
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(DocumentDetail, code=200)
    def get(self, doc_id: str) -> DocumentDetail | ResponseReturnValue:
        """Get document metadata + first N data_graph artifact previews."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return error(_ERR_NOT_FOUND, 404)

            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            with dgs.db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, substr(value, 1, 200) as preview FROM data_graph "
                    "WHERE source=? AND active=1 ORDER BY key LIMIT 5",
                    (f"document:{doc_id}",),
                ).fetchall()
            detail = _document_dto(doc)
            return DocumentDetail(
                **detail.model_dump(mode="python"),
                artifacts=[ArtifactPreview(key=r[0], preview=r[1]) for r in rows],
            )
        except Exception as e:
            logger.error(f"[DOCS API] get_document error: {e}")
            return error(_ERR_INTERNAL, 500)

    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(204, "Soft deleted")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def delete(self, doc_id: str) -> None | ResponseReturnValue:
        """Soft delete a document."""
        try:
            if not _get_document_service().soft_delete(doc_id):
                return error(_ERR_NOT_FOUND, 404)
            return None
        except Exception as e:
            logger.error(f"[DOCS API] delete error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/content")
class DocumentContentResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(200, "Document content", model=_D["DocumentContent"])
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(DocumentContent, code=200)
    def get(self, doc_id: str) -> DocumentContent | ResponseReturnValue:
        """Get full document text reconstructed from data_graph artifacts."""
        try:
            svc = _get_document_service()
            if not svc.get_document(doc_id):
                return error(_ERR_NOT_FOUND, 404)

            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            with dgs.db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph "
                    "WHERE source=? AND active=1 ORDER BY key",
                    (f"document:{doc_id}",),
                ).fetchall()

            return DocumentContent(
                document_id=doc_id,
                total_artifacts=len(rows),
                artifacts=[ArtifactContent(key=r[0], content=r[1]) for r in rows],
            )
        except Exception as e:
            logger.error(f"[DOCS API] get_content error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/download")
@documents_ns.produces(["application/octet-stream"])
class DocumentDownloadResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(200, "File download")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=200)
    def get(self, doc_id: str) -> ResponseReturnValue:
        """Download the original file as an attachment."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return error(_ERR_NOT_FOUND, 404)

            full_path = _resolve_document_file(doc)
            if full_path is None or not os.path.exists(full_path):
                return error(_ERR_FILE_NOT_FOUND, 404)

            return send_file(
                full_path,
                mimetype=cast(str, doc["mime_type"]),
                as_attachment=True,
                download_name=cast(str, doc["original_name"]),
            )
        except Exception as e:
            logger.error(f"[DOCS API] download error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/preview")
@documents_ns.produces(["application/octet-stream"])
class DocumentPreviewResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(200, "File preview")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=200)
    def get(self, doc_id: str) -> ResponseReturnValue:
        """Stream the file for inline browser preview."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return error(_ERR_NOT_FOUND, 404)

            full_path = _resolve_document_file(doc)
            if full_path is None or not os.path.exists(full_path):
                return error(_ERR_FILE_NOT_FOUND, 404)

            return send_file(
                full_path,
                mimetype=cast(str, doc["mime_type"]),
                as_attachment=False,
                download_name=cast(str, doc["original_name"]),
            )
        except Exception as e:
            logger.error(f"[DOCS API] preview error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/classify")
class DocumentClassifyResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.expect(_D["ClassifyRequest"])
    @documents_ns.response(204, "Classification updated")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    @expects(ClassifyRequest)
    def put(self, doc_id: str, dto: ClassifyRequest) -> None | ResponseReturnValue:
        """Update document classification metadata."""
        try:
            svc = _get_document_service()
            if not svc.get_document(doc_id):
                return error(_ERR_NOT_FOUND, 404)
            svc.update_classification(
                doc_id,
                category=dto.category,
                project=dto.project,
                doc_date=dto.date,
                lock=True,
            )
            return None
        except Exception as e:
            logger.error(f"[DOCS API] classify update error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/groups/<field>")
@documents_ns.param("field", "Classification field name")
class DocumentGroupsResource(Resource):
    @require_session
    @documents_ns.response(200, "Classification groups", model=_D["ClassificationGroup"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(ClassificationGroup, code=200)
    def get(self, field: str) -> list[ClassificationGroup] | ResponseReturnValue:
        """Get unique classification groups for a field."""
        try:
            svc = _get_document_service()
            return [
                ClassificationGroup(value=cast(str, g["value"]), count=cast(int, g["count"]))
                for g in svc.get_classification_groups(field)
            ]
        except Exception as e:
            logger.error(f"[DOCS API] groups error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/restore")
class DocumentRestoreResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(204, "Document restored")
    @documents_ns.response(404, "Not found or not deleted", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def post(self, doc_id: str) -> None | ResponseReturnValue:
        """Undo a soft delete."""
        try:
            if not _get_document_service().restore(doc_id):
                return error("Not found or not deleted", 404)
            return None
        except Exception as e:
            logger.error(f"[DOCS API] restore error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/purge")
class DocumentPurgeResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(204, "Document purged")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def delete(self, doc_id: str) -> None | ResponseReturnValue:
        """Immediately hard delete a document."""
        try:
            if not _get_document_service().hard_delete(doc_id):
                return error(_ERR_NOT_FOUND, 404)
            return None
        except Exception as e:
            logger.error(f"[DOCS API] purge error: {e}")
            return error(_ERR_INTERNAL, 500)


def _search_result_dto(row: "dict[str, object]") -> SearchResult:
    source = cast(str, row.get("source", "") or "")
    return SearchResult(
        document_id=source.split(":", 1)[1] if source.startswith("document:") else "",
        key=cast(str, row.get("key", "")),
        content=cast(str, row.get("value", "")),
        source=source,
    )


@documents_ns.route("/search")
class DocumentSearchResource(Resource):
    @require_session
    @documents_ns.param("q", "Search query", required=True)
    @documents_ns.param("limit", "Max results (≤20)", type="integer")
    @documents_ns.response(200, "Search results", model=_D["SearchResponse"])
    @documents_ns.response(422, "Validation failed", model=_D["Error"])
    @documents_ns.response(500, "Search failed", model=_D["Error"])
    @responds(SearchResponse, code=200)
    @expects(SearchQuery, source="args")
    def get(self, dto: SearchQuery) -> SearchResponse | ResponseReturnValue:
        """Search across document artifacts in data_graph."""
        try:
            from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT

            dgs = get_data_graph_service()
            results = dgs.recall(dto.q, kinds=[KIND_DOCUMENT], limit=dto.limit)

            return SearchResponse(
                results=[_search_result_dto(row) for row in results],
                query=dto.q,
            )
        except Exception as e:
            logger.error(f"[DOCS API] search error: {e}")
            return error("Search failed", 500)


@documents_ns.route("/<doc_id>/confirm")
class DocumentConfirmResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.response(204, "Document confirmed")
    @documents_ns.response(400, "Not awaiting confirmation", model=_D["Error"])
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def post(self, doc_id: str) -> None | ResponseReturnValue:
        """Confirm a document after synthesis review."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return error(_ERR_NOT_FOUND, 404)
            if doc["status"] != "awaiting_confirmation":
                return error("Document is not awaiting confirmation", 400)
            svc.update_status(doc_id, "ready", chunk_count=cast(int, doc.get("chunk_count", 0)))
            return None
        except Exception as e:
            logger.error(f"[DOCS API] confirm error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/augment")
class DocumentAugmentResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.expect(_D["AugmentRequest"])
    @documents_ns.response(204, "Document augmented")
    @documents_ns.response(400, "Invalid state", model=_D["Error"])
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    @expects(AugmentRequest)
    def post(self, doc_id: str, dto: AugmentRequest) -> None | ResponseReturnValue:
        """Add user context to a document and confirm it."""
        try:
            svc = _get_document_service()
            doc = svc.get_document(doc_id)
            if not doc:
                return error(_ERR_NOT_FOUND, 404)
            if doc["status"] not in ("awaiting_confirmation", "ready"):
                return error("Document cannot be augmented in its current state", 400)

            metadata = cast("dict[str, object]", doc.get("extracted_metadata") or {})
            metadata["_user_context"] = dto.context

            svc.update_extracted_metadata(
                doc_id,
                metadata=metadata,
                summary=cast(str, doc.get("summary", "")),
                summary_embedding=cast("list[float] | None", doc.get("summary_embedding")),
            )

            if doc["status"] != "ready":
                svc.update_status(doc_id, "ready", chunk_count=cast(int, doc.get("chunk_count", 0)))
            return None
        except Exception as e:
            logger.error(f"[DOCS API] augment error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/<doc_id>/supersede")
class DocumentSupersedeResource(Resource):
    @require_session
    @documents_ns.param("doc_id", "Document identifier")
    @documents_ns.expect(_D["SupersedeRequest"])
    @documents_ns.response(200, "Document superseded", model=_D["SupersedeResponse"])
    @documents_ns.response(404, "New/Old document not found", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(SupersedeResponse, code=200)
    @expects(SupersedeRequest)
    def post(self, doc_id: str, dto: SupersedeRequest) -> SupersedeResponse | ResponseReturnValue:
        """Mark the new document as superseding a prior one."""
        try:
            svc = _get_document_service()
            if not svc.get_document(doc_id):
                return error("New document not found", 404)
            if not svc.get_document(dto.old_id):
                return error("Old document not found", 404)
            svc.set_supersedes(doc_id, dto.old_id)
            svc.soft_delete(dto.old_id)
            return SupersedeResponse(ok=True, supersedes_id=dto.old_id)
        except Exception as e:
            logger.error(f"[DOCS API] supersede error: {e}")
            return error(_ERR_INTERNAL, 500)


# ---------------------------------------------------------------------------
# Watched Folders
# ---------------------------------------------------------------------------

@documents_ns.route("/watched-folders")
class WatchedFoldersResource(Resource):
    @require_session
    @documents_ns.response(200, "All watched folders", model=_D["WatchedFolder"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(WatchedFolder, code=200)
    def get(self) -> list[WatchedFolder] | ResponseReturnValue:
        """List all watched folders."""
        try:
            return [_watched_folder_dto(row) for row in _get_watcher_service().get_all_folders()]
        except Exception as e:
            logger.error(f"[DOCS API] list watched folders error: {e}")
            return error(_ERR_INTERNAL, 500)

    @require_session
    @documents_ns.expect(_D["WatchedFolderCreate"])
    @documents_ns.response(201, "Watched folder created", model=_D["WatchedFolder"])
    @documents_ns.response(403, "Permission denied", model=_D["Error"])
    @documents_ns.response(409, _ERR_ALREADY_WATCHED, model=_D["Error"])
    @documents_ns.response(422, "Validation failed", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(WatchedFolder, code=201)
    @expects(WatchedFolderCreate)
    def post(self, dto: WatchedFolderCreate) -> WatchedFolder | ResponseReturnValue:
        """Register a new watched folder."""
        try:
            folder = _get_watcher_service().create_folder(
                folder_path=dto.folder_path,
                label=dto.label,
                file_patterns=dto.file_patterns,
                ignore_patterns=dto.ignore_patterns,
                recursive=dto.recursive,
                scan_interval=dto.scan_interval,
            )
            return _watched_folder_dto(folder)
        except ValueError as e:
            return error(str(e), 422)
        except PermissionError as e:
            return error(str(e), 403)
        except Exception as e:
            if "UNIQUE constraint" in str(e):
                return error(_ERR_ALREADY_WATCHED, 409)
            logger.error(f"[DOCS API] create watched folder error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/watched-folders/<folder_id>")
@documents_ns.param("folder_id", "Folder identifier")
class WatchedFolderResource(Resource):
    @require_session
    @documents_ns.expect(_D["WatchedFolderUpdate"])
    @documents_ns.response(200, "Updated folder", model=_D["WatchedFolder"])
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(403, "Permission denied", model=_D["Error"])
    @documents_ns.response(422, "Validation failed", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(WatchedFolder, code=200)
    @expects(WatchedFolderUpdate)
    def put(self, folder_id: str, dto: WatchedFolderUpdate) -> WatchedFolder | ResponseReturnValue:
        """Partially update a watched folder's mutable fields."""
        try:
            svc = _get_watcher_service()
            if not svc.get_folder(folder_id):
                return error(_ERR_NOT_FOUND, 404)
            updates = dto.model_dump(exclude_none=True)
            updated = svc.update_folder(folder_id, **updates)
            if updated is None:
                return error(_ERR_NOT_FOUND, 404)
            return _watched_folder_dto(updated)
        except ValueError as e:
            return error(str(e), 422)
        except PermissionError as e:
            return error(str(e), 403)
        except Exception as e:
            logger.error(f"[DOCS API] update watched folder error: {e}")
            return error(_ERR_INTERNAL, 500)

    @require_session
    @documents_ns.param("delete_documents", "Also soft-delete the folder's documents", type="boolean")
    @documents_ns.response(204, "Watched folder deleted")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def delete(self, folder_id: str) -> None | ResponseReturnValue:
        """Remove a watched folder, optionally soft-deleting its documents."""
        delete_documents = request.args.get("delete_documents", "false").lower() == "true"
        try:
            if not _get_watcher_service().delete_folder(folder_id, delete_documents=delete_documents):
                return error(_ERR_NOT_FOUND, 404)
            return None
        except Exception as e:
            logger.error(f"[DOCS API] delete watched folder error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/watched-folders/<folder_id>/scan")
@documents_ns.param("folder_id", "Folder identifier")
class WatchedFolderScanResource(Resource):
    @require_session
    @documents_ns.response(204, "Scan requested")
    @documents_ns.response(404, _ERR_NOT_FOUND, model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(code=204)
    def post(self, folder_id: str) -> None | ResponseReturnValue:
        """Trigger an out-of-schedule immediate scan for a watched folder."""
        try:
            svc = _get_watcher_service()
            if not svc.get_folder(folder_id):
                return error(_ERR_NOT_FOUND, 404)
            svc.trigger_scan(folder_id)
            return None
        except Exception as e:
            logger.error(f"[DOCS API] trigger scan error: {e}")
            return error(_ERR_INTERNAL, 500)


@documents_ns.route("/watched-folders/browse")
class WatchedFolderBrowseResource(Resource):
    @require_session
    @documents_ns.expect(_D["BrowseRequest"])
    @documents_ns.response(200, "Directory listing", model=_D["BrowseResponse"])
    @documents_ns.response(403, "Permission denied", model=_D["Error"])
    @documents_ns.response(422, "Invalid path", model=_D["Error"])
    @documents_ns.response(500, _ERR_INTERNAL, model=_D["Error"])
    @responds(BrowseResponse, code=200)
    @expects(BrowseRequest)
    def post(self, dto: BrowseRequest) -> BrowseResponse | ResponseReturnValue:
        """Browse readable sub-directories at a host filesystem path."""
        try:
            result = _get_watcher_service().browse_directory(dto.path)
            return BrowseResponse(
                current=cast(str, result["current"]),
                parent=cast(_OptStr, result.get("parent")),
                directories=cast(_StrList, result["directories"]),
            )
        except ValueError as e:
            return error(str(e), 422)
        except PermissionError as e:
            return error(str(e), 403)
        except Exception as e:
            logger.error(f"[DOCS API] browse error: {e}")
            return error(_ERR_INTERNAL, 500)


# ---------------------------------------------------------------------------
# DTO builders
# ---------------------------------------------------------------------------

def _document_dto(row: "dict[str, object]") -> Document:
    """Build a :class:`Document` DTO from a service row dict."""
    return Document(
        id=cast(str, row["id"]),
        original_name=cast(str, row["original_name"]),
        mime_type=cast(str, row["mime_type"]),
        file_size_bytes=cast(int, row["file_size_bytes"]),
        file_hash=cast(str, row["file_hash"]),
        page_count=cast("int | None", row.get("page_count")),
        status=cast(str, row["status"]),
        error_message=cast(_OptStr, row.get("error_message")),
        chunk_count=cast("int | None", row.get("chunk_count")),
        source_type=cast(str, row["source_type"]),
        tags=cast(_StrList, row.get("tags") or []),
        summary=cast(_OptStr, row.get("summary")),
        extracted_metadata=cast("dict[str, object]", row.get("extracted_metadata") or {}),
        supersedes_id=cast(_OptStr, row.get("supersedes_id")),
        language=cast(_OptStr, row.get("language")),
        fingerprint=cast(_OptStr, row.get("fingerprint")),
        doc_category=cast(_OptStr, row.get("doc_category")),
        doc_project=cast(_OptStr, row.get("doc_project")),
        doc_date=cast(_OptStr, row.get("doc_date")),
        meta_locked=bool(row.get("meta_locked")),
        watched_folder_id=cast(_OptStr, row.get("watched_folder_id")),
        created_at=cast("datetime", row["created_at"]),
        updated_at=cast("datetime", row["updated_at"]),
        deleted_at=cast(_OptDatetime, row.get("deleted_at")),
        purge_after=cast(_OptDatetime, row.get("purge_after")),
    )


def _watched_folder_dto(row: "dict[str, object]") -> WatchedFolder:
    """Build a :class:`WatchedFolder` DTO from a service row dict."""
    return WatchedFolder(
        id=cast(str, row["id"]),
        folder_path=cast(str, row["folder_path"]),
        label=cast(_OptStr, row.get("label")),
        source_type=cast(str, row["source_type"]),
        enabled=bool(cast(int, row.get("enabled"))),
        file_patterns=cast(_StrList, row.get("file_patterns") or []),
        ignore_patterns=cast(_StrList, row.get("ignore_patterns") or []),
        recursive=bool(cast(int, row.get("recursive"))),
        scan_interval=cast(int, row["scan_interval"]),
        source_config=cast("dict[str, object]", row.get("source_config") or {}),
        created_at=cast("datetime", row["created_at"]),
        updated_at=cast("datetime", row["updated_at"]),
    )


def _resolve_document_file(doc: "dict[str, object]") -> str | None:
    """Resolve the on-disk path for a document, validating uploads stay in-root."""
    if doc.get("watched_folder_id"):
        full_path = cast(str, doc["file_path"])
        return full_path if os.path.isfile(os.path.realpath(full_path)) else None
    full_path = str(FileMapperService.get_documents_path(cast(str, doc["file_path"])))
    return full_path if _validate_file_path(full_path) else None
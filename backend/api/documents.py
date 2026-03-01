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
"""

import hashlib
import logging
import os
import re
import mimetypes
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

from .auth import require_session

logger = logging.getLogger(__name__)

documents_bp = Blueprint("documents", __name__)

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

# Document storage root
DOCUMENTS_ROOT = os.environ.get('DOCUMENTS_ROOT', '/app/data/documents')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_document_service():
    from services.database_service import get_shared_db_service
    from services.document_service import DocumentService
    return DocumentService(get_shared_db_service())


def _serialize_dt(val):
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize_doc(doc: dict) -> dict:
    """Serialize document dict for JSON response."""
    out = dict(doc)
    for field in ('created_at', 'updated_at', 'deleted_at', 'purge_after'):
        if field in out:
            out[field] = _serialize_dt(out[field])
    # Don't send clean_text in list responses (too large)
    out.pop('clean_text', None)
    return out


def _sanitize_filename(name: str) -> str:
    """Sanitize filename: strip path separators, null bytes, control chars."""
    # Remove path separators and null bytes
    name = name.replace('/', '').replace('\\', '').replace('\x00', '')
    # Remove control characters
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)
    # Prevent directory traversal
    name = name.lstrip('.')
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    if not name:
        name = 'unnamed_document'
    # Limit length
    if len(name) > 255:
        ext = os.path.splitext(name)[1]
        name = name[:255 - len(ext)] + ext
    return name


def _validate_file_path(full_path: str) -> bool:
    """Ensure resolved path is within DOCUMENTS_ROOT (prevent symlink attacks)."""
    real_path = os.path.realpath(full_path)
    real_root = os.path.realpath(DOCUMENTS_ROOT)
    return real_path.startswith(real_root)


def _enqueue_processing(doc_id: str):
    """Enqueue document for background processing."""
    try:
        from services import PromptQueue
        from workers.document_worker import process_document_job
        queue = PromptQueue(queue_name="document-queue", worker_func=process_document_job)
        queue.enqueue({'doc_id': doc_id})
        logger.info(f"[DOCS API] Enqueued processing for {doc_id}")
    except Exception as e:
        logger.error(f"[DOCS API] Failed to enqueue processing: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@documents_bp.route("/documents/upload", methods=["POST"])
@require_session
def upload_document():
    """Multipart file upload → save to disk, create DB row, enqueue processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No filename provided"}), 400

    # Sanitize filename
    original_name = _sanitize_filename(file.filename)

    # Check extension
    ext = os.path.splitext(original_name)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type '{ext}' is not supported"}), 400

    # Read file content for size check and hash
    content = file.read()
    if len(content) > MAX_FILE_SIZE:
        return jsonify({"error": f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit"}), 400

    if len(content) == 0:
        return jsonify({"error": "File is empty"}), 400

    # MIME type validation
    content_type = file.content_type or mimetypes.guess_type(original_name)[0] or 'application/octet-stream'

    # Compute file hash
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        svc = _get_document_service()

        # Create document record
        import secrets
        doc_id = secrets.token_hex(4)
        file_path = f"{doc_id}/{original_name}"

        # Save file to disk
        dir_path = os.path.join(DOCUMENTS_ROOT, doc_id)
        os.makedirs(dir_path, exist_ok=True)

        full_path = os.path.join(dir_path, original_name)
        if not _validate_file_path(full_path):
            return jsonify({"error": "Invalid file path"}), 400

        with open(full_path, 'wb') as f:
            f.write(content)

        # Create DB record
        doc_id = svc.create_document(
            original_name=original_name,
            mime_type=content_type,
            file_size=len(content),
            file_path=file_path,
            file_hash=file_hash,
            source_type='upload',
        )

        # Check for exact hash duplicates before processing
        duplicates = svc.find_duplicates(file_hash, None, 0, exclude_id=doc_id)

        # Enqueue for processing
        _enqueue_processing(doc_id)

        response = {
            "id": doc_id,
            "original_name": original_name,
            "status": "pending",
            "file_size": len(content),
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

        return jsonify(response), 201

    except Exception as e:
        logger.error(f"[DOCS API] upload error: {e}", exc_info=True)
        return jsonify({"error": "Upload failed"}), 500


@documents_bp.route("/documents", methods=["GET"])
@require_session
def list_documents():
    """List all documents."""
    include_deleted = request.args.get('include_deleted', 'false').lower() == 'true'
    try:
        svc = _get_document_service()
        docs = svc.get_all_documents(include_deleted=include_deleted)
        return jsonify({"items": [_serialize_doc(d) for d in docs]})
    except Exception as e:
        logger.error(f"[DOCS API] list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>", methods=["GET"])
@require_session
def get_document(doc_id):
    """Get document metadata + first N chunks."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Not found"}), 404

        # Include first 5 chunks
        chunks = svc.get_chunks_for_document(doc_id)[:5]

        result = _serialize_doc(doc)
        result['chunks'] = chunks
        return jsonify({"item": result})
    except Exception as e:
        logger.error(f"[DOCS API] get_document error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/content", methods=["GET"])
@require_session
def get_document_content(doc_id):
    """Get full extracted text, paginated by chunks."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Not found"}), 404

        chunks = svc.get_chunks_for_document(doc_id)
        start = (page - 1) * per_page
        end = start + per_page

        return jsonify({
            "document_id": doc_id,
            "total_chunks": len(chunks),
            "page": page,
            "per_page": per_page,
            "chunks": chunks[start:end],
        })
    except Exception as e:
        logger.error(f"[DOCS API] get_content error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/download", methods=["GET"])
@require_session
def download_document(doc_id):
    """Download original file."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Not found"}), 404

        full_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])
        if not _validate_file_path(full_path) or not os.path.exists(full_path):
            return jsonify({"error": "File not found on disk"}), 404

        return send_file(
            full_path,
            mimetype=doc['mime_type'],
            as_attachment=True,
            download_name=doc['original_name'],
        )
    except Exception as e:
        logger.error(f"[DOCS API] download error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>", methods=["DELETE"])
@require_session
def delete_document(doc_id):
    """Soft-delete a document."""
    try:
        svc = _get_document_service()
        ok = svc.soft_delete(doc_id)
        if not ok:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] delete error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/restore", methods=["POST"])
@require_session
def restore_document(doc_id):
    """Undo soft delete."""
    try:
        svc = _get_document_service()
        ok = svc.restore(doc_id)
        if not ok:
            return jsonify({"error": "Not found or not deleted"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] restore error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/purge", methods=["DELETE"])
@require_session
def purge_document(doc_id):
    """Immediate hard delete."""
    try:
        svc = _get_document_service()
        ok = svc.hard_delete(doc_id)
        if not ok:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] purge error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/search", methods=["GET"])
@require_session
def search_documents():
    """Semantic search across document chunks."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    limit = min(int(request.args.get('limit', 5)), 20)

    try:
        from services.embedding_service import get_embedding_service

        embedding_service = get_embedding_service()
        query_embedding = embedding_service.generate_embedding(query)

        svc = _get_document_service()
        results = svc.search_chunks(query_embedding, query, limit=limit)

        # Serialize results
        serialized = []
        for r in results:
            item = dict(r)
            if 'document_created_at' in item:
                item['document_created_at'] = _serialize_dt(item['document_created_at'])
            serialized.append(item)

        return jsonify({"results": serialized, "query": query})
    except Exception as e:
        logger.error(f"[DOCS API] search error: {e}")
        return jsonify({"error": "Search failed"}), 500


@documents_bp.route("/documents/<doc_id>/confirm", methods=["POST"])
@require_session
def confirm_document(doc_id):
    """Confirm document after synthesis review — marks it as ready."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Not found"}), 404

        if doc['status'] != 'awaiting_confirmation':
            return jsonify({"error": "Document is not awaiting confirmation"}), 400

        svc.update_status(doc_id, 'ready', chunk_count=doc.get('chunk_count', 0))
        return jsonify({"ok": True, "status": "ready"})
    except Exception as e:
        logger.error(f"[DOCS API] confirm error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/augment", methods=["POST"])
@require_session
def augment_document(doc_id):
    """Add user context to a document and confirm it."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Not found"}), 404

        if doc['status'] != 'awaiting_confirmation':
            return jsonify({"error": "Document is not awaiting confirmation"}), 400

        data = request.get_json(silent=True) or {}
        context = (data.get('context') or '').strip()
        if not context:
            return jsonify({"error": "Field 'context' is required"}), 400

        # Store user context in extracted_metadata
        import json
        metadata = doc.get('extracted_metadata') or {}
        metadata['_user_context'] = context

        svc.update_extracted_metadata(
            doc_id,
            metadata=metadata,
            summary=doc.get('summary', ''),
            summary_embedding=doc.get('summary_embedding'),
        )

        svc.update_status(doc_id, 'ready', chunk_count=doc.get('chunk_count', 0))
        return jsonify({"ok": True, "status": "ready"})
    except Exception as e:
        logger.error(f"[DOCS API] augment error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/documents/<doc_id>/supersede", methods=["POST"])
@require_session
def supersede_document(doc_id):
    """Mark a new document as replacing an older one, and soft-delete the old."""
    try:
        svc = _get_document_service()

        new_doc = svc.get_document(doc_id)
        if not new_doc:
            return jsonify({"error": "New document not found"}), 404

        data = request.get_json(silent=True) or {}
        old_id = (data.get('old_id') or '').strip()
        if not old_id:
            return jsonify({"error": "Field 'old_id' is required"}), 400

        old_doc = svc.get_document(old_id)
        if not old_doc:
            return jsonify({"error": "Old document not found"}), 404

        svc.set_supersedes(doc_id, old_id)
        svc.soft_delete(old_id)

        return jsonify({"ok": True, "supersedes_id": old_id})
    except Exception as e:
        logger.error(f"[DOCS API] supersede error: {e}")
        return jsonify({"error": "Internal server error"}), 500



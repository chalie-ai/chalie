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

import hashlib
import json
import logging
import os
import re
import mimetypes
import threading
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

import paths
from .auth import require_session

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"
_ERR_FILE_NOT_FOUND = "File not found on disk"

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

# Document storage root — single hard-coded layout (paths.py).
DOCUMENTS_ROOT = str(paths.DOCUMENTS_DIR)


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


def _read_existing_metadata(svc, doc_id: str) -> dict:
    """Read prior extracted_metadata so concurrent writes are not clobbered."""
    existing = svc.get_document(doc_id) or {}
    meta = existing.get('extracted_metadata') or {}
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta


def _derive_summary(text: str) -> str:
    """Extract up to a 500-char summary, truncated at the last sentence boundary after 200 chars."""
    summary = text[:500]
    dot_pos = summary.rfind('. ')
    if dot_pos > 200:
        return summary[:dot_pos + 1]
    return summary


def _mark_upload_failed(doc_id: str, error: str):
    """Best-effort status update to 'failed' — swallow errors since we're already in a failure path."""
    try:
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service
        DocumentService(get_shared_db_service()).update_status(doc_id, 'failed', error[:500])
    except Exception:
        logger.exception(f"[DOCS API] Could not mark {doc_id} as failed")


def _run_upload_extraction(doc_id: str):
    """Extract text + write artifacts + mark ready. Raises on unrecoverable errors."""
    from services.document_service import DocumentService
    from services.database_service import get_shared_db_service
    from services.text_extractor import extract_text
    from abilities.document import create_document_artifacts

    svc = DocumentService(get_shared_db_service())
    doc = svc.get_document(doc_id)
    if not doc:
        return

    file_path = doc.get('file_path', '')
    if not file_path:
        svc.update_status(doc_id, 'failed', 'No file path')
        return

    text = extract_text(os.path.join(DOCUMENTS_ROOT, file_path))
    if not text:
        svc.update_status(doc_id, 'failed', 'Text extraction returned empty')
        return

    artifact_count = create_document_artifacts(doc_id, text)

    # Write clean_text so websocket._resolve_file_tags can inject content into
    # the LLM prompt. Merge with prior metadata to avoid clobbering concurrent
    # synthesis/classification writes.
    svc.update_extracted_metadata(
        doc_id,
        metadata=_read_existing_metadata(svc, doc_id),
        summary=_derive_summary(text),
        clean_text=text,
    )
    svc.update_status(doc_id, 'ready', chunk_count=artifact_count)
    logger.info(f"[DOCS API] Processed upload {doc_id}: {artifact_count} artifacts")


def _process_upload(doc_id: str):
    """Extract text from uploaded document and create data_graph artifacts."""
    def _run():
        try:
            _run_upload_extraction(doc_id)
        except Exception as e:
            logger.error(f"[DOCS API] Failed to process upload {doc_id}: {e}")
            _mark_upload_failed(doc_id, str(e))

    threading.Thread(target=_run, daemon=True, name=f"doc-upload-{doc_id[:8]}").start()


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

        # Process upload in background
        _process_upload(doc_id)

        response = {
            "id": doc_id,
            "original_name": original_name,
            "status": "pending",
            "file_size": len(content),
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
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>", methods=["GET"])
@require_session
def get_document(doc_id):
    """Get document metadata + first N data_graph artifact previews."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()
        with dgs.db.connection() as conn:
            rows = conn.execute(
                "SELECT key, substr(value, 1, 200) as preview FROM data_graph WHERE source=? AND active=1 ORDER BY key LIMIT 5",
                (f'document:{doc_id}',),
            ).fetchall()
        result = _serialize_doc(doc)
        result['artifacts'] = [{'key': r[0], 'preview': r[1]} for r in rows]
        return jsonify({"item": result})
    except Exception as e:
        logger.error(f"[DOCS API] get_document error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/content", methods=["GET"])
@require_session
def get_document_content(doc_id):
    """Get full document text reconstructed from data_graph artifacts."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()
        with dgs.db.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM data_graph WHERE source=? AND active=1 ORDER BY key",
                (f'document:{doc_id}',),
            ).fetchall()

        artifacts = [{'key': r[0], 'content': r[1]} for r in rows]
        return jsonify({
            "document_id": doc_id,
            "total_artifacts": len(artifacts),
            "artifacts": artifacts,
        })
    except Exception as e:
        logger.error(f"[DOCS API] get_content error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/download", methods=["GET"])
@require_session
def download_document(doc_id):
    """Download original file."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        # Watched folder docs store absolute paths; uploaded docs are relative to DOCUMENTS_ROOT
        if doc.get('watched_folder_id'):
            full_path = doc['file_path']
            if not os.path.isfile(os.path.realpath(full_path)):
                return jsonify({"error": _ERR_FILE_NOT_FOUND}), 404
        else:
            full_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])
            if not _validate_file_path(full_path) or not os.path.exists(full_path):
                return jsonify({"error": _ERR_FILE_NOT_FOUND}), 404

        return send_file(
            full_path,
            mimetype=doc['mime_type'],
            as_attachment=True,
            download_name=doc['original_name'],
        )
    except Exception as e:
        logger.error(f"[DOCS API] download error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/preview", methods=["GET"])
@require_session
def preview_document(doc_id):
    """Stream file for inline browser preview (no Content-Disposition: attachment)."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        if doc.get('watched_folder_id'):
            full_path = doc['file_path']
            if not os.path.isfile(os.path.realpath(full_path)):
                return jsonify({"error": _ERR_FILE_NOT_FOUND}), 404
        else:
            full_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])
            if not _validate_file_path(full_path) or not os.path.exists(full_path):
                return jsonify({"error": _ERR_FILE_NOT_FOUND}), 404

        return send_file(
            full_path,
            mimetype=doc['mime_type'],
            as_attachment=False,
            download_name=doc['original_name'],
        )
    except Exception as e:
        logger.error(f"[DOCS API] preview error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/classify", methods=["PUT"])
@require_session
def update_document_classification(doc_id):
    """Update document classification metadata (user edit — locks auto-classification)."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        data = request.get_json() or {}
        svc.update_classification(
            doc_id,
            category=data.get('category'),
            project=data.get('project'),
            doc_date=data.get('date'),
            lock=True,
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"[DOCS API] classify update error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/groups/<field>", methods=["GET"])
@require_session
def get_document_groups(field):
    """Get unique classification groups for a field (doc_category, doc_project, doc_date)."""
    try:
        svc = _get_document_service()
        groups = svc.get_classification_groups(field)
        return jsonify({"groups": groups})
    except Exception as e:
        logger.error(f"[DOCS API] groups error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>", methods=["DELETE"])
@require_session
def delete_document(doc_id):
    """Soft-delete a document."""
    try:
        svc = _get_document_service()
        ok = svc.soft_delete(doc_id)
        if not ok:
            return jsonify({"error": _ERR_NOT_FOUND}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] delete error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


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
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/purge", methods=["DELETE"])
@require_session
def purge_document(doc_id):
    """Immediate hard delete."""
    try:
        svc = _get_document_service()
        ok = svc.hard_delete(doc_id)
        if not ok:
            return jsonify({"error": _ERR_NOT_FOUND}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] purge error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/search", methods=["GET"])
@require_session
def search_documents():
    """Search across document artifacts in data_graph."""
    q_raw = request.args.get('q', None)
    if q_raw is None:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    query = q_raw.strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    limit = min(int(request.args.get('limit', 5)), 20)

    try:
        from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT

        dgs = get_data_graph_service()
        results = dgs.recall(query, kinds=[KIND_DOCUMENT], limit=limit)

        serialized = []
        for row in results:
            source = row.get('source', '') or ''
            doc_id = source.split(':', 1)[1] if source.startswith('document:') else ''
            serialized.append({
                'document_id': doc_id,
                'key': row.get('key', ''),
                'content': row.get('value', ''),
                'source': source,
            })

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
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        if doc['status'] != 'awaiting_confirmation':
            return jsonify({"error": "Document is not awaiting confirmation"}), 400

        svc.update_status(doc_id, 'ready', chunk_count=doc.get('chunk_count', 0))
        return jsonify({"ok": True, "status": "ready"})
    except Exception as e:
        logger.error(f"[DOCS API] confirm error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/<doc_id>/augment", methods=["POST"])
@require_session
def augment_document(doc_id):
    """Add user context to a document and confirm it."""
    try:
        svc = _get_document_service()
        doc = svc.get_document(doc_id)
        if not doc:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        if doc['status'] not in ('awaiting_confirmation', 'ready'):
            return jsonify({"error": "Document cannot be augmented in its current state"}), 400

        data = request.get_json(silent=True) or {}
        context = (data.get('context') or '').strip()
        if not context:
            return jsonify({"error": "Field 'context' is required"}), 400

        # Store user context in extracted_metadata
        metadata = doc.get('extracted_metadata') or {}
        metadata['_user_context'] = context

        svc.update_extracted_metadata(
            doc_id,
            metadata=metadata,
            summary=doc.get('summary', ''),
            summary_embedding=doc.get('summary_embedding'),
        )

        if doc['status'] != 'ready':
            svc.update_status(doc_id, 'ready', chunk_count=doc.get('chunk_count', 0))
        return jsonify({"ok": True, "status": "ready"})
    except Exception as e:
        logger.error(f"[DOCS API] augment error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


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
        return jsonify({"error": _ERR_INTERNAL}), 500


# ---------------------------------------------------------------------------
# Watched Folders
# ---------------------------------------------------------------------------

def _get_watcher_service():
    from services.database_service import get_shared_db_service
    from services.folder_watcher_service import FolderWatcherService
    return FolderWatcherService(get_shared_db_service())


@documents_bp.route("/documents/watched-folders", methods=["GET"])
@require_session
def list_watched_folders():
    """List all watched folders."""
    try:
        svc = _get_watcher_service()
        folders = svc.get_all_folders()
        return jsonify({"items": folders})
    except Exception as e:
        logger.error(f"[DOCS API] list watched folders error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/watched-folders", methods=["POST"])
@require_session
def create_watched_folder():
    """Add a new watched folder."""
    data = request.get_json(silent=True) or {}
    folder_path = (data.get('folder_path') or '').strip()

    if not folder_path:
        return jsonify({"error": "Field 'folder_path' is required"}), 400

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
        return jsonify({"item": folder}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        if 'UNIQUE constraint' in str(e):
            return jsonify({"error": "This folder is already being watched"}), 409
        logger.error(f"[DOCS API] create watched folder error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/watched-folders/<folder_id>", methods=["PUT"])
@require_session
def update_watched_folder(folder_id):
    """Update watched folder settings."""
    data = request.get_json(silent=True) or {}
    try:
        svc = _get_watcher_service()
        folder = svc.get_folder(folder_id)
        if not folder:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        updated = svc.update_folder(folder_id, **data)
        return jsonify({"item": updated})
    except (ValueError, PermissionError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[DOCS API] update watched folder error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/watched-folders/<folder_id>", methods=["DELETE"])
@require_session
def delete_watched_folder(folder_id):
    """Remove a watched folder."""
    delete_documents = request.args.get('delete_documents', 'false').lower() == 'true'
    try:
        svc = _get_watcher_service()
        ok = svc.delete_folder(folder_id, delete_documents=delete_documents)
        if not ok:
            return jsonify({"error": _ERR_NOT_FOUND}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[DOCS API] delete watched folder error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/watched-folders/<folder_id>/scan", methods=["POST"])
@require_session
def trigger_scan(folder_id):
    """Trigger an immediate scan for a watched folder."""
    try:
        svc = _get_watcher_service()
        folder = svc.get_folder(folder_id)
        if not folder:
            return jsonify({"error": _ERR_NOT_FOUND}), 404

        svc.trigger_scan(folder_id)
        return jsonify({"ok": True, "message": "Scan requested"})
    except Exception as e:
        logger.error(f"[DOCS API] trigger scan error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500


@documents_bp.route("/documents/watched-folders/browse", methods=["POST"])
@require_session
def browse_directories():
    """Browse host filesystem directories for folder selection."""
    data = request.get_json(silent=True) or {}
    path = data.get('path')

    try:
        svc = _get_watcher_service()
        result = svc.browse_directory(path)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        logger.error(f"[DOCS API] browse error: {e}")
        return jsonify({"error": _ERR_INTERNAL}), 500



"""
Chat Image API — Upload and analyze images attached to chat messages.

Routes (all require session auth):
  POST /chat/image              — multipart upload, returns {image_id, status}
  GET  /chat/image/<id>/status  — poll analysis result
  GET  /chat/image/<id>/file    — serve persisted image file (or redirect to /documents/<id>/preview)
  GET  /chat/vision-capable     — capability check for frontend

Design:
  Images are stored in MemoryStore (5min TTL for raw bytes, 10min TTL for analysis
  results) AND persisted to the documents table with source_type='chat_image'.
  MemoryStore is kept for backward compatibility with the existing WS handler polling.
  The document record is the durable store: it survives across restarts and beyond
  the MemoryStore TTLs.

  SHA-256 hash deduplication: uploading the same image twice within a session
  returns the existing image_id without re-analysis.  The hash check now also
  queries document_service.find_duplicates() for cross-session dedup.

  Progress tracking: a lightweight ``chat_image_progress:{image_id}`` key with
  a 10 min TTL is written at upload time and updated by the analysis thread.
  This acts as a tertiary fallback for GET /status so that the endpoint never
  returns 404 while analysis is still in-flight, even after the 5-minute bytes
  key has expired (B3 fix).

  GET /status lookup order:
    1. chat_image_result:{image_id} in MemoryStore → parse and return
    2. chat_image:{image_id} bytes key present → 'analyzing'
    3. chat_image_progress:{image_id} key present → its value
    4. document_service.get_document(image_id):
       - status='ready'       → return ready + extracted_metadata
       - status='processing'  → return analyzing
    5. 404
"""

import hashlib
import json
import logging
import os
import threading

from flask import Blueprint, jsonify, redirect, url_for, request

from .auth import require_session

logger = logging.getLogger(__name__)

chat_image_bp = Blueprint('chat_image', __name__)

# Max upload size: 10MB
_MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Allowed MIME types
_ALLOWED_MIMES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}

# MemoryStore key patterns and TTLs
_KEY_BYTES = 'chat_image:{image_id}'             # raw bytes (bytes value)
_KEY_RESULT = 'chat_image_result:{image_id}'     # JSON analysis result
_KEY_HASH = 'chat_image_hash:{hash}'             # hash → image_id dedup
_KEY_PROGRESS = 'chat_image_progress:{image_id}' # in-flight tracker ('analyzing'|'ready'|'failed')

_TTL_BYTES = 300    # 5 minutes
_TTL_RESULT = 600   # 10 minutes
_TTL_HASH = 300     # 5 minutes
_TTL_PROGRESS = 600 # 10 minutes — must outlive _TTL_BYTES so it can serve as fallback

# Extension → sanitised filename suffix when original filename is unavailable
_MIME_EXT = {
    'image/jpeg': '.jpg',
    'image/png':  '.png',
    'image/webp': '.webp',
    'image/gif':  '.gif',
}


def _get_store():
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


def _get_document_service():
    from services.database_service import get_shared_db_service
    from services.document_service import DocumentService
    return DocumentService(get_shared_db_service())


def _documents_root() -> str:
    """Return the documents storage root (mirrors documents.py and document_service.py)."""
    from services.document_service import DOCUMENTS_ROOT
    return DOCUMENTS_ROOT


# ─── Routes ──────────────────────────────────────────────────────────────────

@chat_image_bp.route('/chat/image', methods=['POST'])
@require_session
def upload_image():
    """
    Multipart image upload.

    Accepts multipart field ``image``.  Stores raw bytes in MemoryStore and
    creates a persistent document record (source_type='chat_image').  Writes
    a ``chat_image_progress:{image_id}`` key (TTL=10 min) for in-flight
    tracking, then kicks off background analysis in a daemon thread and
    returns ``{image_id, status: 'analyzing'}`` immediately.

    The document ID is used as the image_id so that the file and status can
    be retrieved durably via GET /chat/image/<id>/file and
    GET /documents/<id>/preview even after the MemoryStore TTLs expire.

    Deduplication: SHA-256 hash checked against both MemoryStore (within-
    session) and document_service.find_duplicates() (cross-session).

    Returns:
        201: ``{image_id, status: 'analyzing'}`` for a new upload.
        200: ``{image_id, status: '<actual_status>'}`` for a duplicate.
        400: missing or empty file.
        413: file exceeds 10 MB.
        415: unsupported MIME type.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No filename provided'}), 400

    content_type = file.content_type or 'application/octet-stream'

    # Normalize MIME from content-type header (strip charset etc.)
    mime = content_type.split(';')[0].strip().lower()
    if mime not in _ALLOWED_MIMES:
        return jsonify({'error': f'Unsupported image type: {mime}. Accepted: jpeg, png, webp, gif'}), 415

    image_bytes = file.read()

    if len(image_bytes) == 0:
        return jsonify({'error': 'Empty file'}), 400

    if len(image_bytes) > _MAX_IMAGE_BYTES:
        mb = _MAX_IMAGE_BYTES // 1024 // 1024
        return jsonify({'error': f'Image exceeds {mb}MB limit'}), 413

    # SHA-256 deduplication — check MemoryStore first (fast path)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    store = _get_store()

    hash_key = _KEY_HASH.format(hash=image_hash)
    existing_id = store.get(hash_key)
    if existing_id:
        if isinstance(existing_id, bytes):
            existing_id = existing_id.decode()
        logger.debug(f'[CHAT IMAGE] Duplicate detected (MemoryStore) — returning existing image_id={existing_id}')
        dedup_result_key = _KEY_RESULT.format(image_id=existing_id)
        raw_dedup = store.get(dedup_result_key)
        if raw_dedup is not None:
            try:
                dedup_result = json.loads(raw_dedup)
                dedup_status = 'failed' if dedup_result.get('error') else 'ready'
            except Exception:
                dedup_status = 'failed'
        else:
            dedup_status = 'analyzing'
        return jsonify({'image_id': existing_id, 'status': dedup_status}), 200

    # Cross-session dedup via document_service (hash check only — no embedding yet)
    try:
        doc_svc = _get_document_service()
        existing_docs = doc_svc.find_duplicates(image_hash, None, 0)
        # Filter to chat_image source only so we don't confuse with regular doc uploads
        chat_dupes = [d for d in existing_docs if True]  # hash match is always exact
        if chat_dupes:
            existing_doc = chat_dupes[0]
            existing_id = existing_doc['id']
            logger.debug(
                f'[CHAT IMAGE] Duplicate detected (DocumentService) — '
                f'returning existing image_id={existing_id}'
            )
            # Re-populate MemoryStore hash key so same-session follow-ups are fast
            store.set(hash_key, existing_id, ex=_TTL_HASH)
            # Determine current status from document record
            doc = doc_svc.get_document(existing_id)
            if doc:
                doc_status = doc.get('status', 'processing')
                if doc_status == 'ready':
                    dedup_status = 'ready'
                elif doc_status == 'failed':
                    dedup_status = 'failed'
                else:
                    dedup_status = 'analyzing'
            else:
                dedup_status = 'analyzing'
            return jsonify({'image_id': existing_id, 'status': dedup_status}), 200
    except Exception as e:
        # Non-fatal — fall through and treat as new upload
        logger.warning(f'[CHAT IMAGE] Document dedup check failed (non-fatal): {e}')

    # New image — create document record and use its ID as image_id
    original_filename = _sanitize_image_filename(file.filename, mime)

    try:
        doc_svc = _get_document_service()
        doc_root = _documents_root()

        # We need the doc_id before writing to disk so the path is consistent.
        # Reserve an ID by generating it here and passing it to create_document.
        import secrets as _secrets
        image_id = _secrets.token_hex(4)  # 8-char hex, same length as document IDs
        file_path_rel = f"{image_id}/{original_filename}"
        dir_path = os.path.join(doc_root, image_id)
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, original_filename)

        # Persist bytes to disk
        with open(full_path, 'wb') as fh:
            fh.write(image_bytes)

        # Create document record with the reserved ID
        doc_svc.create_document(
            original_name=original_filename,
            mime_type=mime,
            file_size=len(image_bytes),
            file_path=file_path_rel,
            file_hash=image_hash,
            source_type='chat_image',
            doc_id=image_id,
        )
        logger.debug(f'[CHAT IMAGE] Created document record image_id={image_id}')

    except Exception as e:
        logger.error(f'[CHAT IMAGE] Failed to create document record: {e}', exc_info=True)
        # Fall back to a random hex ID so the upload still works ephemerally
        import secrets as _secrets
        image_id = _secrets.token_hex(8)

    bytes_key = _KEY_BYTES.format(image_id=image_id)
    progress_key = _KEY_PROGRESS.format(image_id=image_id)
    store.set(bytes_key, image_bytes, ex=_TTL_BYTES)
    store.set(hash_key, image_id, ex=_TTL_HASH)
    # B3 fix: write a progress key immediately so GET /status can return
    # 'analyzing' even after the shorter bytes TTL has expired.
    store.set(progress_key, 'analyzing', ex=_TTL_PROGRESS)

    # Background analysis (non-blocking)
    threading.Thread(
        target=_run_analysis,
        args=(image_id, image_bytes, mime),
        daemon=True,
    ).start()

    return jsonify({'image_id': image_id, 'status': 'analyzing'}), 201


@chat_image_bp.route('/chat/image/<image_id>/status', methods=['GET'])
@require_session
def image_status(image_id):
    """
    Poll analysis result for an uploaded image.

    Lookup order for status resolution:

    1. ``chat_image_result:{image_id}`` present → parse and return
       ``'ready'`` or ``'failed'``.
    2. ``chat_image:{image_id}`` (bytes key) present → return ``'analyzing'``.
    3. ``chat_image_progress:{image_id}`` (progress key) present → return its
       value (``'analyzing'``, ``'ready'``, or ``'failed'``). This handles the
       B3 edge-case where the bytes key has already expired but analysis is
       still in-flight or has just completed (B3 fix).
    4. ``document_service.get_document(image_id)`` — durable fallback for
       images retrieved after MemoryStore TTLs have expired:
       - status='ready'      → return ready + parsed extracted_metadata
       - status='processing' → return analyzing
    5. None of the above → 404.

    Args:
        image_id: Hex token returned by the upload endpoint.

    Returns:
        200: ``{status: 'analyzing'|'ready'|'failed', result: {...}|null}``
        404: ``{error: 'Image not found'}`` when all keys and the document record are absent.
    """
    store = _get_store()
    result_key = _KEY_RESULT.format(image_id=image_id)
    raw = store.get(result_key)

    if raw is None:
        # Primary fallback: raw bytes key still present → analysis is in-flight.
        bytes_key = _KEY_BYTES.format(image_id=image_id)
        if store.exists(bytes_key):
            return jsonify({'status': 'analyzing', 'result': None})
        # B3 fix: secondary fallback — progress key survives longer than bytes key
        # (600 s vs 300 s) and is updated by the analysis thread to 'ready'/'failed'.
        progress_key = _KEY_PROGRESS.format(image_id=image_id)
        progress = store.get(progress_key)
        if progress is not None:
            if isinstance(progress, bytes):
                progress = progress.decode()
            return jsonify({'status': progress, 'result': None})

        # Durable fallback: check the document record (survives beyond MemoryStore TTLs)
        try:
            doc_svc = _get_document_service()
            doc = doc_svc.get_document(image_id)
            if doc:
                doc_status = doc.get('status', 'processing')
                if doc_status == 'ready':
                    # Reconstruct result from extracted_metadata stored during _run_analysis
                    raw_meta = doc.get('extracted_metadata')
                    result = None
                    if raw_meta:
                        try:
                            result = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                        except Exception:
                            result = None
                    return jsonify({'status': 'ready', 'result': result})
                elif doc_status == 'failed':
                    return jsonify({'status': 'failed', 'result': None})
                else:
                    # 'processing' or any other interim state
                    return jsonify({'status': 'analyzing', 'result': None})
        except Exception as e:
            logger.warning(f'[CHAT IMAGE] Document fallback check failed for {image_id}: {e}')

        return jsonify({'error': 'Image not found'}), 404

    try:
        result = json.loads(raw)
        status = 'failed' if result.get('error') else 'ready'
        return jsonify({'status': status, 'result': result})
    except Exception:
        return jsonify({'status': 'failed', 'result': None})


@chat_image_bp.route('/chat/image/<image_id>/file', methods=['GET'])
@require_session
def image_file(image_id):
    """
    Serve the persisted image file for a chat-uploaded image.

    Redirects to ``GET /documents/<image_id>/preview`` which handles the
    actual file streaming.  This keeps file serving logic in one place and
    lets the existing document preview endpoint handle MIME types, path
    validation, and caching headers.

    Args:
        image_id: Document ID returned by the upload endpoint.

    Returns:
        302: Redirect to /documents/<image_id>/preview.
        404: If the document record does not exist.
    """
    try:
        doc_svc = _get_document_service()
        doc = doc_svc.get_document(image_id)
        if not doc:
            return jsonify({'error': 'Image not found'}), 404
        # Redirect to the existing document preview endpoint
        return redirect(f'/documents/{image_id}/preview')
    except Exception as e:
        logger.error(f'[CHAT IMAGE] image_file error for {image_id}: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@chat_image_bp.route('/chat/vision-capable', methods=['GET'])
@require_session
def vision_capable():
    """
    Check if a vision-capable provider is configured.

    Frontend uses this to show or hide the image attachment option.
    """
    try:
        from services.image_context_service import has_vision_provider
        available = has_vision_provider()
    except Exception:
        available = False
    return jsonify({'available': available})


# ─── Background Analysis ──────────────────────────────────────────────────────

def _run_analysis(image_id: str, image_bytes: bytes, mime_type: str):
    """
    Run image analysis in a background daemon thread and persist the result.

    On completion (success or failure) the function:
    - Writes a JSON result blob to ``chat_image_result:{image_id}`` (10 min TTL).
    - Updates ``chat_image_progress:{image_id}`` to ``'ready'`` or ``'failed'``
      so that ``GET /status`` can return the correct state even after the bytes
      key has expired (B3 fix).
    - Updates the document record: status, clean_text, summary, extracted_metadata,
      and (if possible) a summary embedding — so the result survives MemoryStore TTLs.
    - Publishes ``{"type": "image_ready", "image_id": ..., "status": ...}`` to
      the ``output:events`` pub/sub channel (Step 3 / B5 fix).  The existing
      ``_drift_sender`` thread inside the WebSocket handler forwards all messages
      on that channel to connected clients automatically.

    Args:
        image_id:    Unique identifier assigned during upload (== document ID).
        image_bytes: Raw image content (never written to disk here).
        mime_type:   MIME type string, e.g. ``'image/png'``.
    """
    result_key = _KEY_RESULT.format(image_id=image_id)
    progress_key = _KEY_PROGRESS.format(image_id=image_id)
    try:
        from services.image_context_service import analyze
        result = analyze(image_bytes, mime_type)
        store = _get_store()
        store.set(result_key, json.dumps(result), ex=_TTL_RESULT)
        # B3 fix: stamp progress key as 'ready' so status checks see completion.
        store.set(progress_key, 'ready', ex=_TTL_PROGRESS)

        # Persist result to document record
        _persist_analysis_result(image_id, result, success=True)

        # Step 3 / B5 fix: push analysis completion to connected WebSocket clients.
        store.publish('output:events', json.dumps({
            'type': 'image_ready',
            'image_id': image_id,
            'status': 'ready',
        }))
        logger.info(
            f'[CHAT IMAGE] Analysis complete image_id={image_id} '
            f'has_text={result.get("has_text")} '
            f'time={result.get("analysis_time_ms")}ms'
        )
    except Exception as e:
        logger.error(f'[CHAT IMAGE] Analysis failed image_id={image_id}: {e}', exc_info=True)
        error_result = {'error': str(e), 'description': '', 'ocr_text': '', 'has_text': False}
        store = _get_store()
        store.set(result_key, json.dumps(error_result), ex=_TTL_RESULT)
        # B3 fix: stamp progress key as 'failed' so status checks surface the error.
        store.set(progress_key, 'failed', ex=_TTL_PROGRESS)

        # Persist failure to document record
        _persist_analysis_result(image_id, error_result, success=False)

        # Step 3 / B5 fix: push failure event so the frontend can show an error badge.
        try:
            store.publish('output:events', json.dumps({
                'type': 'image_ready',
                'image_id': image_id,
                'status': 'failed',
            }))
        except Exception:
            pass  # best-effort — do not shadow the original analysis error


def _persist_analysis_result(image_id: str, result: dict, success: bool) -> None:
    """
    Update the document record with the completed (or failed) analysis result.

    Called from ``_run_analysis`` after the vision LLM call.  Non-fatal: any
    exception is logged and swallowed so the MemoryStore path always completes.

    Args:
        image_id: Document ID (== image_id returned at upload time).
        result:   Analysis result dict (``description``, ``ocr_text``, ``has_text``,
                  ``analysis_time_ms`` on success; ``error`` key on failure).
        success:  Whether analysis succeeded.
    """
    try:
        doc_svc = _get_document_service()

        if success:
            description = result.get('description', '')
            ocr_text = result.get('ocr_text', '')
            has_text = bool(result.get('has_text', False))
            analysis_time_ms = result.get('analysis_time_ms')

            extracted_metadata = {
                'description': description,
                'ocr_text': ocr_text,
                'has_text': has_text,
            }
            if analysis_time_ms is not None:
                extracted_metadata['analysis_time_ms'] = analysis_time_ms

            # Build summary: prefer description, fall back to OCR text
            summary_text = description or ocr_text or ''
            summary = summary_text[:500]

            # Optionally generate a summary embedding for semantic search
            summary_embedding = None
            if summary_text:
                try:
                    from services.embedding_service import get_embedding_service
                    emb_svc = get_embedding_service()
                    summary_embedding = emb_svc.generate_embedding(summary_text[:1000])
                except Exception as emb_err:
                    logger.debug(f'[CHAT IMAGE] Embedding generation skipped: {emb_err}')

            doc_svc.update_extracted_metadata(
                doc_id=image_id,
                metadata=extracted_metadata,
                summary=summary,
                summary_embedding=summary_embedding,
                clean_text=ocr_text or description or '',
            )
            doc_svc.update_status(image_id, 'ready')
            logger.debug(f'[CHAT IMAGE] Document record updated to ready image_id={image_id}')
        else:
            error_msg = result.get('error', 'analysis failed')
            doc_svc.update_status(image_id, 'failed', error_message=error_msg)
            logger.debug(f'[CHAT IMAGE] Document record updated to failed image_id={image_id}')

    except Exception as e:
        logger.warning(f'[CHAT IMAGE] Failed to persist analysis result for {image_id}: {e}')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_image_filename(original: str, mime: str) -> str:
    """
    Return a safe filename derived from the upload's original filename.

    Strips path separators, null bytes, control characters, and leading dots.
    Falls back to a MIME-derived name if the result is empty.

    Args:
        original: Raw filename from the multipart upload.
        mime:     MIME type string, used to derive extension as fallback.

    Returns:
        Sanitised filename string (never empty).
    """
    import re
    name = original or ''
    # Strip path separators and null bytes
    name = name.replace('/', '').replace('\\', '').replace('\x00', '')
    # Remove control characters
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)
    # Prevent directory traversal
    name = name.lstrip('.')
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    if not name:
        ext = _MIME_EXT.get(mime, '.img')
        name = f'image{ext}'

    # Limit length
    if len(name) > 255:
        ext = os.path.splitext(name)[1]
        name = name[:255 - len(ext)] + ext

    return name

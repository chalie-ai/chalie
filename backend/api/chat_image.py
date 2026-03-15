"""
Chat Image API — Upload and analyze images attached to chat messages.

Routes (all require session auth):
  POST /chat/image              — multipart upload, returns {image_id, status}
  GET  /chat/image/<id>/status  — poll analysis result
  GET  /chat/vision-capable     — capability check for frontend

Design:
  Images are stored ephemerally in MemoryStore (5min TTL for raw bytes,
  10min TTL for analysis results). They are never written to disk and never
  enter the database. Only the semantic analysis (description + OCR text)
  flows into the cognitive pipeline via the WebSocket chat handler.

  SHA-256 hash deduplication: uploading the same image twice within a session
  returns the existing image_id without re-analysis.

  Progress tracking: a lightweight ``chat_image_progress:{image_id}`` key with
  a 120 s TTL is written at upload time and updated by the analysis thread.
  This acts as a tertiary fallback for GET /status so that the endpoint never
  returns 404 while analysis is still in-flight, even after the 5-minute bytes
  key has expired (B3 fix).
"""

import hashlib
import json
import logging
import secrets
import threading

from flask import Blueprint, jsonify, request

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
_TTL_PROGRESS = 120 # 2 minutes — covers worst-case 60s analysis with margin


def _get_store():
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


# ─── Routes ──────────────────────────────────────────────────────────────────

@chat_image_bp.route('/chat/image', methods=['POST'])
@require_session
def upload_image():
    """
    Multipart image upload.

    Accepts multipart field ``image``. Stores raw bytes in MemoryStore, writes
    a ``chat_image_progress:{image_id}`` key (TTL=120 s) for in-flight tracking,
    then kicks off background analysis in a daemon thread and returns
    ``{image_id, status: 'analyzing'}`` immediately.

    Deduplication: if the SHA-256 hash of the uploaded bytes already exists the
    existing ``image_id`` is returned together with its *actual* current status
    (``'ready'``, ``'failed'``, or ``'analyzing'``) instead of the hardcoded
    ``'analyzing'`` string (B2 fix).

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

    # SHA-256 deduplication
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    store = _get_store()

    hash_key = _KEY_HASH.format(hash=image_hash)
    existing_id = store.get(hash_key)
    if existing_id:
        if isinstance(existing_id, bytes):
            existing_id = existing_id.decode()
        logger.debug(f'[CHAT IMAGE] Duplicate detected — returning existing image_id={existing_id}')
        # B2 fix: return the *actual* status rather than always hardcoding 'analyzing'.
        # Check whether the analysis result is already stored for the existing image.
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

    # New image — assign ID, store bytes, kick off analysis
    image_id = secrets.token_hex(8)

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
    4. None of the above → 404.

    Args:
        image_id: Hex token returned by the upload endpoint.

    Returns:
        200: ``{status: 'analyzing'|'ready'|'failed', result: {...}|null}``
        404: ``{error: 'Image not found'}`` when all keys are absent.
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
        # (120 s vs 300 s) and is updated by the analysis thread to 'ready'/'failed'.
        # This prevents a 404 being returned while analysis is still running after
        # the bytes key has expired.
        progress_key = _KEY_PROGRESS.format(image_id=image_id)
        progress = store.get(progress_key)
        if progress is not None:
            if isinstance(progress, bytes):
                progress = progress.decode()
            return jsonify({'status': progress, 'result': None})
        return jsonify({'error': 'Image not found'}), 404

    try:
        result = json.loads(raw)
        status = 'failed' if result.get('error') else 'ready'
        return jsonify({'status': status, 'result': result})
    except Exception:
        return jsonify({'status': 'failed', 'result': None})


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

    Args:
        image_id:    Unique identifier assigned during upload.
        image_bytes: Raw image content (never written to disk).
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
        logger.info(
            f'[CHAT IMAGE] Analysis complete image_id={image_id} '
            f'has_text={result.get("has_text")} '
            f'time={result.get("analysis_time_ms")}ms'
        )
    except Exception as e:
        logger.error(f'[CHAT IMAGE] Analysis failed image_id={image_id}: {e}', exc_info=True)
        store = _get_store()
        store.set(result_key, json.dumps({'error': str(e), 'description': '', 'ocr_text': '', 'has_text': False}), ex=_TTL_RESULT)
        # B3 fix: stamp progress key as 'failed' so status checks surface the error.
        store.set(progress_key, 'failed', ex=_TTL_PROGRESS)

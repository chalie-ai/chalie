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
_KEY_BYTES = 'chat_image:{image_id}'         # raw bytes (bytes value)
_KEY_RESULT = 'chat_image_result:{image_id}' # JSON analysis result
_KEY_HASH = 'chat_image_hash:{hash}'         # hash → image_id dedup

_TTL_BYTES = 300    # 5 minutes
_TTL_RESULT = 600   # 10 minutes
_TTL_HASH = 300     # 5 minutes


def _get_store():
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


# ─── Routes ──────────────────────────────────────────────────────────────────

@chat_image_bp.route('/chat/image', methods=['POST'])
@require_session
def upload_image():
    """
    Multipart image upload.

    Accepts field name 'image'. Stores bytes in MemoryStore, kicks off
    async analysis, returns image_id immediately.
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
        return jsonify({'image_id': existing_id, 'status': 'analyzing'}), 200

    # New image — assign ID, store bytes, kick off analysis
    image_id = secrets.token_hex(8)

    bytes_key = _KEY_BYTES.format(image_id=image_id)
    store.set(bytes_key, image_bytes, ex=_TTL_BYTES)
    store.set(hash_key, image_id, ex=_TTL_HASH)

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

    Returns:
        {status: 'analyzing'|'ready'|'failed', result: {...} | null}
    """
    store = _get_store()
    result_key = _KEY_RESULT.format(image_id=image_id)
    raw = store.get(result_key)

    if raw is None:
        # Check if we still have the raw bytes (means still analyzing)
        bytes_key = _KEY_BYTES.format(image_id=image_id)
        if store.exists(bytes_key):
            return jsonify({'status': 'analyzing', 'result': None})
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
    """Run image analysis in background thread and store result in MemoryStore."""
    result_key = _KEY_RESULT.format(image_id=image_id)
    try:
        from services.image_context_service import analyze
        result = analyze(image_bytes, mime_type)
        store = _get_store()
        store.set(result_key, json.dumps(result), ex=_TTL_RESULT)
        logger.info(
            f'[CHAT IMAGE] Analysis complete image_id={image_id} '
            f'has_text={result.get("has_text")} '
            f'time={result.get("analysis_time_ms")}ms'
        )
    except Exception as e:
        logger.error(f'[CHAT IMAGE] Analysis failed image_id={image_id}: {e}', exc_info=True)
        store = _get_store()
        store.set(result_key, json.dumps({'error': str(e), 'description': '', 'ocr_text': '', 'has_text': False}), ex=_TTL_RESULT)

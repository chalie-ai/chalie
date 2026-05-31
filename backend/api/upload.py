"""
Upload API — single multipart upload endpoint for chat attachments.

Routes (all require session auth):
  POST /upload — save file to /tmp/chalie_<uuid>.<ext>, return metadata

Design:
  Files are saved to /tmp with a unique chalie_ prefix and are NOT written
  to the database or analysed at upload time. The frontend collects the
  returned tmp_path values and passes them as an ``attachments`` array in
  the POST /chat body. The backend processes attachments (move, DB record,
  OCR/extraction) as part of the normal message processing flow.

  Cleanup: stale /tmp/chalie_* files older than 24 h are swept by
  TmpCleanupWorker (workers/tmp_cleanup_worker.py), registered in run.py.

MIME allowlist (images + documents accepted by text_extractor):
  image/jpeg, image/png, image/webp, image/gif
  application/pdf
  application/vnd.openxmlformats-officedocument.wordprocessingml.document
  application/vnd.openxmlformats-officedocument.presentationml.presentation
  text/plain, text/markdown, text/html
"""

import logging
import os
import secrets

from flask import Blueprint, jsonify, request

# safe_filename: shared werkzeug-backed sanitizer (services/filename_utils.py),
# also used by api/documents.py — single source of truth for upload filenames.
from services.filename_utils import safe_filename
from .auth import require_session

logger = logging.getLogger(__name__)

upload_bp = Blueprint('upload', __name__)

# File size limit: 50 MB (matches Flask MAX_CONTENT_LENGTH)
_MAX_FILE_BYTES = 50 * 1024 * 1024

# Accepted MIME types — must match what text_extractor can handle
_ALLOWED_MIMES: frozenset[str] = frozenset({
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain',
    'text/markdown',
    'text/html',
})

# Extension derived from MIME type when the uploaded filename has no extension
_MIME_TO_EXT: dict[str, str] = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/html': '.html',
}

def _sanitize_filename(original: str, mime: str) -> str:
    """Return a safe upload filename, falling back to a MIME-derived name.

    Delegates name hardening to ``safe_filename``; when that yields nothing
    safe, substitutes ``upload<ext>`` using the MIME-derived extension.
    """
    name = safe_filename(original)
    if not name:
        name = f'upload{_MIME_TO_EXT.get(mime, ".bin")}'
    return name


@upload_bp.route('/upload', methods=['POST'])
@require_session
def upload_file():
    """Save a multipart file to /tmp and return its metadata.

    Accepts multipart field ``file``. Validates MIME type and file size.
    Saves to ``/tmp/chalie_<uuid>.<ext>`` without creating any DB record.

    Returns:
        201: ``{tmp_path, filename, content_type, size}``
        400: missing or empty file
        413: file exceeds 50 MB
        415: unsupported MIME type
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No filename provided'}), 400

    content_type = file.content_type or 'application/octet-stream'
    mime = content_type.split(';')[0].strip().lower()

    if mime not in _ALLOWED_MIMES:
        accepted = ', '.join(sorted(_ALLOWED_MIMES))
        return jsonify({'error': f'Unsupported file type: {mime}. Accepted: {accepted}'}), 415

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({'error': 'Empty file'}), 400

    if len(file_bytes) > _MAX_FILE_BYTES:
        mb = _MAX_FILE_BYTES // 1024 // 1024
        return jsonify({'error': f'File exceeds {mb} MB limit'}), 413

    filename = _sanitize_filename(file.filename, mime)
    ext = os.path.splitext(filename)[1] or _MIME_TO_EXT.get(mime, '.bin')
    unique_id = secrets.token_hex(8)
    tmp_path = f'/tmp/chalie_{unique_id}{ext}'

    try:
        with open(tmp_path, 'wb') as fh:
            fh.write(file_bytes)
    except OSError as exc:
        logger.exception('[UPLOAD] Failed to write tmp file %s: %s', tmp_path, exc)
        return jsonify({'error': 'Failed to save file'}), 500

    logger.debug('[UPLOAD] Saved %s (%d bytes) → %s', filename, len(file_bytes), tmp_path)

    return jsonify({
        'tmp_path': tmp_path,
        'filename': filename,
        'content_type': mime,
        'size': len(file_bytes),
    }), 201

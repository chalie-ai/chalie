"""
Unit tests for the Chat Image REST API (/api/chat/image, /api/chat/vision-capable).

All tests are marked @pytest.mark.unit.
Uses Flask test client; no real vision LLM calls.
"""

import io
import json
import struct
import zlib
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def app():
    """Create a minimal Flask app with only the chat_image blueprint registered."""
    from flask import Flask
    from api.chat_image import chat_image_bp

    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.register_blueprint(chat_image_bp)

    # Bypass session auth for tests — patch validate_session because the decorator
    # is applied at import time; the lazy import inside the wrapper is the live target.
    with patch('services.auth_session_service.validate_session', return_value=True):
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_png():
    """
    Minimal valid PNG image bytes (10×10, solid blue) built from stdlib only.

    Avoids a Pillow import at fixture-time so the tests run in sandboxes where
    the Pillow C extension's shared object cannot be mmap-executed (noexec
    filesystem). The PNG is constructed by hand: PNG signature + IHDR + IDAT +
    IEND chunks, following the PNG spec (RFC 2083).
    """
    def _chunk(tag: bytes, data: bytes) -> bytes:
        """Return a length-prefixed PNG chunk with CRC appended."""
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    width, height = 10, 10
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = _chunk(b'IHDR', ihdr_data)
    # One scanline = filter-byte (0x00) + width*3 RGB bytes (0, 128, 255 = blue)
    scanline = b'\x00' + b'\x00\x80\xff' * width
    raw_data = scanline * height
    idat = _chunk(b'IDAT', zlib.compress(raw_data))
    iend = _chunk(b'IEND', b'')
    return signature + ihdr + idat + iend


@pytest.fixture
def mock_store():
    """MemoryStore mock that behaves like a dict with TTL support."""
    store = {}

    class FakeStore:
        def get(self, key):
            return store.get(key)
        def set(self, key, value, ex=None):
            store[key] = value
        def exists(self, key):
            return key in store

    return FakeStore()


# ─── POST /chat/image ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_upload_returns_image_id(client, sample_png, mock_store):
    """POST /chat/image should return an image_id and status 'analyzing'."""
    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('api.chat_image._run_analysis'):  # don't actually analyze

        data = {'image': (io.BytesIO(sample_png), 'test.png', 'image/png')}
        res = client.post('/chat/image', data=data, content_type='multipart/form-data')

    assert res.status_code == 201
    body = res.get_json()
    assert 'image_id' in body
    assert body['status'] == 'analyzing'
    assert len(body['image_id']) == 16  # token_hex(8) → 16 hex chars


@pytest.mark.unit
def test_upload_rejects_non_image_mime(client, mock_store):
    """POST /chat/image should return 415 for non-image content types."""
    pdf_bytes = b'%PDF-1.4 fake pdf content'
    with patch('api.chat_image._get_store', return_value=mock_store):
        data = {'image': (io.BytesIO(pdf_bytes), 'doc.pdf', 'application/pdf')}
        res = client.post('/chat/image', data=data, content_type='multipart/form-data')

    assert res.status_code == 415


@pytest.mark.unit
def test_upload_rejects_empty_file(client, mock_store):
    """POST /chat/image should return 400 for empty files."""
    with patch('api.chat_image._get_store', return_value=mock_store):
        data = {'image': (io.BytesIO(b''), 'empty.png', 'image/png')}
        res = client.post('/chat/image', data=data, content_type='multipart/form-data')

    assert res.status_code == 400


@pytest.mark.unit
def test_upload_rejects_oversized_file(client, mock_store):
    """POST /chat/image should return 413 for files over 10MB."""
    oversized = b'x' * (10 * 1024 * 1024 + 1)
    with patch('api.chat_image._get_store', return_value=mock_store):
        data = {'image': (io.BytesIO(oversized), 'big.png', 'image/png')}
        res = client.post('/chat/image', data=data, content_type='multipart/form-data')

    assert res.status_code == 413


@pytest.mark.unit
def test_upload_rejects_missing_file(client, mock_store):
    """POST /chat/image should return 400 when no 'image' field is present."""
    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.post('/chat/image', data={}, content_type='multipart/form-data')

    assert res.status_code == 400


@pytest.mark.unit
def test_duplicate_image_returns_same_id(client, sample_png, mock_store):
    """Uploading the same image bytes twice should return the existing image_id."""
    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('api.chat_image._run_analysis'):

        data1 = {'image': (io.BytesIO(sample_png), 'img1.png', 'image/png')}
        res1 = client.post('/chat/image', data=data1, content_type='multipart/form-data')
        id1 = res1.get_json()['image_id']

        # Second upload of same content
        data2 = {'image': (io.BytesIO(sample_png), 'img2.png', 'image/png')}
        res2 = client.post('/chat/image', data=data2, content_type='multipart/form-data')
        id2 = res2.get_json()['image_id']

    assert id1 == id2
    assert res2.status_code == 200  # 200 (not 201) for existing


# ─── GET /chat/image/<id>/status ─────────────────────────────────────────────

@pytest.mark.unit
def test_status_returns_ready_when_result_exists(client, mock_store):
    """GET /chat/image/<id>/status should return 'ready' when result is stored."""
    image_id = 'abc12345abc12345'
    result = {'description': 'A test image.', 'ocr_text': '', 'has_text': False, 'error': None}
    mock_store.set(f'chat_image_result:{image_id}', json.dumps(result))

    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.get(f'/chat/image/{image_id}/status')

    assert res.status_code == 200
    body = res.get_json()
    assert body['status'] == 'ready'
    assert body['result']['description'] == 'A test image.'


@pytest.mark.unit
def test_status_returns_analyzing_while_bytes_exist(client, mock_store):
    """GET /chat/image/<id>/status should return 'analyzing' while bytes are in store."""
    image_id = 'pending0pending0'
    mock_store.set(f'chat_image:{image_id}', b'fake image bytes')  # bytes present, no result yet

    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.get(f'/chat/image/{image_id}/status')

    assert res.status_code == 200
    body = res.get_json()
    assert body['status'] == 'analyzing'
    assert body['result'] is None


@pytest.mark.unit
def test_status_returns_failed_on_error_result(client, mock_store):
    """GET /chat/image/<id>/status should return 'failed' when result has error key."""
    image_id = 'fail0000fail0000'
    error_result = {'error': 'No vision provider', 'description': '', 'ocr_text': '', 'has_text': False}
    mock_store.set(f'chat_image_result:{image_id}', json.dumps(error_result))

    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.get(f'/chat/image/{image_id}/status')

    body = res.get_json()
    assert body['status'] == 'failed'


@pytest.mark.unit
def test_status_returns_404_for_unknown_id(client, mock_store):
    """GET /chat/image/<id>/status should return 404 for unknown image_id."""
    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.get('/chat/image/notexist123/status')

    assert res.status_code == 404


@pytest.mark.unit
def test_status_returns_analyzing_after_bytes_expire(client, mock_store):
    """
    GET /chat/image/<id>/status must return HTTP 200 with status='analyzing'
    when the bytes key has already expired (absent) but the progress key is
    still present with value 'analyzing' (B3 fix).

    Simulates the window where analysis is still in-flight but the 5-minute
    bytes TTL has elapsed before the 10-minute progress TTL.
    """
    image_id = 'b3test000b3test0'
    # Bytes key intentionally absent — simulates TTL expiry.
    # Progress key still alive (within its 10 min window).
    mock_store.set(f'chat_image_progress:{image_id}', 'analyzing')

    with patch('api.chat_image._get_store', return_value=mock_store):
        res = client.get(f'/chat/image/{image_id}/status')

    assert res.status_code == 200
    body = res.get_json()
    assert body['status'] == 'analyzing'
    assert body['result'] is None


@pytest.mark.unit
def test_duplicate_returns_ready_when_analysis_complete(client, sample_png, mock_store):
    """
    Uploading a duplicate image must return status='ready' (not the hardcoded
    'analyzing') when the original analysis result is already stored (B2 fix).

    Workflow:
      1. Upload the image once → image_id assigned, _run_analysis skipped.
      2. Manually write a completed result into mock_store to simulate finished
         analysis.
      3. Upload the same bytes again → dedup hit → should return status='ready'.
    """
    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('api.chat_image._run_analysis'):

        # First upload — creates the hash key in mock_store.
        data1 = {'image': (io.BytesIO(sample_png), 'img1.png', 'image/png')}
        res1 = client.post('/chat/image', data=data1, content_type='multipart/form-data')
        assert res1.status_code == 201
        image_id = res1.get_json()['image_id']

    # Simulate completed analysis by writing the result key directly.
    completed_result = {
        'description': 'A small blue square.',
        'ocr_text': '',
        'has_text': False,
        'error': None,
    }
    mock_store.set(f'chat_image_result:{image_id}', json.dumps(completed_result))

    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('api.chat_image._run_analysis'):

        # Second upload — should be a dedup hit returning actual status.
        data2 = {'image': (io.BytesIO(sample_png), 'img2.png', 'image/png')}
        res2 = client.post('/chat/image', data=data2, content_type='multipart/form-data')

    assert res2.status_code == 200
    body = res2.get_json()
    assert body['image_id'] == image_id
    assert body['status'] == 'ready'


# ─── GET /chat/vision-capable ─────────────────────────────────────────────────

@pytest.mark.unit
def test_vision_capable_true_when_provider_exists(client):
    """GET /chat/vision-capable returns {available: true} when provider configured."""
    with patch('services.image_context_service.has_vision_provider', return_value=True):
        res = client.get('/chat/vision-capable')

    assert res.status_code == 200
    assert res.get_json() == {'available': True}


@pytest.mark.unit
def test_vision_capable_false_when_no_provider(client):
    """GET /chat/vision-capable returns {available: false} when no provider."""
    with patch('services.image_context_service.has_vision_provider', return_value=False):
        res = client.get('/chat/vision-capable')

    assert res.status_code == 200
    assert res.get_json() == {'available': False}


@pytest.mark.unit
def test_vision_capable_false_on_import_error(client):
    """GET /chat/vision-capable returns {available: false} if the service raises."""
    with patch('services.image_context_service.has_vision_provider', side_effect=RuntimeError('PIL not found')):
        res = client.get('/chat/vision-capable')

    # Even on error, endpoint should not 500
    assert res.status_code == 200
    assert res.get_json() == {'available': False}

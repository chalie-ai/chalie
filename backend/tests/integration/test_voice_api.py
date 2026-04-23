"""
Integration: voice API — synthesize (pub/sub streaming) + transcribe + health.

Tests exercise the real Kokoro + Moonshine models.
Each test skips with pytest.skip() if /voice/health reports models unavailable,
so the suite stays green in environments without voice dependencies installed.

Uses the authed_client fixture from conftest.py (real Flask app, auth bypassed,
real SQLite + MemoryStore). The MemoryStore is the same instance seen by
MemoryClientService.create_connection inside the backend, so the test can
subscribe to ``output:events`` and receive the tts_chunk/tts_done frames
published by the synthesize background thread.
"""

import base64
import io
import json
import time

import pytest


def _subscribe_tts_events(store):
    """Subscribe to ``output:events`` before kicking off synthesis.

    Must be called before ``client.post('/voice/synthesize', ...)`` —
    the backend publishes asynchronously in a daemon thread, so a late
    subscription would drop early chunks.
    """
    pubsub = store.pubsub()
    pubsub.subscribe('output:events')
    return pubsub


def _drain_tts_events(pubsub, timeout: float = 60.0):
    """Pull tts_chunk/tts_done frames off the already-subscribed pubsub.

    Returns ``(chunks, done_seen)`` where chunks is a list of the decoded
    JSON payloads (each has ``audio``, ``text``, ``index``, ``total``).
    Raises TimeoutError if tts_done never arrives.
    """
    chunks: list[dict] = []
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            msg = pubsub.get_message(timeout=1.0)
            if not msg or msg.get('type') != 'message':
                continue
            try:
                data = json.loads(msg['data'])
            except (TypeError, json.JSONDecodeError):
                continue
            if data.get('type') == 'tts_chunk':
                chunks.append(data)
            elif data.get('type') == 'tts_done':
                return chunks, True
        raise TimeoutError(f'tts_done not received within {timeout}s')
    finally:
        try:
            pubsub.unsubscribe('output:events')
            pubsub.close()
        except Exception:
            pass


def _voice_available(client) -> bool:
    """Return True only if /voice/health reports status='ok'."""
    resp = client.get('/voice/health')
    if resp.status_code != 200:
        return False
    data = resp.get_json()
    return (data or {}).get('status') == 'ok'


def _decode_chunk_wav(chunk: dict) -> bytes:
    """Base64-decode the ``audio`` field of a tts_chunk frame."""
    return base64.b64decode(chunk['audio'])


@pytest.mark.integration
def test_health_endpoint_shape(authed_client):
    """GET /voice/health → JSON with 'status' key holding a known string."""
    client, _db, _store = authed_client

    resp = client.get('/voice/health')
    # 200 when voice is available (ok or loading); 503 when unavailable
    assert resp.status_code in (200, 503)

    data = resp.get_json()
    assert data is not None
    assert 'status' in data
    assert data['status'] in ('ok', 'loading', 'unavailable')


@pytest.mark.integration
def test_synthesize_streams_tts_chunks(authed_client):
    """POST /voice/synthesize returns {ok, total} + publishes tts_chunk/tts_done."""
    client, _db, store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    pubsub = _subscribe_tts_events(store)

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Hello world.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body is not None
    assert body.get('ok') is True
    assert isinstance(body.get('total'), int) and body['total'] >= 1

    chunks, done = _drain_tts_events(pubsub, timeout=60.0)
    assert done, 'tts_done frame never arrived'
    assert len(chunks) == body['total']

    # First chunk must be a valid WAV blob
    wav_bytes = _decode_chunk_wav(chunks[0])
    assert len(wav_bytes) > 44, 'Chunk too short to be a WAV file'
    assert wav_bytes[:4] == b'RIFF', 'Missing RIFF header'
    assert wav_bytes[8:12] == b'WAVE', 'Missing WAVE marker'

    try:
        import soundfile as sf
        data, samplerate = sf.read(io.BytesIO(wav_bytes))
        duration = len(data) / samplerate
        assert duration > 0.1, f'Audio too short: {duration:.3f}s'
    except ImportError:
        pass  # soundfile not installed — skip duration check


@pytest.mark.integration
@pytest.mark.parametrize(
    'text',
    ['', '   ', '\n\t  \n'],
    ids=['empty', 'whitespace', 'newlines-tabs'],
)
def test_synthesize_rejects_empty_text(authed_client, text):
    """POST /voice/synthesize → 400 for empty or whitespace-only input."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': text},
        content_type='application/json',
    )
    assert resp.status_code == 400

    data = resp.get_json()
    assert data is not None
    assert 'error' in data


@pytest.mark.integration
def test_synthesize_long_text_splits_into_multiple_chunks(authed_client):
    """Long text (>800 chars) splits across multiple tts_chunk frames."""
    client, _db, store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    # Pad well past _MAX_CHUNK_CHARS=800 to force multi-chunk output.
    sentence = (
        'The quick brown fox jumps over the lazy dog. '
        'She sells sea shells by the sea shore. '
        'How much wood would a woodchuck chuck if a woodchuck could chuck wood. '
        'Peter Piper picked a peck of pickled peppers. '
        'All that glitters is not gold and not all who wander are lost. '
    )
    long_text = sentence * 4  # ~1150 chars

    pubsub = _subscribe_tts_events(store)

    resp = client.post(
        '/voice/synthesize',
        json={'text': long_text},
        content_type='application/json',
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body.get('ok') is True
    assert body.get('total', 0) > 1, 'Expected multi-chunk split for >800 char text'

    chunks, done = _drain_tts_events(pubsub, timeout=120.0)
    assert done
    assert len(chunks) == body['total']

    # Each chunk must be a valid WAV
    for i, chunk in enumerate(chunks):
        wav = _decode_chunk_wav(chunk)
        assert wav[:4] == b'RIFF', f'Chunk {i} missing RIFF'
        assert wav[8:12] == b'WAVE', f'Chunk {i} missing WAVE'

    # Combined duration should be comfortably above 3s
    try:
        import soundfile as sf
        total_duration = 0.0
        for chunk in chunks:
            wav = _decode_chunk_wav(chunk)
            data, samplerate = sf.read(io.BytesIO(wav))
            total_duration += len(data) / samplerate
        assert total_duration > 3.0, (
            f'Expected > 3s cumulative for {body["total"]} chunks, got {total_duration:.3f}s'
        )
    except ImportError:
        pass  # soundfile not installed — skip duration check


@pytest.mark.integration
def test_transcribe_roundtrip(authed_client):
    """Synthesize 'hello chalie' → feed the chunk WAV back to /voice/transcribe."""
    client, _db, store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    pubsub = _subscribe_tts_events(store)

    synth_resp = client.post(
        '/voice/synthesize',
        json={'text': 'hello chalie'},
        content_type='application/json',
    )
    assert synth_resp.status_code == 200
    assert synth_resp.get_json().get('ok') is True

    chunks, done = _drain_tts_events(pubsub, timeout=60.0)
    assert done
    assert len(chunks) >= 1

    # 'hello chalie' is one sentence — single chunk carries the full audio.
    wav_bytes = _decode_chunk_wav(chunks[0])

    transcribe_resp = client.post(
        '/voice/transcribe',
        data={'file': (io.BytesIO(wav_bytes), 'test.wav')},
        content_type='multipart/form-data',
    )
    assert transcribe_resp.status_code == 200

    data = transcribe_resp.get_json()
    assert data is not None
    assert 'text' in data

    text = (data.get('text') or '').lower()
    assert 'hello' in text, f"Expected 'hello' in transcript, got: {data.get('text')!r}"
